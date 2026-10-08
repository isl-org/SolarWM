#!/usr/bin/env python3
"""Benchmark a custom fused Triton QKV unpack + RMSNorm kernel.

Does a fused epilogue kernel outperform PyTorch chunk + norm?
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import triton
import triton.language as tl

from solarwm.backends.wan22.runtime.modeling.model import WanRMSNorm
from solarwm.kernels.dit_fused.fused_dit_ops import triton_rmsnorm


@triton.jit
def _fused_qkv_norm_unpack_kernel(
    QKV_ptr,
    WQ_ptr,
    WK_ptr,
    Q_ptr,
    K_ptr,
    V_ptr,
    M: int,
    C: int,
    eps: float,
    stride_qkv_m: int,
    stride_qkv_c: int,
    stride_out_m: int,
    stride_out_c: int,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_C)
    c_mask = cols < C

    row_qkv = QKV_ptr + row * stride_qkv_m
    row_q = Q_ptr + row * stride_out_m
    row_k = K_ptr + row * stride_out_m
    row_v = V_ptr + row * stride_out_m

    # 1. Process Q
    q_ptrs = row_qkv + cols * stride_qkv_c
    q = tl.load(q_ptrs, mask=c_mask, other=0.0).to(tl.float32)
    q_sum_sq = tl.sum(q * q, axis=0)
    q_rstd = 1.0 / tl.sqrt(q_sum_sq / C + eps)
    wq = tl.load(WQ_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)
    q_out = (q * q_rstd) * wq
    tl.store(row_q + cols * stride_out_c, q_out.to(tl.bfloat16), mask=c_mask)

    # 2. Process K
    k_ptrs = row_qkv + (C + cols) * stride_qkv_c
    k = tl.load(k_ptrs, mask=c_mask, other=0.0).to(tl.float32)
    k_sum_sq = tl.sum(k * k, axis=0)
    k_rstd = 1.0 / tl.sqrt(k_sum_sq / C + eps)
    wk = tl.load(WK_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)
    k_out = (k * k_rstd) * wk
    tl.store(row_k + cols * stride_out_c, k_out.to(tl.bfloat16), mask=c_mask)

    # 3. Process V (pure contiguous copy, no norm)
    v_ptrs = row_qkv + (2 * C + cols) * stride_qkv_c
    v = tl.load(v_ptrs, mask=c_mask, other=0.0)
    tl.store(row_v + cols * stride_out_c, v, mask=c_mask)


def triton_fused_qkv_norm_unpack(qkv: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, eps: float = 1e-6):
    b, s, total_c = qkv.shape
    c = total_c // 3
    m = b * s
    qkv_2d = qkv.view(m, total_c)
    
    q_out = torch.empty(m, c, device=qkv.device, dtype=qkv.dtype)
    k_out = torch.empty(m, c, device=qkv.device, dtype=qkv.dtype)
    v_out = torch.empty(m, c, device=qkv.device, dtype=qkv.dtype)

    block_c = 1 << (c - 1).bit_length()
    grid = (m,)

    _fused_qkv_norm_unpack_kernel[grid](
        qkv_2d,
        wq,
        wk,
        q_out,
        k_out,
        v_out,
        m,
        c,
        float(eps),
        qkv_2d.stride(0),
        qkv_2d.stride(1),
        q_out.stride(0),
        q_out.stride(1),
        BLOCK_C=block_c,
    )
    return q_out.view(b, s, c), k_out.view(b, s, c), v_out.view(b, s, c)


def time_fn(fn, warmup=30, iters=150):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def test_triton_fused():
    device = "xpu"
    dtype = torch.bfloat16
    dim = 3072
    num_heads = 24
    head_dim = 128
    s = 1215
    eps = 1e-6

    x = torch.randn(1, s, dim, device=device, dtype=dtype)
    q_proj = nn.Linear(dim, dim, device=device, dtype=dtype)
    k_proj = nn.Linear(dim, dim, device=device, dtype=dtype)
    v_proj = nn.Linear(dim, dim, device=device, dtype=dtype)
    qkv_proj = nn.Linear(dim, 3 * dim, device=device, dtype=dtype)

    with torch.no_grad():
        qkv_proj.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
        qkv_proj.bias.copy_(torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0))

    norm_q = WanRMSNorm(dim, eps=eps).to(device=device, dtype=dtype)
    norm_k = WanRMSNorm(dim, eps=eps).to(device=device, dtype=dtype)

    # 1. Baseline: 3x Linear + Triton RMSNorm
    def baseline():
        q = triton_rmsnorm(q_proj(x), norm_q.weight, eps).view(1, s, num_heads, head_dim)
        k = triton_rmsnorm(k_proj(x), norm_k.weight, eps).view(1, s, num_heads, head_dim)
        v = v_proj(x).view(1, s, num_heads, head_dim)
        return q, k, v

    # 2. Fused QKV + Single Triton Epilogue Kernel
    def fused_triton_epilogue():
        qkv = qkv_proj(x)
        q, k, v = triton_fused_qkv_norm_unpack(qkv, norm_q.weight, norm_k.weight, eps)
        return (
            q.view(1, s, num_heads, head_dim),
            k.view(1, s, num_heads, head_dim),
            v.view(1, s, num_heads, head_dim),
        )

    # Parity check
    q_b, k_b, v_b = baseline()
    q_f, k_f, v_f = fused_triton_epilogue()
    print("Verification:")
    print("  max |q_b - q_f|:", (q_b - q_f).abs().max().item())
    print("  max |k_b - k_f|:", (k_b - k_f).abs().max().item())
    print("  max |v_b - v_f|:", (v_b - v_f).abs().max().item())
    print("  v_f is_contiguous:", v_f.is_contiguous())

    t_b = time_fn(baseline)
    t_f = time_fn(fused_triton_epilogue)
    print(f"\nBaseline (3x Linear + Triton RMSNorm):         {t_b:.3f} ms")
    print(f"Fused QKV (1x Linear + Fused Triton Epilogue): {t_f:.3f} ms (speedup: {t_b/t_f:.2f}x, save: {t_b - t_f:+.3f} ms)")


if __name__ == "__main__":
    test_triton_fused()
