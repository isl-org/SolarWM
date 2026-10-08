#!/usr/bin/env python3
"""Detailed investigation of QKV projection and downstream impacts.

Investigates:
1. Baseline vs Fused GEMM isolation across different sequence lengths:
   S in [405, 880, 1215, 2430, 4860]
2. Baseline + WanRMSNorm vs Baseline + Triton RMSNorm vs Fused equivalents
3. The cost of strided slices on downstream operations:
   - Does v.contiguous() get triggered?
   - Cost of v.contiguous() on XPU
   - Full QKV -> RoPE/PRoPE -> Attention pipeline comparison
4. torch.compile impact on Baseline vs Fused
5. Memory allocation analysis (peak transient VRAM)
"""

from __future__ import annotations

import gc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from solarwm.backends.wan22.runtime.modeling.model import WanRMSNorm
from solarwm.kernels.dit_fused.fused_dit_ops import triton_rmsnorm


def synchronize(device: str = "xpu") -> None:
    torch.xpu.synchronize()


def time_fn(fn, warmup: int = 40, iters: int = 150) -> float:
    for _ in range(warmup):
        fn()
    synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def test_qkv_deep_dive():
    device = "xpu"
    dtype = torch.bfloat16
    dim = 3072
    num_heads = 24
    head_dim = 128
    eps = 1e-6

    print("======================================================================")
    print("1. SWEEP ACROSS SEQUENCE LENGTHS: GEMM-ONLY (Baseline 3x vs Fused 1x)")
    print("======================================================================")
    
    # Linear modules
    q_proj = nn.Linear(dim, dim, device=device, dtype=dtype)
    k_proj = nn.Linear(dim, dim, device=device, dtype=dtype)
    v_proj = nn.Linear(dim, dim, device=device, dtype=dtype)
    qkv_proj = nn.Linear(dim, 3 * dim, device=device, dtype=dtype)

    with torch.no_grad():
        qkv_proj.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
        qkv_proj.bias.copy_(torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0))

    seq_lengths = [405, 880, 1215, 2430, 3645, 4860]
    for s in seq_lengths:
        x = torch.randn(1, s, dim, device=device, dtype=dtype)
        
        # 3 separate GEMMs
        def run_3_gemm():
            q = q_proj(x)
            k = k_proj(x)
            v = v_proj(x)
            return q, k, v

        # 1 fused GEMM
        def run_1_gemm():
            return qkv_proj(x)

        t3 = time_fn(run_3_gemm)
        t1 = time_fn(run_1_gemm)
        speedup = t3 / t1
        delta_us = (t3 - t1) * 1000.0
        print(f"S={s:4d} | 3x GEMM: {t3:6.3f} ms | 1x GEMM: {t1:6.3f} ms | Speedup: {speedup:5.2f}x | Diff: {delta_us:+6.1f} us")

    print("\n======================================================================")
    print("2. COMPLETE PROJECTION + NORMALIZATION COMPARISON (S=1215)")
    print("======================================================================")
    s = 1215
    x = torch.randn(1, s, dim, device=device, dtype=dtype)
    norm_q = WanRMSNorm(dim, eps=eps).to(device=device, dtype=dtype)
    norm_k = WanRMSNorm(dim, eps=eps).to(device=device, dtype=dtype)

    # Variant 1: Baseline (3x GEMM + PyTorch WanRMSNorm)
    def v1_baseline_pyt():
        q = norm_q(q_proj(x)).view(1, s, num_heads, head_dim)
        k = norm_k(k_proj(x)).view(1, s, num_heads, head_dim)
        v = v_proj(x).view(1, s, num_heads, head_dim)
        return q, k, v

    # Variant 2: Baseline (3x GEMM + Triton WanRMSNorm)
    def v2_baseline_triton():
        q = triton_rmsnorm(q_proj(x), norm_q.weight, eps).view(1, s, num_heads, head_dim)
        k = triton_rmsnorm(k_proj(x), norm_k.weight, eps).view(1, s, num_heads, head_dim)
        v = v_proj(x).view(1, s, num_heads, head_dim)
        return q, k, v

    # Variant 3: Fused (1x GEMM + chunk + PyTorch WanRMSNorm)
    def v3_fused_chunk_pyt():
        q_raw, k_raw, v_raw = qkv_proj(x).chunk(3, dim=-1)
        q = norm_q(q_raw).view(1, s, num_heads, head_dim)
        k = norm_k(k_raw).view(1, s, num_heads, head_dim)
        v = v_raw.view(1, s, num_heads, head_dim)
        return q, k, v

    # Variant 4: Fused (1x GEMM + chunk + Triton WanRMSNorm)
    def v4_fused_chunk_triton():
        q_raw, k_raw, v_raw = qkv_proj(x).chunk(3, dim=-1)
        q = triton_rmsnorm(q_raw, norm_q.weight, eps).view(1, s, num_heads, head_dim)
        k = triton_rmsnorm(k_raw, norm_k.weight, eps).view(1, s, num_heads, head_dim)
        v = v_raw.view(1, s, num_heads, head_dim)
        return q, k, v

    # Variant 5: Fused + contiguous v (if downstream requires contiguous v)
    def v5_fused_chunk_triton_contiguous_v():
        q_raw, k_raw, v_raw = qkv_proj(x).chunk(3, dim=-1)
        q = triton_rmsnorm(q_raw, norm_q.weight, eps).view(1, s, num_heads, head_dim)
        k = triton_rmsnorm(k_raw, norm_k.weight, eps).view(1, s, num_heads, head_dim)
        v = v_raw.contiguous().view(1, s, num_heads, head_dim)
        return q, k, v

    t_v1 = time_fn(v1_baseline_pyt)
    t_v2 = time_fn(v2_baseline_triton)
    t_v3 = time_fn(v3_fused_chunk_pyt)
    t_v4 = time_fn(v4_fused_chunk_triton)
    t_v5 = time_fn(v5_fused_chunk_triton_contiguous_v)

    print(f"Variant 1 (Baseline 3x GEMM + PyTorch Norm):          {t_v1:6.3f} ms (ref)")
    print(f"Variant 2 (Baseline 3x GEMM + Triton Norm):           {t_v2:6.3f} ms (speedup: {t_v1/t_v2:.2f}x, save: {t_v1-t_v2:+.3f} ms)")
    print(f"Variant 3 (Fused 1x GEMM + PyTorch Norm):             {t_v3:6.3f} ms (speedup: {t_v1/t_v3:.2f}x, save: {t_v1-t_v3:+.3f} ms)")
    print(f"Variant 4 (Fused 1x GEMM + Triton Norm):              {t_v4:6.3f} ms (speedup: {t_v1/t_v4:.2f}x, save: {t_v1-t_v4:+.3f} ms)")
    print(f"Variant 5 (Fused 1x GEMM + Triton Norm + v.contig()): {t_v5:6.3f} ms (speedup: {t_v1/t_v5:.2f}x, save: {t_v1-t_v5:+.3f} ms)")

    print(f"\nDirect Comparison under SAME Norm:")
    print(f"  PyTorch Norm: Baseline {t_v1:.3f} ms vs Fused {t_v3:.3f} ms -> Delta: {t_v1 - t_v3:+.3f} ms")
    print(f"  Triton  Norm: Baseline {t_v2:.3f} ms vs Fused {t_v4:.3f} ms -> Delta: {t_v2 - t_v4:+.3f} ms")
    print(f"  Cost of v.contiguous(): {t_v5 - t_v4:.3f} ms ({((t_v5 - t_v4)*1000):.1f} us)")

    print("\n======================================================================")
    print("3. DOWNSTREAM ATTENTION PIPELINE IMPACT")
    print("======================================================================")
    # Check what happens to v in fused_rope_prope_sdpa
    from solarwm.kernels.fused_rope_prope_sage.fused_sdpa import rope_prope_transform

    q_contig = torch.randn(1, s, num_heads, head_dim, device=device, dtype=dtype)
    v_contig = torch.randn(1, s, num_heads, head_dim, device=device, dtype=dtype)

    # Simulate strided v from fused chunk
    qkv = torch.randn(1, s, 3 * dim, device=device, dtype=dtype)
    _, _, v_strided_raw = qkv.chunk(3, dim=-1)
    v_strided = v_strided_raw.view(1, s, num_heads, head_dim)

    print(f"v_contig.is_contiguous(): {v_contig.is_contiguous()}")
    print(f"v_strided.is_contiguous(): {v_strided.is_contiguous()}")

    p_inv = torch.eye(3, 4, device=device, dtype=torch.float32).unsqueeze(0)  # [1, 3, 4]
    
    # Time rope_prope_transform on contiguous v vs strided v
    t_rope_v_contig = time_fn(lambda: rope_prope_transform(v_contig, p_inv, None, out_layout="HND"))
    t_rope_v_strided = time_fn(lambda: rope_prope_transform(v_strided, p_inv, None, out_layout="HND"))
    print(f"rope_prope_transform(v_contig):  {t_rope_v_contig:6.3f} ms")
    print(f"rope_prope_transform(v_strided): {t_rope_v_strided:6.3f} ms (slowdown due to non-contiguous copy: +{(t_rope_v_strided - t_rope_v_contig)*1000:.1f} us)")

    print("\n======================================================================")
    print("4. PEAK MEMORY ALLOCATION (TRANSIENT BUFFERS)")
    print("======================================================================")
    torch.xpu.empty_cache()
    torch.xpu.reset_peak_memory_stats()
    _ = v1_baseline_pyt()
    mem_base = torch.xpu.max_memory_allocated() / (1024 * 1024)

    torch.xpu.empty_cache()
    torch.xpu.reset_peak_memory_stats()
    _ = v3_fused_chunk_pyt()
    mem_fused = torch.xpu.max_memory_allocated() / (1024 * 1024)

    print(f"Peak memory during Baseline QKV: {mem_base:.2f} MB")
    print(f"Peak memory during Fused QKV:    {mem_fused:.2f} MB (difference: {mem_fused - mem_base:+.2f} MB)")


if __name__ == "__main__":
    test_qkv_deep_dive()
