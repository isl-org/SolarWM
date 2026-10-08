#!/usr/bin/env python3
"""Benchmark QKV projection variants for Wan2.2 DiT on Intel Arc XPU.

Compares:
1. Baseline: 3 separate nn.Linear projections (q, k, v)
2. Fused QKV Option A: Single Linear + chunk(3, dim=-1)
3. Fused QKV Option B: Single Linear + split(3072, dim=-1)
4. Fused QKV Option C: Single Linear + slice indexing [..., 0:3072], etc.
5. Fused QKV Option D: Single Linear + view(B, S, 3, N, D).unbind(2)
6. Fused QKV Option E: Single Linear + chunk + triton_rmsnorm

Tests at:
- dim = 3072, num_heads = 24, head_dim = 128
- dtype = torch.bfloat16
- shapes: S in [405, 1215] (B=1)
"""

from __future__ import annotations

import gc
import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from solarwm.backends.wan22.runtime.modeling.model import WanRMSNorm
from solarwm.kernels.dit_fused.fused_dit_ops import triton_rmsnorm


def synchronize(device: str) -> None:
    if device.startswith("xpu"):
        torch.xpu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def time_fn(fn: Callable[[], None], device: str, warmup: int = 50, iters: int = 200) -> float:
    # Warmup
    for _ in range(warmup):
        fn()
    synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    end = time.perf_counter()

    return (end - start) * 1000.0 / iters  # latency in ms


class BaselineQKV(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    def gemm_only(self, x: torch.Tensor):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return q, k, v


class FusedQKVChunk(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q_raw, k_raw, v_raw = self.qkv(x).chunk(3, dim=-1)
        q = self.norm_q(q_raw).view(b, s, n, d)
        k = self.norm_k(k_raw).view(b, s, n, d)
        v = v_raw.view(b, s, n, d)
        return q, k, v

    def gemm_only(self, x: torch.Tensor):
        return self.qkv(x)


class FusedQKVSplit(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q_raw, k_raw, v_raw = torch.split(self.qkv(x), self.dim, dim=-1)
        q = self.norm_q(q_raw).view(b, s, n, d)
        k = self.norm_k(k_raw).view(b, s, n, d)
        v = v_raw.view(b, s, n, d)
        return q, k, v


class FusedQKVSlice(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        qkv = self.qkv(x)
        d_ = self.dim
        q = self.norm_q(qkv[..., :d_]).view(b, s, n, d)
        k = self.norm_k(qkv[..., d_:2*d_]).view(b, s, n, d)
        v = qkv[..., 2*d_:].view(b, s, n, d)
        return q, k, v


class FusedQKVUnbind(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        # Reshape to [b, s, 3, n, d] and unbind along dimension 2
        qkv = self.qkv(x).view(b, s, 3, self.dim)
        q_raw, k_raw, v_raw = qkv.unbind(dim=2)
        q = self.norm_q(q_raw).view(b, s, n, d)
        k = self.norm_k(k_raw).view(b, s, n, d)
        v = v_raw.view(b, s, n, d)
        return q, k, v


class FusedQKVTritonNorm(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.norm_q_weight = nn.Parameter(torch.ones(dim))
        self.norm_k_weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q_raw, k_raw, v_raw = self.qkv(x).chunk(3, dim=-1)
        q = triton_rmsnorm(q_raw, self.norm_q_weight, self.eps).view(b, s, n, d)
        k = triton_rmsnorm(k_raw, self.norm_k_weight, self.eps).view(b, s, n, d)
        v = v_raw.view(b, s, n, d)
        return q, k, v


def run_benchmark(device: str = "xpu"):
    print(f"=== QKV Projection Benchmark on {device} ===")
    print(f"Device name: {torch.xpu.get_device_name(0) if device.startswith('xpu') else 'CPU'}")
    
    dim = 3072
    num_heads = 24
    dtype = torch.bfloat16
    eps = 1e-6

    # Test sequence lengths:
    # 405 = 1 latent frame
    # 1215 = 3 latent frames (Stage 2 chunk)
    seq_lens = [405, 1215]

    for s in seq_lens:
        b = 1
        print(f"\n------------------------------------------------------------")
        print(f"Configuration: Batch={b}, SeqLen={s} (tokens), Dim={dim}, Heads={num_heads}, Dtype={dtype}")
        print(f"Tokens: {s}, GEMM sizes: M={s}, K={dim}, N={dim} (x3) vs N={3*dim} (x1)")
        print(f"------------------------------------------------------------")

        x = torch.randn(b, s, dim, device=device, dtype=dtype)

        # Initialize modules
        base = BaselineQKV(dim, num_heads, eps=eps).to(device=device, dtype=dtype)
        fused_chunk = FusedQKVChunk(dim, num_heads, eps=eps).to(device=device, dtype=dtype)
        fused_split = FusedQKVSplit(dim, num_heads, eps=eps).to(device=device, dtype=dtype)
        fused_slice = FusedQKVSlice(dim, num_heads, eps=eps).to(device=device, dtype=dtype)
        fused_unbind = FusedQKVUnbind(dim, num_heads, eps=eps).to(device=device, dtype=dtype)
        fused_triton = FusedQKVTritonNorm(dim, num_heads, eps=eps).to(device=device, dtype=dtype)

        # Copy baseline weights to fused models for exact equivalence
        with torch.no_grad():
            W_qkv = torch.cat([base.q.weight, base.k.weight, base.v.weight], dim=0)
            b_qkv = torch.cat([base.q.bias, base.k.bias, base.v.bias], dim=0)
            for m in [fused_chunk, fused_split, fused_slice, fused_unbind, fused_triton]:
                m.qkv.weight.copy_(W_qkv)
                m.qkv.bias.copy_(b_qkv)
            fused_triton.norm_q_weight.copy_(base.norm_q.weight)
            fused_triton.norm_k_weight.copy_(base.norm_k.weight)

        # Verify numerical parity
        with torch.no_grad():
            q0, k0, v0 = base(x)
            q1, k1, v1 = fused_chunk(x)
            q2, k2, v2 = fused_triton(x)

            diff_q = (q0 - q1).abs().max().item()
            diff_k = (k0 - k1).abs().max().item()
            diff_v = (v0 - v1).abs().max().item()
            print(f"Numerical verification (Baseline vs Fused Chunk):")
            print(f"  max |q0 - q1|: {diff_q:.2e}, |k0 - k1|: {diff_k:.2e}, |v0 - v1|: {diff_v:.2e}")
            diff_qt = (q0 - q2).abs().max().item()
            print(f"Numerical verification (Baseline vs Fused TritonNorm):")
            print(f"  max |q0 - q_triton|: {diff_qt:.2e}")

        # Benchmark GEMM-only
        print(f"\n--- Part 1: GEMM-only Latency (Linear execution without unpacking/norm) ---")
        t_base_gemm = time_fn(lambda: base.gemm_only(x), device)
        t_fused_gemm = time_fn(lambda: fused_chunk.gemm_only(x), device)
        speedup_gemm = t_base_gemm / t_fused_gemm
        print(f"Baseline (3x Linear [K=3072, N=3072]): {t_base_gemm:6.3f} ms")
        print(f"Fused    (1x Linear [K=3072, N=9216]): {t_fused_gemm:6.3f} ms (speedup: {speedup_gemm:5.2f}x, save: {t_base_gemm - t_fused_gemm:+.3f} ms)")

        # Benchmark Full Projection (GEMM + Unpack + Norm + Reshape)
        print(f"\n--- Part 2: Full QKV Projection Latency (GEMM + Split/Unpack + Norms + Reshape) ---")
        t_base = time_fn(lambda: base(x), device)
        t_chunk = time_fn(lambda: fused_chunk(x), device)
        t_split = time_fn(lambda: fused_split(x), device)
        t_slice = time_fn(lambda: fused_slice(x), device)
        t_unbind = time_fn(lambda: fused_unbind(x), device)
        t_triton = time_fn(lambda: fused_triton(x), device)

        print(f"Baseline (3x Linear + 2x WanRMSNorm):      {t_base:6.3f} ms  (1.00x)")
        print(f"Fused + chunk(3, dim=-1):                 {t_chunk:6.3f} ms  ({t_base / t_chunk:5.2f}x, save: {t_base - t_chunk:+.3f} ms)")
        print(f"Fused + split(dim, dim=-1):               {t_split:6.3f} ms  ({t_base / t_split:5.2f}x, save: {t_base - t_split:+.3f} ms)")
        print(f"Fused + slice indexing [..., :d]:         {t_slice:6.3f} ms  ({t_base / t_slice:5.2f}x, save: {t_base - t_slice:+.3f} ms)")
        print(f"Fused + view(B,S,3,D).unbind(2):          {t_unbind:6.3f} ms  ({t_base / t_unbind:5.2f}x, save: {t_base - t_unbind:+.3f} ms)")
        print(f"Fused + chunk + triton_rmsnorm:           {t_triton:6.3f} ms  ({t_base / t_triton:5.2f}x, save: {t_base - t_triton:+.3f} ms)")

        # Full DiT impact calculation
        # 30 layers in Wan2.2 5B
        # 5 forward passes per chunk (4 denoise + 1 commit) = 150 projections per chunk
        diff_per_pass = t_base - t_chunk
        diff_per_layer = diff_per_pass
        total_save_per_chunk = diff_per_pass * 30 * 5
        print(f"\n--- Model-Scale Projection (30 layers, 5 passes/chunk = 150 self-attention projections):")
        print(f"  Per-projection delta: {diff_per_pass:+.3f} ms")
        print(f"  Total projected delta per generated chunk (150 calls): {total_save_per_chunk:+.1f} ms")


if __name__ == "__main__":
    run_benchmark("xpu")
