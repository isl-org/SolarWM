#!/usr/bin/env python3
"""Benchmark DiT operator fusions on Intel Arc XPU.

Measures:
1. Micro-benchmarks for each fused primitive vs eager baseline:
   - Fused AdaLN + LayerNorm (Triton) vs PyTorch eager
   - Fused WanRMSNorm (Triton) vs PyTorch eager
   - Fused Gated Residual Accumulation (native oneAPI addcmul_) vs PyTorch eager
   - Fused Linear GEMM + GELU(tanh) (oneDNN _linear_pointwise) vs PyTorch eager
2. Full DiT Transformer Block forward pass (dim=3072, ffn_dim=13824, L=1215 tokens)
3. DRAM memory traffic eliminated per block and across 120-pass rollout chunks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from solarwm.backends.wan22.runtime.modeling.causal_model import CausalWanAttentionBlock
from solarwm.kernels.dit_fused import (
    fused_gated_residual_add_,
    onednn_linear_gelu_tanh,
    triton_layernorm_adaln,
    triton_rmsnorm,
)


def benchmark_primitives(device: str = "xpu:0", repeats: int = 100) -> dict[str, Any]:
    print("=" * 70)
    print("1. DiT Operator Fusion Microbenchmarks (Production Shapes)")
    print("=" * 70)

    B, S, C = 1, 1215, 3072
    num_tokens = 5
    mod_seqlen = S // num_tokens
    results = {}

    # Target 1: AdaLN + LayerNorm
    x = torch.randn(B, S, C, device=device, dtype=torch.bfloat16)
    scale = torch.randn(B, num_tokens, 1, C, device=device, dtype=torch.bfloat16)
    shift = torch.randn(B, num_tokens, 1, C, device=device, dtype=torch.bfloat16)

    def eager_adaln():
        norm = F.layer_norm(x.float(), (C,), eps=1e-6).to(torch.bfloat16)
        return (norm.unflatten(1, (num_tokens, mod_seqlen)) * (1.0 + scale) + shift).flatten(1, 2)

    def fused_adaln():
        return triton_layernorm_adaln(x, scale, shift, eps=1e-6)

    # Warmup
    for _ in range(20):
        _ = eager_adaln()
        _ = fused_adaln()
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = eager_adaln()
    torch.xpu.synchronize()
    t_eager_adaln = (time.perf_counter() - t0) / repeats * 1000

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = fused_adaln()
    torch.xpu.synchronize()
    t_fused_adaln = (time.perf_counter() - t0) / repeats * 1000

    results["adaln_layernorm"] = {
        "eager_ms": round(t_eager_adaln, 4),
        "fused_ms": round(t_fused_adaln, 4),
        "speedup": round(t_eager_adaln / t_fused_adaln, 2),
        "traffic_saved_mb": round((S * C * 2 * 3) / 1024**2, 2),  # 3 intermediate passes
    }
    print(f"Target 1 (AdaLN+LayerNorm):  Eager={t_eager_adaln:.3f} ms | Fused={t_fused_adaln:.3f} ms | Speedup={t_eager_adaln/t_fused_adaln:.2f}x")

    # Target 2: Gated Residual Add
    y = torch.randn(B, S, C, device=device, dtype=torch.bfloat16)
    gate = torch.randn(B, num_tokens, 1, C, device=device, dtype=torch.bfloat16)
    x_test = x.clone()

    def eager_gated_add():
        return x + (y.unflatten(1, (num_tokens, mod_seqlen)) * gate).flatten(1, 2)

    def fused_gated_add():
        return fused_gated_residual_add_(x_test, y, gate, num_tokens, mod_seqlen)

    for _ in range(20):
        _ = eager_gated_add()
        _ = fused_gated_add()
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = eager_gated_add()
    torch.xpu.synchronize()
    t_eager_gated = (time.perf_counter() - t0) / repeats * 1000

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = fused_gated_add()
    torch.xpu.synchronize()
    t_fused_gated = (time.perf_counter() - t0) / repeats * 1000

    results["gated_residual_add"] = {
        "eager_ms": round(t_eager_gated, 4),
        "fused_ms": round(t_fused_gated, 4),
        "speedup": round(t_eager_gated / t_fused_gated, 2),
        "traffic_saved_mb": round((S * C * 2 * 2) / 1024**2, 2),  # bypasses intermediate product
    }
    print(f"Target 2 (Gated Resid Add): Eager={t_eager_gated:.3f} ms | Fused={t_fused_gated:.3f} ms | Speedup={t_eager_gated/t_fused_gated:.2f}x")

    # Target 3: WanRMSNorm
    weight = torch.randn(C, device=device, dtype=torch.bfloat16)

    def eager_rmsnorm():
        return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)).type_as(x) * weight

    def fused_rmsnorm():
        return triton_rmsnorm(x, weight, eps=1e-5)

    for _ in range(20):
        _ = eager_rmsnorm()
        _ = fused_rmsnorm()
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = eager_rmsnorm()
    torch.xpu.synchronize()
    t_eager_rms = (time.perf_counter() - t0) / repeats * 1000

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = fused_rmsnorm()
    torch.xpu.synchronize()
    t_fused_rms = (time.perf_counter() - t0) / repeats * 1000

    results["rmsnorm"] = {
        "eager_ms": round(t_eager_rms, 4),
        "fused_ms": round(t_fused_rms, 4),
        "speedup": round(t_eager_rms / t_fused_rms, 2),
        "traffic_saved_mb": round((S * C * 2 * 2) / 1024**2, 2),
    }
    print(f"Target 3 (WanRMSNorm):       Eager={t_eager_rms:.3f} ms | Fused={t_fused_rms:.3f} ms | Speedup={t_eager_rms/t_fused_rms:.2f}x")

    # Target 4: oneDNN Linear + GELU(tanh) Epilogue
    N_ffn = 13824
    w_ffn = torch.randn(N_ffn, C, device=device, dtype=torch.bfloat16)
    b_ffn = torch.randn(N_ffn, device=device, dtype=torch.bfloat16)

    def eager_linear_gelu():
        return F.gelu(F.linear(x, w_ffn, b_ffn), approximate="tanh")

    def fused_linear_gelu():
        return onednn_linear_gelu_tanh(x, w_ffn, b_ffn)

    for _ in range(20):
        _ = eager_linear_gelu()
        _ = fused_linear_gelu()
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = eager_linear_gelu()
    torch.xpu.synchronize()
    t_eager_gelu = (time.perf_counter() - t0) / repeats * 1000

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = fused_linear_gelu()
    torch.xpu.synchronize()
    t_fused_gelu = (time.perf_counter() - t0) / repeats * 1000

    results["linear_gelu_epilogue"] = {
        "eager_ms": round(t_eager_gelu, 4),
        "fused_ms": round(t_fused_gelu, 4),
        "speedup": round(t_eager_gelu / t_fused_gelu, 2),
        "traffic_saved_mb": round((S * N_ffn * 2) / 1024**2, 2),  # eliminates unactivated write
    }
    print(f"Target 4 (oneDNN Linear+GELU):Eager={t_eager_gelu:.3f} ms | Fused={t_fused_gelu:.3f} ms | Speedup={t_eager_gelu/t_fused_gelu:.2f}x")

    return results


def benchmark_production_block(device: str = "xpu:0", repeats: int = 50) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("2. Full DiT Attention Block Benchmark (dim=3072, ffn_dim=13824, L=1215)")
    print("=" * 70)

    dim = 3072
    ffn_dim = 13824
    num_heads = 24
    S = 1215
    num_tokens = 5

    block_eager = CausalWanAttentionBlock(
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        local_attn_size=6,
        sink_size=1,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        dit_fused_ops=False,
    ).to(device, dtype=torch.bfloat16)
    block_eager.eval()

    block_fused = CausalWanAttentionBlock(
        dim=dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        local_attn_size=6,
        sink_size=1,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        dit_fused_ops=True,
    ).to(device, dtype=torch.bfloat16)
    block_fused.eval()
    block_fused.load_state_dict(block_eager.state_dict())

    x = torch.randn(1, S, dim, device=device, dtype=torch.bfloat16)
    e = torch.randn(1, num_tokens, 6, dim, device=device, dtype=torch.bfloat16)
    grid_sizes = torch.tensor([[1, 27, 45]], device=device)
    seq_lens = torch.tensor([S], device=device)
    d = dim // num_heads
    freqs = torch.randn(1024, d // 2, device=device, dtype=torch.float32)
    context = torch.randn(1, 512, dim, device=device, dtype=torch.bfloat16)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = block_eager(x, e, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs, context=context, context_lens=None, kv_cache=None)
            _ = block_fused(x, e, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs, context=context, context_lens=None, kv_cache=None)
        torch.xpu.synchronize()

        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = block_eager(x, e, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs, context=context, context_lens=None, kv_cache=None)
        torch.xpu.synchronize()
        t_eager_block = (time.perf_counter() - t0) / repeats * 1000

        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = block_fused(x, e, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs, context=context, context_lens=None, kv_cache=None)
        torch.xpu.synchronize()
        t_fused_block = (time.perf_counter() - t0) / repeats * 1000

    saved_per_block = t_eager_block - t_fused_block
    saved_chunk = saved_per_block * 120  # 30 blocks * 4 denoise steps

    print(f"Eager DiT Block Forward:  {t_eager_block:.3f} ms")
    print(f"Fused DiT Block Forward:  {t_fused_block:.3f} ms")
    print(f"Block Latency Savings:    {saved_per_block:.3f} ms ({t_eager_block/t_fused_block:.2f}x speedup)")
    print(f"Projected Chunk Savings:  {saved_chunk:.1f} ms across 120 block evaluations")

    return {
        "eager_block_ms": round(t_eager_block, 3),
        "fused_block_ms": round(t_fused_block, 3),
        "speedup": round(t_eager_block / t_fused_block, 3),
        "saved_per_block_ms": round(saved_per_block, 3),
        "projected_chunk_saved_ms": round(saved_chunk, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark DiT Operator Fusions on Intel XPU")
    parser.add_argument("--device", default="xpu:0", help="Execution device")
    parser.add_argument("--repeats", type=int, default=50, help="Benchmark repeat count")
    parser.add_argument("--output", default="bench_dit_operator_fusions.json", help="Output JSON path")
    args = parser.parse_args()

    assert torch.xpu.is_available(), "Intel XPU device is required"
    print(f"Running benchmarks on: {torch.xpu.get_device_name(0)}")

    primitives = benchmark_primitives(device=args.device, repeats=args.repeats)
    block = benchmark_production_block(device=args.device, repeats=args.repeats)

    total_traffic_saved_mb = (
        primitives["adaln_layernorm"]["traffic_saved_mb"] * 2  # pre-attn + pre-ffn
        + primitives["gated_residual_add"]["traffic_saved_mb"] * 2  # attn + ffn res
        + primitives["rmsnorm"]["traffic_saved_mb"] * 2  # norm_q + norm_k
        + primitives["linear_gelu_epilogue"]["traffic_saved_mb"]  # ffn up-proj
    )
    total_chunk_traffic_gb = round((total_traffic_saved_mb * 120) / 1024, 2)

    report = {
        "device": torch.xpu.get_device_name(0),
        "primitives": primitives,
        "block": block,
        "traffic_savings": {
            "saved_per_block_mb": round(total_traffic_saved_mb, 2),
            "total_chunk_saved_gb": total_chunk_traffic_gb,
        },
    }

    print("\n" + "=" * 70)
    print("3. Memory Traffic Summary")
    print("=" * 70)
    print(f"DRAM Traffic Saved per Block: {total_traffic_saved_mb:.2f} MB")
    print(f"Total Chunk HBM Traffic Saved: {total_chunk_traffic_gb:.2f} GB (120 block evaluations)")

    out_path = Path(args.output)
    with open(out_path, "w") as fp:
        json.dump(report, fp, indent=2)
    print(f"\nWrote full benchmark report to {out_path.resolve()}")


if __name__ == "__main__":
    main()
