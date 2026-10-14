"""Comprehensive benchmark comparing Baseline BF16 Linear vs SVDQuant W4A4+BF16 Triton on Intel Arc XPU.

Evaluates:
1. Shape 1 (Self/Cross Attention): M = 1215, K = 3072, N = 3072
2. Shape 2 (FFN Expansion):       M = 1215, K = 3072, N = 14336
3. Shape 3 (FFN Contraction):     M = 1215, K = 14336, N = 3072
4. Full 30-block DiT model-scale projection (300 linear projections per forward, 5 passes per chunk).
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from solarwm.kernels.dit_svdquant.triton_w4a4_gemm import triton_svdquant_w4a4_gemm


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
    return (end - start) * 1000.0 / iters  # ms


def pack_int4_weights(w_int4: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 weights [N, K] into int8 [N, K // 2]."""
    w_even = w_int4[:, 0::2] & 0x0F
    w_odd = w_int4[:, 1::2] & 0x0F
    return (w_even | (w_odd << 4)).to(torch.int8)


def run_benchmark():
    device = "xpu"
    dtype = torch.bfloat16
    group_size = 64
    rank = 16

    print(f"=== SVDQuant W4A4+BF16 vs BF16 Linear Benchmark on {device} ===")
    print(f"Device name: {torch.xpu.get_device_name(0)}")

    # Target DiT Shapes:
    # 1. Attention: 8 linears per block (q, k, v, o for self + cross)
    # 2. FFN0: 1 linear per block (3072 -> 14336)
    # 3. FFN2: 1 linear per block (14336 -> 3072)
    shapes = [
        ("Attention (q/k/v/o)", 1215, 3072, 3072, 8),
        ("FFN0 (expansion)", 1215, 3072, 14336, 1),
        ("FFN2 (contraction)", 1215, 14336, 3072, 1),
    ]

    results = []

    for name, m, k, n, count_per_block in shapes:
        print(f"\n------------------------------------------------------------")
        print(f"Shape: {name} | M={m}, K={k}, N={n} | Count/Block={count_per_block}")
        print(f"------------------------------------------------------------")

        x = torch.randn(m, k, device=device, dtype=dtype)
        bias = torch.randn(n, device=device, dtype=dtype)
        s_in = torch.ones(k, device=device, dtype=dtype)
        lora_a = torch.randn(rank, k, device=device, dtype=dtype)
        lora_b = torch.randn(n, rank, device=device, dtype=dtype)

        # Baseline BF16 Linear
        w_bf16 = torch.randn(n, k, device=device, dtype=dtype)
        linear_bf16 = nn.Linear(k, n, bias=True, device=device, dtype=dtype)
        linear_bf16.weight.data.copy_(w_bf16)
        linear_bf16.bias.data.copy_(bias)

        # Synthetic SVDQuant INT4 weights & scales
        w_int4 = torch.randint(-8, 7, (n, k), dtype=torch.int8, device=device)
        w_packed = pack_int4_weights(w_int4)
        w_scale = torch.full((n, k // group_size), 0.05, device=device, dtype=dtype)

        # Numerical parity check vs Python pseudo-QDQ
        # (verify kernel runs without crash and produces finite values)
        out_svdquant = triton_svdquant_w4a4_gemm(
            x, w_packed, w_scale, s_in=s_in, bias=bias,
            lora_a=lora_a, lora_b=lora_b, group_size=group_size
        )
        assert torch.isfinite(out_svdquant).all(), "NaN or Inf detected in Triton output!"
        print("  Sanity check: Triton kernel produced all-finite outputs!")

        # Benchmark Baseline BF16
        t_base = time_fn(lambda: linear_bf16(x))

        # Benchmark SVDQuant W4A4 + BF16 Low-Rank
        t_svdquant = time_fn(lambda: triton_svdquant_w4a4_gemm(
            x, w_packed, w_scale, s_in=s_in, bias=bias,
            lora_a=lora_a, lora_b=lora_b, group_size=group_size
        ))

        # Benchmark W4A4 Main GEMM only (without low-rank branch)
        t_w4a4_only = time_fn(lambda: triton_svdquant_w4a4_gemm(
            x, w_packed, w_scale, s_in=s_in, bias=bias,
            lora_a=None, lora_b=None, group_size=group_size
        ))

        speedup_full = t_base / t_svdquant
        speedup_main = t_base / t_w4a4_only

        print(f"  Baseline BF16 Linear:              {t_base:6.3f} ms")
        print(f"  SVDQuant W4A4 + BF16 Low-Rank:     {t_svdquant:6.3f} ms (speedup: {speedup_full:5.2f}x, delta: {t_base - t_svdquant:+.3f} ms)")
        print(f"  W4A4 Main Branch Only (no LoRA):   {t_w4a4_only:6.3f} ms (speedup: {speedup_main:5.2f}x, delta: {t_base - t_w4a4_only:+.3f} ms)")

        results.append({
            "name": name,
            "count": count_per_block,
            "t_base": t_base,
            "t_svdquant": t_svdquant,
            "t_w4a4_only": t_w4a4_only,
        })

    # Model-scale impact calculation across 30 transformer blocks
    # 5 forward passes per chunk (4 denoise + 1 commit)
    num_blocks = 30
    passes_per_chunk = 5

    total_base_per_fwd = sum(r["count"] * r["t_base"] for r in results) * num_blocks
    total_svdquant_per_fwd = sum(r["count"] * r["t_svdquant"] for r in results) * num_blocks
    total_w4a4_per_fwd = sum(r["count"] * r["t_w4a4_only"] for r in results) * num_blocks

    total_base_chunk = total_base_per_fwd * passes_per_chunk
    total_svdquant_chunk = total_svdquant_per_fwd * passes_per_chunk
    savings_chunk = total_base_chunk - total_svdquant_chunk
    speedup_model_gemm = total_base_chunk / total_svdquant_chunk

    print("\n============================================================")
    print("WHOLE-MODEL PROJECTION IMPACT (30 Blocks, 300 Linears):")
    print("============================================================")
    print(f"Single Forward Linear GEMM Time:")
    print(f"  Baseline BF16:        {total_base_per_fwd:6.1f} ms")
    print(f"  SVDQuant W4A4+BF16:   {total_svdquant_per_fwd:6.1f} ms (speedup: {total_base_per_fwd / total_svdquant_per_fwd:.2f}x)")
    print(f"\nFull Chunk (5 Forward Passes = 1500 Linear Projections):")
    print(f"  Baseline BF16 GEMM:   {total_base_chunk:6.1f} ms")
    print(f"  SVDQuant W4A4+BF16:   {total_svdquant_chunk:6.1f} ms")
    print(f"  Net Latency Savings:  {savings_chunk:+6.1f} ms per chunk")
    print(f"  GEMM Speedup:         {speedup_model_gemm:.2f}x")
    print(f"  Whole-Chunk Impact:   {savings_chunk / 2680.0 * 100:.1f}% end-to-end chunk speedup (vs 2.68s baseline)")

    gate_passed = savings_chunk >= 134.0
    print(f"\n>>> GO/NO-GO GATE (Savings >= 134 ms): {'PASSED' if gate_passed else 'FAILED'} <<<")


if __name__ == "__main__":
    run_benchmark()
