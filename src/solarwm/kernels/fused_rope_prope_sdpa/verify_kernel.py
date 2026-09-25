#!/usr/bin/env python3
"""CLI utility to verify numerical equivalence of candidate kernels against the reference within allowed ULPs.

Usage:
    # Run ULP equivalence verification of reference code against in-tree model:
    python -m solarwm.kernels.fused_rope_prope_sdpa.verify_kernel

    # Run with specific RoPE dtype and max_ulp tolerance on XPU or CPU:
    python -m solarwm.kernels.fused_rope_prope_sdpa.verify_kernel --device xpu --rope-dtype float32 --max-ulp 2

    # Benchmark latency of the reference implementation:
    python -m solarwm.kernels.fused_rope_prope_sdpa.verify_kernel --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solarwm.backends.wan22.runtime.modeling.attention import attention
from solarwm.backends.wan22.runtime.modeling.camera_prope import prope_apply_fns_separate_cached
from solarwm.backends.wan22.runtime.modeling.causal_model import block_relativistic_rope
from solarwm.kernels.fused_rope_prope_sdpa import (
    compare_tensors_ulp,
    create_test_inputs,
    fused_rope_prope_sdpa_reference,
)


def run_intree_pipeline(inputs: dict) -> torch.Tensor:
    """Run the exact in-tree PyTorch pipeline from causal_model.py."""
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    freqs = inputs["freqs"]
    q_grid = inputs["q_grid_sizes"]
    k_grid = inputs["k_grid_sizes"]
    q_viewmats = inputs["q_viewmats"]
    q_Ks = inputs["q_Ks"]
    kv_viewmats = inputs["kv_viewmats"]
    kv_Ks = inputs["kv_Ks"]
    q_start = inputs["q_start_frame"]
    k_start = inputs["k_start_frame"]
    rope_dtype = inputs["rope_dtype"]

    roped_q = block_relativistic_rope(q, q_grid, freqs, start_frame=q_start, rope_dtype=rope_dtype)
    roped_k = block_relativistic_rope(k, k_grid, freqs, start_frame=k_start, rope_dtype=rope_dtype)

    cache = {}
    apply_fn_q, apply_fn_kv, apply_fn_o = prope_apply_fns_separate_cached(
        cache,
        head_dim=q.shape[-1],
        q_viewmats=q_viewmats,
        q_Ks=q_Ks,
        kv_viewmats=kv_viewmats,
        kv_Ks=kv_Ks,
        kv_window=(0, k.shape[1]),
    )

    attn_q = apply_fn_q(roped_q.transpose(1, 2)).transpose(1, 2)
    attn_k = apply_fn_kv(roped_k.transpose(1, 2)).transpose(1, 2)
    attn_v = apply_fn_kv(v.transpose(1, 2)).transpose(1, 2)

    sdpa_out = attention(attn_q, attn_k, attn_v)
    return apply_fn_o(sdpa_out.transpose(1, 2)).transpose(1, 2)


def verify_kernel(
    candidate_fn=None,
    rope_dtype: str = "float64",
    device: str = "cpu",
    max_ulp: int = 2,
    benchmark: bool = False,
) -> bool:
    print(f"\n=========================================================================")
    print(f" Verifying RoPE-PRoPE-SDPA (Device: {device.upper()}, RoPE: {rope_dtype}, Max ULP: {max_ulp})")
    print(f"=========================================================================")

    inputs = create_test_inputs(
        batch=1,
        num_heads=24,
        head_dim=128,
        q_frames=3,
        kv_frames=15,
        h_patches=15,
        w_patches=27,
        device=device,
        rope_dtype=rope_dtype,
        seed=42,
    )

    print(f"Input shapes: Q {list(inputs['q'].shape)}, K {list(inputs['k'].shape)}, V {list(inputs['v'].shape)}")

    # 1. Compute in-tree ground truth
    t0 = time.perf_counter()
    intree_out = run_intree_pipeline(inputs)
    if device == "xpu":
        torch.xpu.synchronize()
    print(f"In-tree reference computed in {(time.perf_counter() - t0)*1000:.2f} ms")

    # 2. Compute standalone reference output
    t0 = time.perf_counter()
    ref_out = fused_rope_prope_sdpa_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        freqs=inputs["freqs"],
        q_grid_sizes=inputs["q_grid_sizes"],
        k_grid_sizes=inputs["k_grid_sizes"],
        q_viewmats=inputs["q_viewmats"],
        q_Ks=inputs["q_Ks"],
        kv_viewmats=inputs["kv_viewmats"],
        kv_Ks=inputs["kv_Ks"],
        q_start_frame=inputs["q_start_frame"],
        k_start_frame=inputs["k_start_frame"],
        rope_dtype=rope_dtype,
    )
    if device == "xpu":
        torch.xpu.synchronize()
    print(f"Extracted package reference computed in {(time.perf_counter() - t0)*1000:.2f} ms")

    # 3. Check ULP equivalence between in-tree and extracted reference
    report_ref = compare_tensors_ulp(ref_out, intree_out, max_ulp=max_ulp)
    print("\n--- Check 1: Extracted Reference vs In-tree Model ---")
    print(report_ref)
    if not report_ref.is_equivalent:
        return False

    # 4. If a custom candidate kernel was supplied, test it!
    if candidate_fn is not None:
        print("\n--- Check 2: Custom Candidate Kernel vs Ground Truth ---")
        cand_out = candidate_fn(inputs)
        report_cand = compare_tensors_ulp(cand_out, intree_out, max_ulp=max_ulp)
        print(report_cand)
        if not report_cand.is_equivalent:
            return False

    # 5. Optional benchmark
    if benchmark:
        print("\n--- Benchmarking Latency (30 warmup, 50 repeats) ---")
        N = 50
        for _ in range(10):
            _ = fused_rope_prope_sdpa_reference(
                inputs["q"], inputs["k"], inputs["v"],
                freqs=inputs["freqs"],
                q_grid_sizes=inputs["q_grid_sizes"], k_grid_sizes=inputs["k_grid_sizes"],
                q_viewmats=inputs["q_viewmats"], q_Ks=inputs["q_Ks"],
                kv_viewmats=inputs["kv_viewmats"], kv_Ks=inputs["kv_Ks"],
                q_start_frame=inputs["q_start_frame"], k_start_frame=inputs["k_start_frame"],
                rope_dtype=rope_dtype,
            )
        if device == "xpu":
            torch.xpu.synchronize()

        t0 = time.perf_counter()
        for _ in range(N):
            _ = fused_rope_prope_sdpa_reference(
                inputs["q"], inputs["k"], inputs["v"],
                freqs=inputs["freqs"],
                q_grid_sizes=inputs["q_grid_sizes"], k_grid_sizes=inputs["k_grid_sizes"],
                q_viewmats=inputs["q_viewmats"], q_Ks=inputs["q_Ks"],
                kv_viewmats=inputs["kv_viewmats"], kv_Ks=inputs["kv_Ks"],
                q_start_frame=inputs["q_start_frame"], k_start_frame=inputs["k_start_frame"],
                rope_dtype=rope_dtype,
            )
        if device == "xpu":
            torch.xpu.synchronize()
        per_layer_ms = (time.perf_counter() - t0) / N * 1000.0
        print(f"Single Layer Latency: {per_layer_ms:.3f} ms | 30 Layers: {per_layer_ms * 30:.2f} ms")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify RoPE-PRoPE-SDPA numerical equivalence within allowed ULPs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "xpu", "cuda"], help="Device to run on")
    parser.add_argument(
        "--rope-dtype",
        default="all",
        choices=["all", "float64", "float32", "float16"],
        help="RoPE precision dtype",
    )
    parser.add_argument(
        "--max-ulp",
        type=int,
        default=2,
        help="Maximum allowable difference in Units in the Last Place (default: 2)",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    args = parser.parse_args()

    if args.device == "xpu" and not torch.xpu.is_available():
        print("XPU not available on this system, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    dtypes = ["float64", "float32", "float16"] if args.rope_dtype == "all" else [args.rope_dtype]

    all_passed = True
    for dt in dtypes:
        ok = verify_kernel(rope_dtype=dt, device=device, max_ulp=args.max_ulp, benchmark=args.benchmark)
        if not ok:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print(f" ALL VERIFICATIONS PASSED: Reference is equivalent within <= {args.max_ulp} ULPs across all dtypes!")
        return 0
    else:
        print(f" VERIFICATION FAILED: Discrepancies exceeded {args.max_ulp} ULPs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
