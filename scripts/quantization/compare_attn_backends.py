"""Compare activation statistics and Hessians across SDPA, SageAttention v1, and SageAttention v2."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
QUANT_DIR = Path(__file__).resolve().parents[1] / "quant"
if str(QUANT_DIR) not in sys.path:
    sys.path.insert(0, str(QUANT_DIR))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
import torch.nn as nn

from scripts.quantization.calibrate_svdquant import (
    ActivationCollector,
    build_provider,
    guarded_samples,
    iter_latent_samples,
    rollout,
)
from scripts.quantization.svdquant_int4 import ATTENTION_BACKENDS, with_attention_backend


def compare_backends(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"Loading Stage2 provider on {device}...")
    provider = build_provider(args.provider_args)
    model = getattr(provider.diffusion, "module", getattr(provider.diffusion, "model", provider.diffusion))

    shards = [Path(s) for s in args.shards]
    samples = list(iter_latent_samples(shards, per_shard=args.per_shard))
    samples = list(guarded_samples(provider, samples, latent_frames=args.latent_frames))
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"Using {len(samples)} samples for backend comparison.")

    check_blocks = [0, len(model.blocks) // 2, len(model.blocks) - 1]
    check_targets = ["x", "ffn_down"]
    check_bins = ["0", "2", "commit"]

    backends = list(ATTENTION_BACKENDS)
    collectors: dict[str, ActivationCollector] = {}

    for b in backends:
        print(f"\n--- Running capture with backend: {b} ---", flush=True)
        collector = ActivationCollector(
            num_blocks=len(model.blocks),
            subsample_limit=2048,
            target_blocks=check_blocks,
        )
        collector.attach(model)
        t0 = time.time()
        with with_attention_backend(model, b):
            for idx, sample in enumerate(samples):
                print(f"  [{b}] Sample {idx + 1}/{len(samples)}...", flush=True)
                rollout(provider, sample, latent_frames=args.latent_frames, seed=args.seed + idx)
        collector.detach()
        print(f"Captured {b} in {time.time() - t0:.2f}s", flush=True)
        collectors[b] = collector

    print("\n" + "=" * 80)
    print("ATTENTION BACKEND ACTIVATION COMPARISON REPORT")
    print("=" * 80)

    pairs = [
        ("sdpa", "sageattnv1"),
    ]
    report: dict[str, Any] = {"pairs": {}}

    for name_a, name_b in pairs:
        col_a = collectors[name_a]
        col_b = collectors[name_b]

        amax_rel_diffs = []
        hessian_rel_diffs = []
        hessian_cos_sims = []

        for b_idx in check_blocks:
            for tgt in check_targets:
                for b_name in check_bins:
                    s_a = col_a.stats.get((b_idx, tgt, b_name))
                    s_b = col_b.stats.get((b_idx, tgt, b_name))
                    if s_a is None or s_b is None:
                        continue

                    # Absmax relative difference
                    a_a = s_a["absmax"]
                    a_b = s_b["absmax"]
                    diff = torch.norm(a_a - a_b) / (torch.norm(a_a).clamp(min=1e-8))
                    amax_rel_diffs.append(diff.item())

                    # Hessian relative diff & cosine similarity
                    h_a = s_a["hessian"].flatten()
                    h_b = s_b["hessian"].flatten()
                    h_diff = torch.norm(h_a - h_b) / (torch.norm(h_a).clamp(min=1e-8))
                    hessian_rel_diffs.append(h_diff.item())

                    dot = torch.dot(h_a.double(), h_b.double())
                    denom = (torch.norm(h_a.double()) * torch.norm(h_b.double())).clamp(min=1e-12)
                    cos = (dot / denom).clamp(-1.0, 1.0).item()
                    hessian_cos_sims.append(cos)

        avg_amax_diff = sum(amax_rel_diffs) / max(1, len(amax_rel_diffs))
        avg_h_diff = sum(hessian_rel_diffs) / max(1, len(hessian_rel_diffs))
        avg_h_cos = sum(hessian_cos_sims) / max(1, len(hessian_cos_sims))
        min_h_cos = min(hessian_cos_sims) if hessian_cos_sims else 1.0

        pair_summary = {
            "avg_absmax_rel_diff": avg_amax_diff,
            "avg_hessian_rel_diff": avg_h_diff,
            "avg_hessian_cosine_similarity": avg_h_cos,
            "min_hessian_cosine_similarity": min_h_cos,
        }
        report["pairs"][f"{name_a}_vs_{name_b}"] = pair_summary

        print(f"\n[{name_a} vs {name_b}]")
        print(f"  Avg absmax relative difference:       {avg_amax_diff:.4%}")
        print(f"  Avg Hessian relative difference:      {avg_h_diff:.4%}")
        print(f"  Avg Hessian cosine similarity:         {avg_h_cos:.6f}")
        print(f"  Min Hessian cosine similarity:         {min_h_cos:.6f}")

    out_file = Path(args.output_json) if args.output_json else Path("attn_backend_comparison.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote comparison metrics to {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Attention Backends on Activation Stats")
    parser.add_argument("--shards", nargs="+", required=True, help="List of latent-WDS .tar shards")
    parser.add_argument("--per-shard", type=int, default=1, help="Samples to take per shard")
    parser.add_argument("--max-samples", type=int, default=2, help="Max total calibration samples to compare")
    parser.add_argument("--latent-frames", type=int, default=39, help="Latent frames per sample")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device")
    parser.add_argument("--output-json", type=str, default="attn_backend_comparison.json")
    args, rest = parser.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    args.provider_args = rest
    return args


if __name__ == "__main__":
    compare_backends(parse_args())
