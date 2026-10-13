"""Short (18 latent frame) rollout with chunk-4 timing and latent error vs the first config.

Usage: python verify_svdquant_rollout.py LINEAR_QUANT:ATTENTION_BACKEND[:radial] ...
e.g.   full_precision:sdpa svdquant_k4:sageattnv1:radial
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_chunk4_timing import measure_pipeline  # noqa: E402


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def parse(spec: str) -> tuple[str, str, bool]:
    parts = spec.split(":")
    return parts[0], parts[1], len(parts) > 2 and parts[2] == "radial"


def main() -> None:
    specs = sys.argv[1:] or ["full_precision:sdpa", "svdquant_k4:sdpa"]
    base = None
    for spec in specs:
        lq, ab, radial = parse(spec)
        res = measure_pipeline(linear_quant=lq, attn_backend=ab, return_latents=True, radial=radial)
        lat, gt = res["latents"], res["gt_latents"]
        steps = res["chunk4_steps_ms"]
        line = (f"{spec:58s} | steps ms: " + " ".join(f"{k}={v:.1f}" for k, v in steps.items())
                + f" | DiT {res['chunk4_dit_total_ms']:.1f} ms | finite {bool(torch.isfinite(lat).all())}"
                + f" | vs GT {rel(lat, gt):.4f}"
                + f" | VAE bf16 {res['chunk4_bf16_vae_ms']:.1f} ms quarot {res['chunk4_quarot_vae_ms']:.1f} ms")
        if base is None:
            base = lat
        else:
            line += f" | vs first {rel(lat, base):.4f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
