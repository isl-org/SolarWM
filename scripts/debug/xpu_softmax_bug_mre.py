#!/usr/bin/env python3
"""Minimal reproducer: torch.softmax on Intel XPU is wrong for some last-dim sizes.

Dependency-free apart from torch. Inputs are fixed (no RNG), so results are comparable
across machines and torch versions. Reference is float64 on CPU.

Observed on torch 2.14.0+xpu (Intel B-series):
  * fp32: wrong for last-dim sizes 576, 640, 768, 1215, 1620, 2430, 3240
          (mean abs error 3e-4..1e-3, max up to 1.4e-2; correct sizes give ~1e-9)
  * bf16: returns NaN for last-dim sizes 1215, 2430, 3240, 8505

Usage:
  python scripts/debug/xpu_softmax_bug_mre.py
  python scripts/debug/xpu_softmax_bug_mre.py --device cuda        # expect all PASS
  python scripts/debug/xpu_softmax_bug_mre.py --include-large      # adds size 8505
  python scripts/debug/xpu_softmax_bug_mre.py --json report.json

Exit code 0 when every size passes, 1 when any size fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch

# Sizes seen in Wan 2.2 on XPU. 576/1620 are the Wan VAE AttentionBlock sequence
# lengths at latent 24x24 and 30x54; 1215 is the Stage2 self-attention length.
SIZES = (128, 256, 320, 384, 400, 405, 512, 576, 640, 768, 784, 810, 900, 1024, 1215, 1620, 2048, 2430, 3240)
LARGE_SIZES = (4096, 8505)

# fp32 softmax should be exact to ~1e-7; bf16 rounding alone stays well under 1e-3.
TOLERANCE = {"fp32": 1e-6, "bf16": 1e-3}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def pick_device(name: str) -> torch.device:
    if name == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise SystemExit("torch.xpu is not available in this environment")
        return torch.device("xpu", 0)
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("torch.cuda is not available in this environment")
        return torch.device("cuda", 0)
    if name == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"unknown device: {name}")


def fixed_scores(seq: int) -> torch.Tensor:
    """Deterministic [1, 1, seq, seq] logits in roughly [-4, 4]."""

    base = torch.arange(seq * seq, dtype=torch.float32)
    return (torch.sin(base * 7e-4) * 4.0).view(1, 1, seq, seq)


def sdpa_check(device: torch.device) -> list[dict[str, Any]]:
    """Downstream consequence: SDPA at the Wan VAE AttentionBlock shape (1 head, head_dim 640)."""

    rows: list[dict[str, Any]] = []
    for seq in (576, 900, 1620):
        count = seq * 640
        q = (torch.sin(torch.arange(count, dtype=torch.float32) * 1.3e-3) * 0.5).view(1, 1, seq, 640)
        k = (torch.cos(torch.arange(count, dtype=torch.float32) * 1.7e-3) * 0.5).view(1, 1, seq, 640)
        v = (torch.sin(torch.arange(count, dtype=torch.float32) * 1.1e-3 + 1.0) * 0.5).view(
            1, 1, seq, 640
        )
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.double(), k.double(), v.double()
        ).float()
        for label, dtype in DTYPES.items():
            got = torch.nn.functional.scaled_dot_product_attention(
                q.to(device=device, dtype=dtype),
                k.to(device=device, dtype=dtype),
                v.to(device=device, dtype=dtype),
            ).float().cpu()
            finite = bool(torch.isfinite(got).all())
            max_abs = float((reference - got).abs().max().item()) if finite else float("nan")
            verdict = "pass" if finite and max_abs <= 2e-3 else "FAIL"
            rows.append({"seq": seq, "dtype": label, "max_abs_diff": max_abs, "finite": finite})
            print(f"  sdpa seq={seq:>5} d=640 {label} max={max_abs:12.4g} finite={finite}  {verdict}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--include-large", action="store_true", help="also test 4096 and 8505")
    parser.add_argument(
        "--also-sdpa",
        action="store_true",
        help="additionally check SDPA at the Wan VAE AttentionBlock shape",
    )
    parser.add_argument("--json", type=str, default=None, help="write a JSON report here")
    args = parser.parse_args()

    device = pick_device(args.device)
    sizes = SIZES + (LARGE_SIZES if args.include_large else ())

    build = getattr(torch.version, "xpu", None) or getattr(torch.version, "cuda", None)
    header = {
        "torch": torch.__version__,
        "device": device.type,
        "backend_version": build,
    }
    print(f"torch={header['torch']} device={device.type} backend={build}")
    print(f"{'size':>6} {'fp32 max':>12} {'fp32 mean':>12} {'bf16 max':>12} {'bf16 mean':>12}  verdict")

    rows: list[dict[str, Any]] = []
    failures: list[int] = []
    for seq in sizes:
        scores = fixed_scores(seq)
        reference = torch.softmax(scores.double(), dim=-1).float()
        row: dict[str, Any] = {"size": seq}
        bad: list[str] = []
        for label, dtype in DTYPES.items():
            value = torch.softmax(scores.to(device=device, dtype=dtype), dim=-1).float().cpu()
            finite = bool(torch.isfinite(value).all())
            diff = (reference - value).abs()
            max_abs = float(diff.max().item()) if finite else float("nan")
            mean_abs = float(diff.mean().item()) if finite else float("nan")
            row[label] = {"max_abs_diff": max_abs, "mean_abs_diff": mean_abs, "finite": finite}
            if not finite:
                bad.append(f"{label}:non-finite")
            elif max_abs > TOLERANCE[label]:
                bad.append(f"{label}:max={max_abs:.3g}")
        row["failures"] = bad
        rows.append(row)
        if bad:
            failures.append(seq)
        verdict = "FAIL " + ",".join(bad) if bad else "pass"
        print(
            f"{seq:>6} {row['fp32']['max_abs_diff']:12.4g} {row['fp32']['mean_abs_diff']:12.4g}"
            f" {row['bf16']['max_abs_diff']:12.4g} {row['bf16']['mean_abs_diff']:12.4g}  {verdict}"
        )

    sdpa_rows: list[dict[str, Any]] = []
    if args.also_sdpa:
        print()
        sdpa_rows = sdpa_check(device)

    print()
    if failures:
        print(f"FAILING SIZES ({len(failures)}/{len(sizes)}): {failures}")
    else:
        print(f"ALL {len(sizes)} SIZES PASS")

    if args.json:
        payload = {
            "schema": "solarwm.xpu-softmax-bug-mre.v1",
            **header,
            "tolerance": TOLERANCE,
            "failing_sizes": failures,
            "results": rows,
            "sdpa_vae_shape": sdpa_rows,
        }
        parent = os.path.dirname(os.path.abspath(args.json))
        os.makedirs(parent, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
