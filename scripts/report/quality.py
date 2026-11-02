"""Speed and output error of each quantization method and their combinations.

Runs `bench_chunk.py --save-frames` at HEAD for each configuration below and
compares its decoded 13-chunk video (153 frames, 480x864) with the all-BF16
reference (SDPA attention, BF16 DiT linears, BF16 VAE) on the 0-255 pixel
scale, over the first chunk (9 frames) and the last chunk (12 frames). Speed is
the steady-state DiT, VAE and chunk time of the same run.

SVDQuant uses the k4 weights (4 timestep sets) everywhere except the standalone
k1 and k2 rows (1 and 2 timestep sets). The weight folder also
follows the attention backend: SDPA loads `sdpa_k4`, Sage v1 `sageattnv1_k4`,
and Sage v2, which has no weights of its own, is run only with BF16 linears.

The reference is run twice as the run-to-run noise floor. Videos and metrics
land in `outputs/report_chain/quality_153f/`.

    python scripts/report/quality.py           # reuses saved videos
    python scripts/report/quality.py --rerun
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

from run_chain import BENCH, REPO

OUT = REPO / "outputs/report_chain/quality_153f"
SAGE1, SAGE2 = "model.attention_backend=sageattnv1", "model.attention_backend=sageattnv2"
SVDQ, W8A8 = "model.linear_quant=svdquant_k4", "runtime.stage2_vae_quant=quarot_w8a8"
CONFIGS = [  # name, label, overrides
    ("bf16", "BF16 reference", []),
    ("bf16_repeat", "BF16 reference run twice", []),
    ("sage1", "SageAttention v1", [SAGE1]),
    ("sage2", "SageAttention v2", [SAGE2]),
    ("svdq_k1", "SVDQuant W4A4 k1", ["model.linear_quant=svdquant_k1"]),
    ("svdq_k2", "SVDQuant W4A4 k2", ["model.linear_quant=svdquant_k2"]),
    ("svdq", "SVDQuant W4A4 k4", [SVDQ]),
    ("w8a8", "QuaRot dynamic W8A8 VAE", [W8A8]),
    ("svdq_w8a8", "SVDQuant + W8A8", [SVDQ, W8A8]),
    ("sage1_svdq", "Sage v1 + SVDQuant", [SAGE1, SVDQ]),
    ("sage1_svdq_w8a8", "Sage v1 + SVDQuant + W8A8", [SAGE1, SVDQ, W8A8]),
]
FRAMES_PER_CHUNK = 12
CHUNKS = 13


def _run(name: str, overrides: list[str], rerun: bool) -> tuple[torch.Tensor, dict]:
    path = OUT / f"{name}.pt"
    if rerun or not path.exists():
        cmd = [sys.executable, str(BENCH), "--output-json", str(OUT / f"{name}.json"), "--save-frames", str(path),
               "--chunks", str(CHUNKS), "--vae-repeats", "1"]
        for override in overrides:
            cmd += ["--set", override]
        print(f"[{name}] {' '.join(cmd[2:])}", flush=True)
        subprocess.run(cmd, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, check=True)
    video = torch.load(path)
    if video.dtype != torch.uint8:
        video = (video.float().clamp(-1, 1) + 1) * 127.5
    return video.float(), json.loads((OUT / f"{name}.json").read_text())


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    rmse = (a - b).pow(2).mean().sqrt().item()
    return dict(rmse=rmse, psnr_db=math.inf if rmse == 0 else 20 * math.log10(255.0 / rmse))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ref, ref_timing = _run("bf16", [], args.rerun)
    first = ref.shape[1] - FRAMES_PER_CHUNK * (CHUNKS - 1)
    last = slice(ref.shape[1] - FRAMES_PER_CHUNK, None)
    rows = {}
    print(f"{'config':28s} {'DiT ms':>8s} {'VAE ms':>8s} {'chunk ms':>9s} {'chunk x':>8s} {'PSNR 1st':>9s} {'PSNR last':>9s}")
    for name, label, overrides in CONFIGS:
        video, timing = _run(name, overrides, args.rerun)
        dit, vae = timing["dit_chunk_ms"], timing["vae_tile_ms"]
        rows[name] = dict(label=label, overrides=overrides, frames=video.shape[1], dit_ms=dit, vae_ms=vae,
                          chunk_ms=dit + vae, chunk_speedup=ref_timing["chunk_ms"] / (dit + vae),
                          first_chunk=_metrics(video[:, :first], ref[:, :first]),
                          last_chunk=_metrics(video[:, last], ref[:, last]))
        r = rows[name]
        print(f"{label:28s} {dit:8.1f} {vae:8.1f} {dit + vae:9.1f} {r['chunk_speedup']:7.2f}x "
              f"{r['first_chunk']['psnr_db']:7.2f}dB {r['last_chunk']['psnr_db']:7.2f}dB")
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
