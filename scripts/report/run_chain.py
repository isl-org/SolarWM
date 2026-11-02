"""Reproduce the optimization chain of the XPU porting report on one B70.

Every step runs `bench_chunk.py` (from this checkout) against the listed commit
with the listed config overrides, on the identical dataset sample, seed, and
480x864 workload. A step's speedup is relative to its `ref` step: the previous
cumulative step, or for alternatives (Sage v1/v2, torch.compile, ...) the step
they replace. Results land in `outputs/report_chain/<step>.json`; existing
results are reused unless `--rerun` is given.

    python scripts/report/run_chain.py                 # all steps
    python scripts/report/run_chain.py --steps S7 S7a  # a subset
    python scripts/report/run_chain.py --table         # print the table only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "scripts/report/bench_chunk.py"
RESULTS = REPO / "outputs/report_chain"
WORKTREES = REPO / "outputs/report_worktrees"

# Last commit where every pre-SVDQuant optimization is still a config toggle.
TOGGLES = "4d0c2f9"
OFF_4D0 = [
    "runtime.stage2_fused_kernel=none",
    "runtime.stage2_encoded_kv_ring=false",
    "runtime.stage2_dit_fused_ops=false",
    "runtime.stage2_vae_channels_last=false",
    "runtime.stage2_vae_fused_kernels=false",
    "runtime.stage2_vae_int8_quarot=false",
]


def _with(base: list[str], *changes: str) -> list[str]:
    keys = {c.split("=", 1)[0] for c in changes}
    return [o for o in base if o.split("=", 1)[0] not in keys] + list(changes)


S6 = _with(OFF_4D0, "runtime.stage2_vae_channels_last=true", "runtime.stage2_vae_fused_kernels=true")
S7 = _with(S6, "runtime.stage2_encoded_kv_ring=true")
S8 = _with(S7, "runtime.stage2_fused_kernel=fused_rope_prope_sdpa")
S9 = _with(S8, "runtime.stage2_dit_fused_ops=true")
HEAD_FINAL = ["runtime.stage2_vae_quant=quarot_w8a8", "model.linear_quant=svdquant_k4"]

# (step, label, commit, overrides, ref, extra bench args)
STEPS = [
    ("S0", "Plain CUDA->XPU port", "001e586", [], None, []),
    ("S1", "+ host-sync removal, streaming VAE decode (WS3, WS5)", "b0491d0", [], "S0", []),
    ("S2", "+ buffer reuse, uint8 D2H, cached scheduler grids (WS5, WS6, WS17); KV cache cloned", "0352a3c", ["inference.kv_cache_mode=clone"], "S1", []),
    ("S3", "+ circular KV cache, PRoPE memo, .item() removal (WS4, WS12, WS16)", "0352a3c", [], "S2", []),
    ("S3a", "alt: torch.compile DiT blocks (WS7)", "0352a3c", ["inference.stage2_xpu_compile_blocks=true"], "S3", []),
    ("S4", "+ VAE elementwise in bf16 (WS14)", TOGGLES, OFF_4D0, "S3", []),
    ("S5", "+ VAE channels-last conv layout (WS14)", TOGGLES, _with(OFF_4D0, "runtime.stage2_vae_channels_last=true"), "S4", []),
    ("S6", "+ fused Triton VAE RMSNorm+SiLU (WS20)", TOGGLES, S6, "S5", []),
    ("S7", "+ RoPEd KV ring (WS13)", TOGGLES, S7, "S6", []),
    ("S8", "+ fused RoPE+PRoPE+SDPA Triton kernel", TOGGLES, S8, "S7", []),
    ("S8a", "alt: fused RoPE+PRoPE+SageAttention v1", TOGGLES, _with(S7, "runtime.stage2_fused_kernel=fused_rope_prope_sage"), "S7", []),
    ("S9", "+ DiT operator fusions (WS22)", TOGGLES, S9, "S8", []),
    ("S9a", "alt: torch.compile VAE decoder (WS24)", TOGGLES, _with(S9, "runtime.stage2_vae_compile=true"), "S9", []),
    ("S10", "+ QuaRot W8A8 VAE (WS21)", TOGGLES, _with(S9, "runtime.stage2_vae_int8_quarot=true"), "S9", []),
    ("S11", "+ attention-path rewrite with unified kernel selection", "32b1dd9", HEAD_FINAL[:1], "S10", []),
    ("S12", "+ VAE oneDNN append_sum epilogue, QuaRot prologue reuse (WS23)", "b66f630", HEAD_FINAL[:1], "S11", []),
    ("S13", "HEAD, same config (no further BF16 changes)", "HEAD", HEAD_FINAL[:1], "S12", []),
    ("S14", "+ SVDQuant W4A4 DiT linears, KernelFoundry kernel (WS25)", "HEAD", HEAD_FINAL, "S13", []),
    ("S14a", "alt: SVDQuant + SageAttention v1", "HEAD", HEAD_FINAL + ["model.attention_backend=sageattnv1"], "S14", []),
    ("S14c", "alt: SVDQuant + radial sparse attention", "HEAD", HEAD_FINAL + ["runtime.stage2_radial_attention=true"], "S14", []),
    ("R320", "S14 at 320x576 (different workload)", "HEAD", HEAD_FINAL, None, ["--res-320p"]),
]
TOTAL = ("S0", "S14")


def _checkout(commit: str) -> Path:
    if commit == "HEAD":
        return REPO
    path = WORKTREES / commit
    if not path.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(path), commit], cwd=REPO, check=True)
    return path


def _run(step: str, commit: str, overrides: list[str], extra: list[str]) -> None:
    cwd = _checkout(commit)
    cmd = [sys.executable, str(BENCH), "--output-json", str(RESULTS / f"{step}.json"), *extra]
    for override in overrides:
        cmd += ["--set", override]
    env = {**os.environ, "PYTHONPATH": str(cwd / "src")}
    print(f"[{step}] {commit}: {' '.join(cmd[2:])}", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _table() -> None:
    results = {s[0]: json.loads((RESULTS / f"{s[0]}.json").read_text()) for s in STEPS if (RESULTS / f"{s[0]}.json").exists()}
    print("| Step | Change | Commit | vs | DiT ms | DiT speedup | VAE ms | VAE speedup | Chunk ms | Chunk speedup | FPS |")
    print("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for step, label, _, _, ref, _ in STEPS:
        if step not in results:
            continue
        r = results[step]
        cells = []
        for key in ("dit_chunk_ms", "vae_tile_ms", "chunk_ms"):
            speedup = f"{results[ref][key] / r[key]:.2f}x" if ref in results else ""
            cells += [f"{r[key]:.1f}", speedup]
        print(f"| {step} | {label} | `{r['commit']}` | {ref or ''} | {' | '.join(cells)} | {r['fps']:.2f} |")
    first, last = TOTAL
    if first in results and last in results:
        print(f"\nTotal {first} -> {last}: {results[first]['chunk_ms'] / results[last]['chunk_ms']:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", default=[s[0] for s in STEPS])
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()
    if not args.table:
        for step, _, commit, overrides, _, extra in STEPS:
            if step in args.steps and (args.rerun or not (RESULTS / f"{step}.json").exists()):
                _run(step, commit, overrides, extra)
    _table()


if __name__ == "__main__":
    main()
