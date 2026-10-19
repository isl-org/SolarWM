"""Detailed timing benchmark of Chunk 4 (steady state):
- Individual diffusion forward passes (steps 0, 1, 2, 3, and commit)
- Total DiT chunk 4 time
- VAE decode time for chunk 4 (BF16 VAE vs QuaRot W8A8 Quantized VAE)
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from solarwm.backends.wan22.runtime import stage2
from solarwm.backends.wan22.runtime.modeling.vae_quarot import apply_vae_quarot
from solarwm.config import load_config
from scripts.quantization.run_svdquant_inference import (
    _NoTextEncoder,
    iter_latent_samples,
    sample_inputs,
    sync_device,
)

stage2.WanTextEncoder = _NoTextEncoder

CONFIG_PATH = "configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml"
SHARDS = [Path("/home/ssheorey/data/SolarWM-Data/releases-v1/latent-wds/wan22-ti2v5b-153f-480p-v1/abot/shards/kept-high-000008.tar")]


def measure_pipeline(
    linear_quant: str = "svdquant_k4",
    attn_backend: str = "sdpa",
    return_latents: bool = False,
    radial: bool = False,
):
    overrides = [
        "model.base_path=/home/ssheorey/models/SolarWM/SolarWM-5B-base",
        "checkpoint.path=/home/ssheorey/models/SolarWM/SolarWM-5B-sgf-stage2-81f",
        "data.index_root=/home/ssheorey/data/SolarWM-Data/releases-v1",
        "data.transport.root=/home/ssheorey/data/SolarWM-Data/releases-v1",
        "inference.device=xpu",
        f"model.linear_quant={linear_quant}",
        f"model.attention_backend={attn_backend}",
        f"runtime.stage2_radial_attention={'true' if radial else 'false'}",
    ]
    values = load_config(CONFIG_PATH, overrides).values
    provider = stage2.build_stage2_generation_provider(values)
    provider._load_role(str(provider.plan.passes[0].weights))
    device = provider.device

    sample = next(iter_latent_samples(SHARDS, per_shard=1))
    # 6 chunks (18 latent frames) ensures chunk 4 is fully in steady-state (full KV cache window)
    latent_frames = 18
    generation_pass = replace(
        provider.plan.passes[0],
        rollout_latent_frames=latent_frames,
        min_rollout_latent_frames=latent_frames,
    )
    first_latent, condition, camera, _ = sample_inputs(provider, sample, latent_frames)
    rng = torch.Generator(device=device).manual_seed(1234)

    # Instrument provider.diffusion to record exact forward pass times
    chunk_step_times: dict[int, dict[int | str, float]] = {}
    current_chunk = [0]

    class TimedDiffusionWrapper:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __call__(self, *args, **kwargs):
            radial_step = kwargs.get("radial_step", None)
            sync_device(device)
            t0 = time.perf_counter()
            out = self.inner(*args, **kwargs)
            sync_device(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            c_idx = current_chunk[0]
            step_key = "commit" if radial_step == 4 else radial_step
            chunk_step_times.setdefault(c_idx, {})[step_key] = elapsed_ms
            return out

    provider.diffusion = TimedDiffusionWrapper(provider.diffusion)

    # We also time the overall chunk 4 duration in DiT
    orig_stage2_latents = stage2._stage2_self_forcing_latents

    chunk_total_times: dict[int, float] = {}
    chunk_start_times: dict[int, float] = {}

    orig_slice_cam = stage2._slice_camera

    def hooked_slice_camera(*args, **kwargs):
        start_f = kwargs.get("start_frame", 0)
        c_idx = start_f // 3
        current_chunk[0] = c_idx
        sync_device(device)
        chunk_start_times[c_idx] = time.perf_counter()
        return orig_slice_cam(*args, **kwargs)

    stage2._slice_camera = hooked_slice_camera

    with torch.no_grad():
        output_latents, _ = orig_stage2_latents(
            provider, generation_pass, first_latent, condition, camera, rng
        )

    stage2._slice_camera = orig_slice_cam

    # Compute chunk totals
    for c_idx in chunk_step_times:
        steps_dict = chunk_step_times[c_idx]
        total_steps_ms = sum(steps_dict.values())
        chunk_total_times[c_idx] = total_steps_ms

    # Extract chunk 4 latents [1, 3, 48, 30, 54] for VAE decode timing
    chunk4_latent = output_latents[:, 12:15].contiguous()

    # Time VAE decode for chunk 4 (3 latent frames -> 13 pixel frames)
    # 1. Standard BF16 VAE
    sync_device(device)
    # Warmup
    with torch.no_grad():
        _ = provider.vae.decode_streaming(chunk4_latent, chunk_latent_frames=4)
    sync_device(device)

    bf16_vae_runs = []
    for _ in range(5):
        sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = provider.vae.decode_streaming(chunk4_latent, chunk_latent_frames=4)
        sync_device(device)
        bf16_vae_runs.append((time.perf_counter() - t0) * 1000.0)
    bf16_vae_chunk_ms = float(torch.tensor(bf16_vae_runs).median().item())

    # 2. Block-256 QuaRot W8A8 Quantized VAE
    apply_vae_quarot(provider.vae.module.decoder, enabled=True)
    sync_device(device)
    # Warmup
    with torch.no_grad():
        _ = provider.vae.decode_streaming(chunk4_latent, chunk_latent_frames=4)
    sync_device(device)

    quant_vae_runs = []
    for _ in range(5):
        sync_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = provider.vae.decode_streaming(chunk4_latent, chunk_latent_frames=4)
        sync_device(device)
        quant_vae_runs.append((time.perf_counter() - t0) * 1000.0)
    quant_vae_chunk_ms = float(torch.tensor(quant_vae_runs).median().item())
    apply_vae_quarot(provider.vae.module.decoder, enabled=False)

    result = {
        "linear_quant": linear_quant,
        "attn_backend": attn_backend,
        "radial": radial,
        "chunk_dit_total_ms": chunk_total_times,
        "chunk4_steps_ms": chunk_step_times.get(4, {}),
        "chunk4_dit_total_ms": chunk_total_times.get(4, 0.0),
        "chunk4_bf16_vae_ms": bf16_vae_chunk_ms,
        "chunk4_quarot_vae_ms": quant_vae_chunk_ms,
    }
    if return_latents:
        _, _, _, gt_latents = sample_inputs(provider, sample, latent_frames)
        result["latents"] = output_latents.float().cpu()
        result["gt_latents"] = gt_latents.float().cpu()
    del provider
    gc.collect()
    torch.xpu.empty_cache()
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-backends", nargs="+", default=None, help="List of attention backends to evaluate with svdquant_k4 (e.g. sdpa sageattnv1 sageattnv2)")
    parser.add_argument("--output-json", default="outputs/svdquant_eval/eval_153f/chunk4_timing.json")
    args = parser.parse_args()

    if args.attn_backends:
        configs = [("svdquant_k4", ab) for ab in args.attn_backends]
    else:
        configs = [
            ("full_precision", "sdpa"),
            ("svdquant_k4", "sdpa"),
            ("svdquant_k4", "sageattnv1"),
            ("svdquant_k4", "sageattnv2"),
            ("svdquant_k1", "sdpa"),
        ]
    results = []
    for lq, ab in configs:
        print(f"\nEvaluating Linear={lq}, Attn={ab}...")
        res = measure_pipeline(linear_quant=lq, attn_backend=ab)
        results.append(res)
        c4_steps = res["chunk4_steps_ms"]
        print(f"  Chunk 4 Step 0: {c4_steps.get(0, 0):.2f} ms")
        print(f"  Chunk 4 Step 1: {c4_steps.get(1, 0):.2f} ms")
        print(f"  Chunk 4 Step 2: {c4_steps.get(2, 0):.2f} ms")
        print(f"  Chunk 4 Step 3: {c4_steps.get(3, 0):.2f} ms")
        print(f"  Chunk 4 Commit: {c4_steps.get('commit', 0):.2f} ms")
        print(f"  Total DiT Chunk 4: {res['chunk4_dit_total_ms']:.2f} ms")
        total_6chunks = sum(res["chunk_dit_total_ms"].values())
        print(f"  Chunk 0-5 DiT times: {', '.join(f'c{c}: {t:.1f}ms' for c, t in sorted(res['chunk_dit_total_ms'].items()))}")
        print(f"  Total 6-chunk DiT time: {total_6chunks:.2f} ms")
        print(f"  Standard BF16 VAE Chunk 4: {res['chunk4_bf16_vae_ms']:.2f} ms")
        print(f"  QuaRot W8A8 VAE Chunk 4:   {res['chunk4_quarot_vae_ms']:.2f} ms (Speedup: {res['chunk4_bf16_vae_ms'] / res['chunk4_quarot_vae_ms']:.2f}x)")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved timing to {out_path}")
