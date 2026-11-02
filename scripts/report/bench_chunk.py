"""Steady-state per-chunk timing of Wan2.2 TI2V-5B Stage2 on one XPU.

Measures one 3-latent chunk (12 pixel frames): DiT time for the 4 denoise and
1 commit forwards, and the streaming VAE decode time of the same chunk. Only
APIs that exist unchanged from the first XPU port commit (001e586) to HEAD are
used, so the same script measures every commit of the optimization chain:

    git worktree add outputs/report_worktrees/<commit> <commit>
    cd outputs/report_worktrees/<commit>
    PYTHONPATH=src python /abs/path/to/HEAD/scripts/report/bench_chunk.py --set ...

The text encoder is never loaded: prompt embeddings, the first latent, and the
camera trajectory come from one preencoded dataset sample, so every commit
denoises the identical input.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import tarfile
import time
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from safetensors.torch import load as load_safetensors

from solarwm.backends.wan22.runtime import stage2
from solarwm.backends.wan22.runtime.data import CameraGuards, build_camera_tokens
from solarwm.config import load_config

DEFAULT_SHARD = "/home/ssheorey/data/SolarWM-Data/releases-v1/latent-wds/wan22-ti2v5b-153f-480p-v1/abot/shards/kept-high-000008.tar"
ASSET_OVERRIDES = [
    "model.base_path=/home/ssheorey/models/SolarWM/SolarWM-5B-base",
    "checkpoint.path=/home/ssheorey/models/SolarWM/SolarWM-5B-sgf-stage2-81f",
    "data.index_root=/home/ssheorey/data/SolarWM-Data/releases-v1",
    "data.transport.root=/home/ssheorey/data/SolarWM-Data/releases-v1",
    "inference.device=xpu",
]
RES_320P = [
    "model.frame_sequence_length=180",
    "data.latent_shape=[21, 48, 20, 36]",
    "data.height=320",
    "data.width=576",
]
FORWARDS_PER_CHUNK = 5


class _NoTextEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        return self


def _first_sample(shard: Path) -> dict:
    suffixes = (".preencoded.safetensors", ".camera.npz", ".manifest.json")
    pending: dict[str, dict[str, bytes]] = {}
    with tarfile.open(shard) as tar:
        for member in tar:
            suffix = next((s for s in suffixes if member.name.endswith(s)), None)
            if suffix is None:
                continue
            entry = pending.setdefault(member.name[: -len(suffix)], {})
            entry[suffix] = tar.extractfile(member).read()
            if len(entry) == len(suffixes):
                tensors = load_safetensors(entry[".preencoded.safetensors"])
                manifest = json.loads(entry[".manifest.json"])
                return {
                    "latents": tensors["latents"],
                    "prompt_embeds": tensors["prompt_embeds"],
                    "camera_payload": entry[".camera.npz"],
                    "manifest": manifest,
                    "start_frame": int(manifest["preencoding"]["start_frame"]),
                    "pixel_frames": int(manifest["preencoding"]["pixel_frames"]),
                }
    raise RuntimeError(f"no complete sample in {shard}")


def _inputs(provider, sample: dict, latent_frames: int):
    config = provider.config
    data = config["data"]
    frame_tokens = int(config["model"]["frame_sequence_length"])
    start = sample["start_frame"]
    camera = build_camera_tokens(
        sample["camera_payload"],
        tuple(range(start, start + sample["pixel_frames"])),
        sample["manifest"],
        source_fps=float(sample["manifest"]["video"].get("fps") or 16.0),
        output_fps=float(data.get("fps", 16.0)),
        frame_sequence_length=frame_tokens,
        guards=CameraGuards(
            max_rel_translation=float(data["max_rel_translation"]),
            max_camera_abs=float(data["max_camera_abs"]),
        ),
        configured_array_key=str(data["camera_array_key"]),
    )
    tokens = latent_frames * frame_tokens
    first_latent = sample["latents"][:1].unsqueeze(0).to(provider.device, torch.bfloat16)
    height, width = (int(v) for v in data["latent_shape"][-2:])
    if tuple(first_latent.shape[-2:]) != (height, width):
        first_latent = F.interpolate(first_latent[0].float(), size=(height, width), mode="bilinear").unsqueeze(0).to(torch.bfloat16)
    condition = {"prompt_embeds": sample["prompt_embeds"].unsqueeze(0).to(provider.device, torch.bfloat16)}
    camera = {
        "viewmats": camera["viewmats"][:tokens].unsqueeze(0).to(provider.device, torch.float32),
        "K": camera["K"][:tokens].unsqueeze(0).to(provider.device, torch.float32),
    }
    return first_latent.contiguous(), condition, camera


class _TimedDiffusion:
    def __init__(self, inner, times: list[float], profile_chunk: int | None):
        self.inner = inner
        self.times = times
        self.profile_chunk = profile_chunk
        self.profiler = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        if self.profile_chunk is not None and len(self.times) == FORWARDS_PER_CHUNK * self.profile_chunk:
            self.profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU])
            self.profiler.__enter__()
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        out = self.inner(*args, **kwargs)
        torch.xpu.synchronize()
        self.times.append((time.perf_counter() - t0) * 1000.0)
        if self.profiler is not None and len(self.times) == FORWARDS_PER_CHUNK * (self.profile_chunk + 1):
            self.profiler.__exit__(None, None, None)
        return out


def run(overrides: list[str], config_path: str, shard: Path, chunks: int, steady_from: int, vae_repeats: int, profile_chunk: int | None = None, save_frames: Path | None = None) -> dict:
    stage2.WanTextEncoder = _NoTextEncoder
    values = load_config(config_path, ASSET_OVERRIDES + overrides).values
    provider = stage2.build_stage2_generation_provider(values)
    provider._load_role(str(provider.plan.passes[0].weights))

    latent_frames = 3 * chunks
    fields = {f.name for f in dataclasses.fields(provider.plan.passes[0])}
    horizon = {k: latent_frames for k in ("rollout_latent_frames", "min_rollout_latent_frames") if k in fields}
    generation_pass = dataclasses.replace(provider.plan.passes[0], **horizon)
    first_latent, condition, camera = _inputs(provider, _first_sample(shard), latent_frames)

    forward_ms: list[float] = []
    timed = _TimedDiffusion(provider.diffusion, forward_ms, profile_chunk)
    provider.diffusion = timed
    rng = torch.Generator(device=provider.device).manual_seed(1234)
    torch.xpu.reset_peak_memory_stats()
    with torch.no_grad():
        latents, _ = stage2._stage2_self_forcing_latents(provider, generation_pass, first_latent, condition, camera, rng)
    if len(forward_ms) != FORWARDS_PER_CHUNK * chunks:
        raise RuntimeError(f"expected {FORWARDS_PER_CHUNK * chunks} forwards, timed {len(forward_ms)}")
    kernel_table = None
    if timed.profiler is not None:
        kernel_table = timed.profiler.key_averages().table(sort_by="self_xpu_time_total", row_limit=40)
    dit_ms = [sum(forward_ms[FORWARDS_PER_CHUNK * c : FORWARDS_PER_CHUNK * (c + 1)]) for c in range(chunks)]

    # Every repeat streams all chunks through one continuous causal cache, as in
    # production; the first repeat is warmup.
    vae_ms: list[list[float]] = []
    frames = []
    with torch.no_grad():
        for repeat in range(vae_repeats + 1):
            tiles = []
            stream = provider.vae.decode_streaming_chunks(latents, chunk_latent_frames=3)
            while True:
                torch.xpu.synchronize()
                t0 = time.perf_counter()
                tile = next(stream, None)
                torch.xpu.synchronize()
                if tile is None:
                    break
                tiles.append((time.perf_counter() - t0) * 1000.0)
                if save_frames is not None and repeat == vae_repeats:
                    frames.append(tile)
                del tile
            vae_ms.append(tiles)
    vae_ms = vae_ms[1:]
    if save_frames is not None:
        save_frames.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.cat(frames, dim=1), save_frames)

    steady = range(steady_from, chunks)
    dit = median(dit_ms[c] for c in steady)
    vae = median(run[c] for run in vae_ms for c in steady)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {
        "commit": commit,
        "config": config_path,
        "overrides": overrides,
        "chunks": chunks,
        "steady_chunks": list(steady),
        "dit_chunk_ms_per_chunk": dit_ms,
        "vae_tile_ms_per_repeat": vae_ms,
        "dit_chunk_ms": dit,
        "vae_tile_ms": vae,
        "chunk_ms": dit + vae,
        "fps": 12.0 / ((dit + vae) / 1000.0),
        "peak_allocated_gib": torch.xpu.max_memory_allocated() / 2**30,
        "dit_kernel_table": kernel_table,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml")
    parser.add_argument("--set", action="append", default=[], help="config override KEY=VALUE (repeatable)")
    parser.add_argument("--res-320p", action="store_true", help="run at 320x576 instead of 480x864")
    parser.add_argument("--shard", type=Path, default=Path(DEFAULT_SHARD))
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--steady-from", type=int, default=5, help="first chunk whose 18-latent KV window is full")
    parser.add_argument("--vae-repeats", type=int, default=2)
    parser.add_argument("--profile-chunk", type=int, default=None, help="store a DiT kernel table for this chunk (its timings then include profiler overhead)")
    parser.add_argument("--save-frames", type=Path, default=None, help="save the decoded BTCHW video of the last VAE repeat (torch.save)")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    overrides = (RES_320P if args.res_320p else []) + args.set
    result = run(overrides, args.config, args.shard, args.chunks, args.steady_from, args.vae_repeats, args.profile_chunk, args.save_frames)
    if result["dit_kernel_table"]:
        print(result["dit_kernel_table"])
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(f"{result['commit']} {' '.join(overrides)}: DiT {result['dit_chunk_ms']:.1f} ms, VAE {result['vae_tile_ms']:.1f} ms, chunk {result['chunk_ms']:.1f} ms ({result['fps']:.2f} fps)")


if __name__ == "__main__":
    main()
