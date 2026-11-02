"""Generate short clips for no-reference VQA on the local Wan test split."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import gzip
import json
import random
import subprocess
import tarfile
from pathlib import Path

import torch
from safetensors.torch import load as load_safetensors

import bench_chunk
from solarwm.backends.wan22.runtime import stage2
from solarwm.config import load_config


REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "data/releases-v1/recipes/clean-153f/latent-wds/wan22-ti2v5b-153f-480p-v1/test-index.jsonl.gz"
CONFIG_PATH = "configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml"
OUT = REPO / "outputs/report_chain/vqa_153f"
FPS = 16
LATENT_FRAMES = 39
EXAMPLE_SEED = 20261024

CONFIGS = [
    ("bf16", []),
    ("sage1", ["model.attention_backend=sageattnv1"]),
    ("sage2", ["model.attention_backend=sageattnv2"]),
    ("svdq_k1", ["model.linear_quant=svdquant_k1"]),
    ("svdq_k2", ["model.linear_quant=svdquant_k2"]),
    ("svdq", ["model.linear_quant=svdquant_k4"]),
    ("w8a8", ["runtime.stage2_vae_quant=quarot_w8a8"]),
    ("svdq_w8a8", ["model.linear_quant=svdquant_k4", "runtime.stage2_vae_quant=quarot_w8a8"]),
    ("sage1_svdq", ["model.attention_backend=sageattnv1", "model.linear_quant=svdquant_k4"]),
    ("sage1_svdq_w8a8", ["model.attention_backend=sageattnv1", "model.linear_quant=svdquant_k4", "runtime.stage2_vae_quant=quarot_w8a8"]),
]


def _read_rows() -> list[dict]:
    rows = []
    with gzip.open(INDEX, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            shard = REPO / "data/releases-v1" / row["shard"]
            if shard.exists():
                row = dict(row)
                row["source"] = Path(row["shard"]).parts[-3]
                rows.append(row)
    return rows


def _examples(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    sources = rng.sample(sorted(by_source), min(4, len(by_source)))
    return [rng.choice(by_source[source]) for source in sources]


def _samples_for_shard(shard: Path, rows: list[dict]):
    wanted = {}
    for row in rows:
        wanted[row["preencoded_member"]] = (row, "preencoded")
        wanted[row["camera_member"]] = (row, "camera")
        wanted[row["manifest_member"]] = (row, "manifest")
    pending: dict[str, dict[str, bytes]] = {}
    with tarfile.open(shard) as tar:
        for member in tar:
            match = wanted.get(member.name)
            if match is None:
                continue
            row, kind = match
            value = tar.extractfile(member).read()
            entry = pending.setdefault(row["key"], {})
            entry[kind] = value
            if len(entry) != 3:
                continue
            tensors = load_safetensors(entry["preencoded"])
            manifest = json.loads(entry["manifest"])
            yield {
                "key": row["key"],
                "sample_id": row["sample_id"],
                "source": row["source"],
                "latents": tensors["latents"],
                "prompt_embeds": tensors["prompt_embeds"],
                "camera_payload": entry["camera"],
                "manifest": manifest,
                "start_frame": int(row["start_frame"]),
                "pixel_frames": int(manifest["preencoding"]["pixel_frames"]),
            }
            del pending[row["key"]]


def _iter_samples(rows: list[dict]):
    by_shard: dict[Path, list[dict]] = {}
    for row in rows:
        by_shard.setdefault(REPO / "data/releases-v1" / row["shard"], []).append(row)
    for shard, shard_rows in by_shard.items():
        yield from _samples_for_shard(shard, shard_rows)


def _safe_key(key: str) -> str:
    return key.replace("/", "__")


def _writer(path: Path, crf: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "864x480",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv444p" if crf == 0 else "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def _write_tile(writer, tile: torch.Tensor) -> None:
    frames = (
        ((tile[0].float().clamp(-1, 1) + 1) * 127.5)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .numpy()
    )
    writer.stdin.write(frames.tobytes())


def _close_writer(writer, path: Path) -> None:
    writer.stdin.close()
    if writer.wait() != 0:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {path}")


def _build_provider(overrides: list[str]):
    stage2.WanTextEncoder = bench_chunk._NoTextEncoder
    values = load_config(CONFIG_PATH, bench_chunk.ASSET_OVERRIDES + overrides).values
    provider = stage2.build_stage2_generation_provider(values)
    provider._load_role(str(provider.plan.passes[0].weights))
    fields = {field.name for field in dataclasses.fields(provider.plan.passes[0])}
    horizon = {
        key: LATENT_FRAMES
        for key in ("rollout_latent_frames", "min_rollout_latent_frames")
        if key in fields
    }
    return provider, dataclasses.replace(provider.plan.passes[0], **horizon)


def _decode_video(provider, latents, writers: tuple | None):
    first_writer, last_writer, full_writer = writers or (None, None, None)
    last_tile = None
    for tile_number, tile in enumerate(provider.vae.decode_streaming_chunks(latents, chunk_latent_frames=3)):
        if tile_number == 0 and first_writer is not None:
            _write_tile(first_writer, tile[:, :9])
        if full_writer is not None:
            _write_tile(full_writer, tile)
        last_tile = tile
    if last_writer is not None:
        _write_tile(last_writer, last_tile)


def _run_sample(provider, generation_pass, sample: dict, paths: dict[str, Path], is_example: bool, ground_truth: bool = False):
    if ground_truth:
        latents = sample["latents"].unsqueeze(0).to(provider.device, torch.bfloat16)
    else:
        first_latent, condition, camera = bench_chunk._inputs(provider, sample, LATENT_FRAMES)
        rng = torch.Generator(device=provider.device).manual_seed(1234)
        with torch.no_grad():
            latents, _ = stage2._stage2_self_forcing_latents(
                provider,
                generation_pass,
                first_latent,
                condition,
                camera,
                rng,
            )

    first_writer = _writer(paths["first"], 0)
    last_writer = _writer(paths["last"], 0)
    full_writer = _writer(paths["full"], 18) if is_example and "full" in paths else None
    try:
        _decode_video(provider, latents, (first_writer, last_writer, full_writer))
        _close_writer(first_writer, paths["first"])
        _close_writer(last_writer, paths["last"])
        if full_writer is not None:
            _close_writer(full_writer, paths["full"])
    except BaseException:
        for writer in (first_writer, last_writer, full_writer):
            if writer is not None and writer.poll() is None:
                writer.kill()
                writer.wait()
        for path in paths.values():
            path.unlink(missing_ok=True)
        raise
    finally:
        del latents


def _clip_paths(config: str, key: str, example: bool, ground_truth: bool) -> dict[str, Path]:
    safe = _safe_key(key)
    if ground_truth:
        root = OUT / "ground_truth"
    else:
        root = OUT / config
    paths = {
        "first": root / f"{safe}_first.mp4",
        "last": root / f"{safe}_last.mp4",
    }
    if example:
        paths["full"] = OUT / "examples" / f"{safe}_{'ground_truth' if ground_truth else config}.mp4"
    return paths


def _needs(paths: dict[str, Path]) -> bool:
    return any(not path.exists() for path in paths.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(name for name, _ in CONFIGS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ground-truth", action="store_true")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--example-seed", type=int, default=EXAMPLE_SEED)
    args = parser.parse_args()

    rows = _read_rows()
    selected = _examples(rows, args.example_seed)
    example_keys = {row["key"] for row in selected[: args.examples]}
    (OUT / "examples.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "examples.json").write_text(
        json.dumps(
            {
                "seed": args.example_seed,
                "samples": [
                    {"key": row["key"], "sample_id": row["sample_id"], "source": row["source"]}
                    for row in selected[: args.examples]
                ],
            },
            indent=2,
        )
    )

    wanted_configs = set(args.configs.split(","))
    configs = [(name, overrides) for name, overrides in CONFIGS if name in wanted_configs]
    if len(configs) != len(wanted_configs):
        raise ValueError(f"unknown config: {wanted_configs - {name for name, _ in CONFIGS}}")

    samples = list(_iter_samples(rows))
    if len(samples) != len(rows):
        raise RuntimeError(f"index has {len(rows)} local rows but only read {len(samples)} complete samples")
    (OUT / "samples.json").write_text(
        json.dumps(
            [
                {"key": sample["key"], "sample_id": sample["sample_id"], "source": sample["source"]}
                for sample in samples
            ],
            indent=2,
        )
    )
    if args.limit is not None:
        samples = samples[: args.limit]
    by_key = {sample["key"]: sample for sample in samples}

    if args.ground_truth:
        configs = [("bf16", [])]
    for config, overrides in configs:
        print(f"[{config}] loading provider", flush=True)
        provider, generation_pass = _build_provider(overrides)
        for number, sample in enumerate(samples, 1):
            is_example = sample["key"] in example_keys
            paths = _clip_paths(config, sample["key"], is_example, args.ground_truth)
            if not _needs(paths):
                print(f"[{config}] {number}/{len(samples)} skip {sample['key']}", flush=True)
                continue
            try:
                _run_sample(provider, generation_pass, sample, paths, is_example, args.ground_truth)
                print(f"[{config}] {number}/{len(samples)} done {sample['key']}", flush=True)
            except Exception as exc:
                print(f"[{config}] {number}/{len(samples)} FAIL {sample['key']}: {exc!r}", flush=True)
            gc.collect()
        del generation_pass, provider
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    if args.ground_truth:
        print(f"ground truth complete: {len(by_key)} samples", flush=True)


if __name__ == "__main__":
    main()
