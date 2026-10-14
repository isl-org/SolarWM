"""Decode locally available Wan2.2-TI2V-5B validation latents to MP4.

This is a VAE-only review tool. It never loads or runs the diffusion model.

Example:
    .venv-wan22-xpu/bin/python scripts/debug/decode_wan22_5b_review.py

The default selection is one deterministic validation window per source dataset
from the locally present 480p/153f test shards.

Use ``--all --include-examples`` to decode every locally present Wan2.2-5B
validation window and the six Wan2.2-5B format examples.
"""

from __future__ import annotations

import argparse
import gzip
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from av import VideoFrame, open as av_open
from safetensors.torch import load as load_safetensors

from solarwm.backends.wan22.runtime.components import Wan5BVAE


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "releases-v1"
DEFAULT_INDEX = (
    DEFAULT_DATA_ROOT
    / "recipes"
    / "clean-153f"
    / "latent-wds"
    / "wan22-ti2v5b-153f-480p-v1"
    / "test-index.jsonl.gz"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "wan22_5b_vae_review"
DEFAULT_VAE = Path("/home/ssheorey/models/SolarWM/SolarWM-5B-base/vae/Wan2.2_VAE.pth")


def _load_rows(index_path: Path, data_root: Path) -> list[dict[str, Any]]:
    with gzip.open(index_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return [
        row
        for row in rows
        if row.get("shard") and (data_root / row["shard"]).is_file()
    ]


def _load_example_rows(data_root: Path) -> list[dict[str, Any]]:
    rows = []
    for shard_path in sorted(
        (data_root / "example" / "latent").glob("wan22-ti2v5b-*.tar")
    ):
        with tarfile.open(shard_path, "r") as archive:
            preencoded = next(
                member
                for member in archive.getmembers()
                if member.name.endswith(".preencoded.safetensors")
            )
            manifest_member = next(
                member
                for member in archive.getmembers()
                if member.name.endswith(".manifest.json")
            )
            manifest = json.load(archive.extractfile(manifest_member))
        rows.append(
            {
                "sample_id": manifest["sample_id"],
                "preencoded_member": preencoded.name,
                "manifest_member": manifest_member.name,
                "shard": str(shard_path.relative_to(data_root)),
                "source_fps": manifest.get("video", {}).get("fps", 16.0),
            }
        )
    return rows


def _select_rows(
    rows: list[dict[str, Any]], limit_per_dataset: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        dataset = str(row["sample_id"]).split("/", 1)[0]
        grouped.setdefault(dataset, []).append(row)

    selected = []
    for dataset in sorted(grouped):
        candidates = sorted(grouped[dataset], key=lambda row: row["sample_id"])
        selected.extend(candidates[:limit_per_dataset])
    return selected


def _load_latents(
    data_root: Path, row: dict[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    shard_path = data_root / row["shard"]
    with tarfile.open(shard_path, "r") as archive:
        member = archive.extractfile(row["preencoded_member"])
        if member is None:
            raise RuntimeError(
                f"{row['sample_id']}: missing {row['preencoded_member']} in {shard_path}"
            )
        payload = load_safetensors(member.read())
        manifest_member = archive.extractfile(row["manifest_member"])
        manifest = json.load(manifest_member) if manifest_member is not None else {}

    latents = payload.get("latents")
    if (
        latents is None
        or latents.ndim != 4
        or int(latents.shape[1]) != 48
        or int(latents.shape[2]) <= 0
        or int(latents.shape[3]) <= 0
    ):
        shape = None if latents is None else tuple(latents.shape)
        raise RuntimeError(f"{row['sample_id']}: unexpected Wan2.2-5B latent shape {shape}")
    return latents.contiguous(), manifest


def _sync(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _write_mp4(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av_open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=int(round(fps)))
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for frame in frames:
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _to_uint8(decoded: torch.Tensor) -> np.ndarray:
    # Wan VAE returns [B, T, C, H, W] in [-1, 1].
    if decoded.ndim != 5 or decoded.shape[0] != 1 or decoded.shape[2] != 3:
        raise RuntimeError(f"unexpected decoded video shape {tuple(decoded.shape)}")
    frames = decoded[0].float().clamp(-1, 1)
    return (
        frames.permute(0, 2, 3, 1)
        .mul(127.5)
        .add(127.5)
        .round()
        .to(torch.uint8)
        .numpy()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vae", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--limit-per-dataset", type=int, default=1)
    parser.add_argument(
        "--all",
        action="store_true",
        help="decode every locally available validation row instead of one per dataset",
    )
    parser.add_argument(
        "--include-examples",
        action="store_true",
        help="also decode the six Wan2.2-5B archives in releases-v1/example/latent",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit_per_dataset < 1:
        parser.error("--limit-per-dataset must be positive")
    if not args.vae.is_file():
        parser.error(f"VAE weights not found: {args.vae}")

    rows = _load_rows(args.index, args.data_root)
    if args.all:
        selected = sorted(rows, key=lambda row: row["sample_id"])
    else:
        selected = _select_rows(rows, args.limit_per_dataset)
    if args.include_examples:
        selected.extend(_load_example_rows(args.data_root))
    if not selected:
        parser.error("no locally available validation rows found")

    device = torch.device(args.device)
    if device.type == "xpu" and not torch.xpu.is_available():
        parser.error("XPU requested but torch.xpu is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Selected {len(selected)} validation clips from {len(rows)} locally available rows")
    print(f"Loading Wan2.2-5B VAE: {args.vae}")
    vae = Wan5BVAE(args.vae, xpu_channels_last=device.type == "xpu")
    vae.to(device, dtype=torch.bfloat16)

    manifest: dict[str, Any] = {
        "decoder": "Wan5BVAE",
        "vae": str(args.vae),
        "device": str(device),
        "source_index": str(args.index),
        "latent_format": "Wan2.2-TI2V-5B [T,48,H,W] bfloat16",
        "fps": 16.0,
        "videos": [],
    }
    try:
        with torch.no_grad():
            for number, row in enumerate(selected, start=1):
                safe_id = str(row["sample_id"]).replace("/", "__")
                output_path = args.output_dir / f"{safe_id}.mp4"
                if output_path.exists() and not args.overwrite:
                    print(f"[{number}/{len(selected)}] exists: {output_path.name}")
                    manifest["videos"].append(
                        {
                            "sample_id": row["sample_id"],
                            "output": str(output_path),
                            "status": "existing",
                        }
                    )
                    continue

                print(f"[{number}/{len(selected)}] decoding {row['sample_id']}")
                latents, sample_manifest = _load_latents(args.data_root, row)
                latents = latents.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                _sync(device)
                decoded = vae.decode_streaming(latents, chunk_latent_frames=60)
                _sync(device)
                frames = _to_uint8(decoded)
                if not np.isfinite(frames).all():
                    raise RuntimeError(f"{row['sample_id']}: decoded frames are non-finite")
                _write_mp4(output_path, frames, float(row.get("source_fps", 16.0)))
                del latents, decoded
                if device.type == "xpu":
                    torch.xpu.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

                prompt = sample_manifest.get("prompt", {}).get("text")
                manifest["videos"].append(
                    {
                        "sample_id": row["sample_id"],
                        "output": str(output_path),
                        "frames": int(frames.shape[0]),
                        "height": int(frames.shape[1]),
                        "width": int(frames.shape[2]),
                        "prompt": prompt,
                        "status": "decoded",
                    }
                )
                print(f"  wrote {output_path} ({frames.shape[0]} frames)")
    finally:
        del vae
        if device.type == "xpu":
            torch.xpu.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote review manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
