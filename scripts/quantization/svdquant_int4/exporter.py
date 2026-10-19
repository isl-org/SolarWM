"""Safetensors export and manifest serialization for SVDQuant INT4 checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def export_svdquant_safetensors(
    output_dir: Path | str,
    quantized_sets: dict[str, dict[str, torch.Tensor]],
    manifest: dict[str, Any],
    max_shard_size_gb: float | None = None,
) -> Path:
    """Export quantized tensors across weight sets to per-set safetensors and write manifest.json.

    Each noise-level checkpoint set (e.g. set0, set1) is saved to its own .safetensors file (~1.83 GiB / 1.96 GB),
    ensuring each file stays under GitHub release's 2 GB per-asset limit without arbitrary sharding:
      - k1 (1 checkpoint): svdquant_weights.safetensors
      - k2 (2 checkpoints): svdquant_weights_set0.safetensors, svdquant_weights_set1.safetensors
      - k4 (4 checkpoints): svdquant_weights_set0.safetensors ... svdquant_weights_set3.safetensors

    Args:
        output_dir: Destination directory
        quantized_sets: Mapping of weight set tag (e.g. "set0", "set1") -> {tensor_key: tensor}
        manifest: Metadata and activation clipping policy dictionary
        max_shard_size_gb: Deprecated/unused. Retained for signature compatibility.

    Returns:
        Path to generated manifest.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shard_files: list[str] = []
    weight_map: dict[str, str] = {}

    for set_name in sorted(quantized_sets.keys()):
        tensors = quantized_sets[set_name]
        shard_name = (
            "svdquant_weights.safetensors"
            if len(quantized_sets) == 1
            else f"svdquant_weights_{set_name}.safetensors"
        )
        set_tensors = {f"{set_name}.{k}": v.contiguous() for k, v in tensors.items()}
        save_file(set_tensors, str(out / shard_name))
        shard_files.append(shard_name)
        for k in set_tensors.keys():
            weight_map[k] = shard_name

    manifest_content = dict(manifest)
    manifest_content["shards"] = shard_files
    manifest_content["weight_map"] = weight_map

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_content, f, indent=2)

    return manifest_path


def reshard_svdquant_directory(directory: Path | str) -> list[str]:
    """Convert an existing SVDQuant directory to the 1-checkpoint-per-set safetensors format.

    Reads all existing *.safetensors in `directory`, groups keys by set prefix (e.g. "set0", "set1"),
    writes out svdquant_weights.safetensors (for 1 set) or svdquant_weights_{set}.safetensors (for >1 sets),
    removes obsolete shard files, and updates manifest.json.

    Returns:
        List of generated shard filenames.
    """
    target = Path(directory)
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found in {target}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    existing_shards = sorted(target.glob("*.safetensors"))
    if not existing_shards:
        raise FileNotFoundError(f"No safetensors found in {target}")

    # Load all tensors grouped by set
    sets: dict[str, dict[str, torch.Tensor]] = {}
    for shard in existing_shards:
        data = load_file(str(shard), device="cpu")
        for key, tensor in data.items():
            set_name, mod_key = key.split(".", 1)
            sets.setdefault(set_name, {})[mod_key] = tensor

    # Remove old safetensors files before writing standardized shards
    for shard in existing_shards:
        shard.unlink()

    # Re-export with standardized layout
    export_svdquant_safetensors(target, sets, manifest)
    with open(manifest_path, "r", encoding="utf-8") as f:
        updated = json.load(f)
    return list(updated.get("shards", []))


def load_svdquant_manifest(manifest_path: Path | str) -> dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
