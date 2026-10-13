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
    max_shard_size_gb: float = 4.0,
) -> Path:
    """Export quantized tensors across weight sets to sharded safetensors and write manifest.json.

    Args:
        output_dir: Destination directory
        quantized_sets: Mapping of weight set tag (e.g. "set0", "set1") -> {tensor_key: tensor}
        manifest: Metadata and activation clipping policy dictionary
        max_shard_size_gb: Maximum size per shard in GB

    Returns:
        Path to generated manifest.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_tensors: dict[str, torch.Tensor] = {}
    for set_name, tensors in quantized_sets.items():
        for k, v in tensors.items():
            full_key = f"{set_name}.{k}"
            all_tensors[full_key] = v.contiguous()

    # Determine sharding if large, or single shard if <= max_shard_size_gb
    total_bytes = sum(t.element_size() * t.nelement() for t in all_tensors.values())
    max_bytes = int(max_shard_size_gb * (1024**3))

    shard_files: list[str] = []
    weight_map: dict[str, str] = {}

    if total_bytes <= max_bytes:
        shard_name = "svdquant_weights.safetensors"
        save_file(all_tensors, str(out / shard_name))
        shard_files.append(shard_name)
        for k in all_tensors.keys():
            weight_map[k] = shard_name
    else:
        # Shard by size
        current_shard: dict[str, torch.Tensor] = {}
        current_bytes = 0
        shard_idx = 0

        for k, v in all_tensors.items():
            t_bytes = v.element_size() * v.nelement()
            if current_bytes + t_bytes > max_bytes and current_shard:
                s_name = f"svdquant_weights-{shard_idx:05d}-of-shards.safetensors"
                save_file(current_shard, str(out / s_name))
                shard_files.append(s_name)
                current_shard = {}
                current_bytes = 0
                shard_idx += 1

            current_shard[k] = v
            current_bytes += t_bytes

        if current_shard:
            s_name = f"svdquant_weights-{shard_idx:05d}-of-shards.safetensors"
            save_file(current_shard, str(out / s_name))
            shard_files.append(s_name)

        # Update shards total count in filenames
        num_shards = len(shard_files)
        final_shard_files: list[str] = []
        for i, old_name in enumerate(shard_files):
            new_name = f"svdquant_weights-{i:05d}-of-{num_shards:05d}.safetensors"
            (out / old_name).rename(out / new_name)
            final_shard_files.append(new_name)

        # Build weight map
        for s_name in final_shard_files:
            keys = load_file(str(out / s_name)).keys()
            for k in keys:
                weight_map[k] = s_name

    manifest_content = dict(manifest)
    manifest_content["shards"] = final_shard_files if total_bytes > max_bytes else shard_files
    manifest_content["weight_map"] = weight_map

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_content, f, indent=2)

    return manifest_path


def load_svdquant_manifest(manifest_path: Path | str) -> dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
