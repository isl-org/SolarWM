import json

import pytest
import torch
from safetensors.torch import save_file

from solarwm.checkpoint.svdquant import validate_checkpoint


@pytest.mark.parametrize("prefixed", [False, True])
def test_release_and_canonical_names(tmp_path, prefixed):
    directory = tmp_path / "sdpa_k1"
    directory.mkdir()
    prefix = "sdpa_k1_" if prefixed else ""
    key = "set0.blocks.0.ffn.0.qweight"
    save_file({key: torch.zeros(2, 2)}, directory / f"{prefix}weights.safetensors")
    manifest = {
        "format_version": "solarwm-svdquant-v1", "num_sets": 1,
        "set_mapping": {"0": "set0", "commit": "set0"},
        "shards": ["weights.safetensors"], "weight_map": {key: "weights.safetensors"},
    }
    (directory / f"{prefix}manifest.json").write_text(json.dumps(manifest))
    result, paths = validate_checkpoint(directory)
    assert result == manifest
    assert paths[0].name == f"{prefix}weights.safetensors"


def test_missing_set_shard_is_explicit(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format_version": "solarwm-svdquant-v1", "num_sets": 1,
        "shards": ["missing.safetensors"], "weight_map": {"set0.w": "missing.safetensors"},
        "set_mapping": {"0": "set0"},
    }))
    with pytest.raises(FileNotFoundError, match="shard missing"):
        validate_checkpoint(tmp_path)


def test_missing_tensor_is_rejected(tmp_path):
    save_file({"set0.w": torch.zeros(1)}, tmp_path / "weights.safetensors")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format_version": "solarwm-svdquant-v1", "num_sets": 1,
        "shards": ["weights.safetensors"], "set_mapping": {"0": "set0"},
        "weight_map": {"set0.w": "weights.safetensors", "set0.missing": "weights.safetensors"},
    }))
    with pytest.raises(ValueError, match="inventory differs"):
        validate_checkpoint(tmp_path)


def test_duplicate_tensor_is_rejected(tmp_path):
    for name in ("a.safetensors", "b.safetensors"):
        save_file({"set0.w": torch.zeros(1)}, tmp_path / name)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format_version": "solarwm-svdquant-v1", "num_sets": 1,
        "shards": ["a.safetensors", "b.safetensors"], "set_mapping": {"0": "set0"},
        "weight_map": {"set0.w": "a.safetensors"},
    }))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_checkpoint(tmp_path)
