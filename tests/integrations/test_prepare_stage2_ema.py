import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from solarwm.runtime.serialization import canonical_json_bytes


def test_prepare_preserves_ema_and_original(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "prepare_stage2_ema.py"
    spec = importlib.util.spec_from_file_location("prepare_stage2_ema", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    source.mkdir()
    ema = torch.tensor([1.125, 2.25], dtype=torch.float32)
    torch.save({"generator": {"w": ema + 1}, "generator_ema": {"w": ema},
                "config": {"test": True}}, source / "model.pt")
    manifest = {
        "schema": "solarwm.checkpoint.v2", "step": 1,
        "contract": {"family": "wan22_ti2v_5b", "stage": "stage2",
                     "causal_mode": "self_gradient_forcing", "objective": "flow_matching",
                     "objective_variant": "sgf_v1", "camera_translation_transform": "linear",
                     "parameterization": "full-parameter-live-ema", "sp_size": 1,
                     "data_generation": "inference-portable:v1"},
        "required_components": ["model.pt"],
        "files": [{"path": "model.pt", "size": (source / "model.pt").stat().st_size}],
        "metadata": {"available_weights": ["live", "ema"]},
    }
    data = canonical_json_bytes(manifest)
    (source / "checkpoint-manifest.json").write_bytes(data)
    (source / "COMPLETE.json").write_bytes(canonical_json_bytes({
        "schema": "solarwm.checkpoint-complete.v2", "step": 1,
        "manifest_digest": hashlib.blake2s(data).hexdigest(),
    }))
    (source / "release-manifest.json").write_text(json.dumps({
        "load": {"default_weights": "ema"},
        "identity": {"model": {"weight_role": "live+ema"}},
    }))
    target = tmp_path / "ema"
    assert module.prepare(source, target) == target
    result = torch.load(target / "model.pt", weights_only=True, mmap=True)
    assert "generator" not in result
    assert torch.equal(result["generator_ema"]["w"], ema.to(torch.bfloat16))
    assert result["config"] == {"test": True}
    assert "generator" in torch.load(source / "model.pt", weights_only=True)
    assert module.prepare(source, target) == target
