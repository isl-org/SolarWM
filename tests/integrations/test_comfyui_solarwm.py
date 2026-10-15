from __future__ import annotations

import json
from queue import Queue

import numpy as np
import pytest
import torch

from integrations.comfyui_solarwm.factory import _interactive_config
from integrations.comfyui_solarwm.manager import (
    SessionState,
    _VideoRecorder,
    _jpeg_bytes,
)
from integrations.comfyui_solarwm.node import SolarWMInteractive
from solarwm.backends.wan22.runtime.interactive_camera import FreeFlyCamera


class _Runtime:
    done = False
    latent_index = 0
    emitted_pixel_frames = 0
    provider = None

    def close(self):
        pass


def test_session_controls_are_monotonic_and_focus_safe():
    session = SessionState(
        "test",
        _Runtime(),
        FreeFlyCamera(),
        object(),
        events=Queue(),
    )
    session.control({"sequence": 3, "pressed": ["KeyW"]})
    session.control({"sequence": 2, "pressed": ["KeyS"]})
    assert session.controls == {"KeyW"}
    assert session.control_sequence == 3
    session.control({"sequence": 4, "action": "brake", "pressed": []})
    assert np.allclose(session.camera._velocity, 0.0)
    session.control({"sequence": 5, "action": "drop", "pressed": ["KeyA"]})
    assert not session.controls
    assert np.allclose(session.camera._velocity, 0.0)


def test_interactive_node_exposes_stage2_runtime_options():
    required = SolarWMInteractive.INPUT_TYPES()["required"]
    assert required["seed"][1].get("control_after_generate") is True
    assert required["attention_backend"][1]["options"] == ["sdpa", "sageattnv1"]
    assert required["linear_quant"][1]["options"] == [
        "full_precision",
        "svdquant_k1",
        "svdquant_k2",
        "svdquant_k4",
    ]
    assert required["stage2_vae_quant"][1]["options"] == [
        "full_precision",
        "quarot_w8a8",
    ]
    assert required["camera_motion"][1]["options"] == [
        "straight_about_turn_10s",
        "circle_twice_10s",
        "manual_only",
    ]
    assert required["camera_motion"][1]["default"] == "straight_about_turn_10s"
    assert required["translation_speed"][1]["default"] == 0.1
    assert required["rotation_speed"][1]["default"] == 0.1
    assert required["display_scale"][1]["options"] == ["1x", "2x", "fit"]
    assert required["display_scale"][1]["default"] == "2x"
    assert required["display_fps"][1]["options"] == [4, 8, 12, 16, 24, 30]
    assert required["display_fps"][1]["default"] == 8

    values = SolarWMInteractive().build_config(
        "config.yaml",
        "anchor.webp",
        "prompt",
        42,
        120,
        "sageattnv1",
        "svdquant_k2",
        "/weights",
        "quarot_w8a8",
        "straight_about_turn_10s",
        0.5,
        0.393,
    )
    payload = json.loads(values[0])
    assert payload["attention_backend"] == "sageattnv1"
    assert payload["linear_quant"] == "svdquant_k2"
    assert payload["svdquant_weights_root"] == "/weights"
    assert payload["stage2_vae_quant"] == "quarot_w8a8"
    assert payload["camera_motion"] == "straight_about_turn_10s"
    assert payload["translation_speed"] == 0.5
    assert payload["rotation_speed"] == 0.393
    assert payload["display_scale"] == "2x"
    assert payload["display_fps"] == 8


def test_interactive_config_applies_and_validates_stage2_options():
    base = {"model": {}, "runtime": {}, "checkpoint": {"path": "/checkpoint"}}
    config = _interactive_config(
        {
            "attention_backend": "sageattnv1",
            "linear_quant": "svdquant_k4",
            "svdquant_weights_root": "/weights",
            "stage2_vae_quant": "quarot_w8a8",
        },
        base,
    )
    assert config["model"] == {
        "attention_backend": "sageattnv1",
        "linear_quant": "svdquant_k4",
        "svdquant_weights_root": "/weights",
    }
    assert config["runtime"] == {"stage2_vae_quant": "quarot_w8a8"}

    with pytest.raises(ValueError, match="unsupported linear_quant"):
        _interactive_config({"linear_quant": "int8"}, base)


def test_jpeg_encoder_accepts_decoded_channel_first_frames():
    import torch
    from PIL import Image
    from io import BytesIO

    pixels = torch.zeros((1, 1, 3, 2, 8), dtype=torch.float32)
    encoded = _jpeg_bytes(pixels)
    with Image.open(BytesIO(encoded)) as image:
        assert image.mode == "RGB"
        assert image.size == (8, 2)


def test_interactive_video_recorder_writes_replayable_mp4(tmp_path):
    import av
    import torch

    path = tmp_path / "interactive.mp4"
    pixels = torch.zeros((1, 2, 3, 4, 6), dtype=torch.float32)
    pixels[0, 1, 0].fill_(1.0)
    recorder = _VideoRecorder(path, 16.0)
    recorder.write(pixels, 0)
    recorder.write(pixels, 1)
    recorder.close()

    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        frames = list(container.decode(video=0))
    assert stream.width == 6
    assert stream.height == 4
    assert len(frames) == 2


def test_workflow_template_widgets_values_aligned():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    template_paths = [
        repo_root / "integrations/comfyui_solarwm/SolarWM_Interactive_Camera.json",
        repo_root / "integrations/comfyui_solarwm/example_workflows/SolarWM_Interactive_Camera.json",
        Path("/home/ssheorey/code/ComfyUI/user/default/workflows/SolarWM_Interactive_Camera.json"),
    ]

    for path in template_paths:
        if not path.exists():
            continue
        workflow = json.loads(path.read_text(encoding="utf-8"))
        node = next(n for n in workflow["nodes"] if n.get("type") == "SolarWMInteractive")
        wv = node["widgets_values"]
        # Exactly 15 widgets:
        # [config_path, initial_image, prompt, seed, control_after_generate,
        #  max_latent_frames, attention_backend, linear_quant, svdquant_weights_root,
        #  stage2_vae_quant, camera_motion, translation_speed, rotation_speed,
        #  display_scale, display_fps]
        assert len(wv) == 15, f"{path} widgets_values length mismatch: {len(wv)} != 15"
        assert isinstance(wv[0], str) and wv[0].endswith(".yaml")
        assert isinstance(wv[1], str) and ("eastshade" in wv[1] or wv[1].endswith((".webp", ".png", ".jpg")))
        assert isinstance(wv[2], str)
        assert isinstance(wv[3], int) and wv[3] == 42
        assert wv[4] in ("fixed", "increment", "decrement", "randomize"), f"control_after_generate: {wv[4]}"
        assert isinstance(wv[5], int) and wv[5] == 120, f"max_latent_frames: {wv[5]}"
        assert wv[6] in ("sdpa", "sageattnv1"), f"attention_backend: {wv[6]}"
        assert wv[7] in ("full_precision", "svdquant_k1", "svdquant_k2", "svdquant_k4"), f"linear_quant: {wv[7]}"
        assert isinstance(wv[8], str) and "svdquant" in wv[8]
        assert wv[9] in ("full_precision", "quarot_w8a8"), f"stage2_vae_quant: {wv[9]}"
        assert wv[10] in (
            "straight_about_turn_10s",
            "circle_twice_10s",
            "manual_only",
        ), f"camera_motion: {wv[10]}"
        assert isinstance(wv[11], (int, float)) and wv[11] == 0.1
        assert isinstance(wv[12], (int, float)) and abs(wv[12] - 0.1) < 1e-3
        assert wv[13] in ("1x", "2x", "fit") and wv[13] == "2x"
        assert wv[14] in (4, 8, 12, 16, 24, 30) and wv[14] == 8


def test_session_run_does_not_close_provider_on_finish():
    class DummyProvider:
        closed = False

        def close(self):
            self.closed = True

    class DummyRuntime:
        done = True
        latent_index = 0
        emitted_pixel_frames = 0
        provider = DummyProvider()
        closed = False

        def close(self):
            self.closed = True

    class DummyDecodeContext:
        exited = False

        def __enter__(self):
            return lambda x: x

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.exited = True

    session = SessionState(
        "test-keep-vram",
        DummyRuntime(),
        FreeFlyCamera(),
        DummyDecodeContext(),
        events=Queue(),
    )
    session._run()

    assert session.runtime.closed is True
    assert session.decode_context.exited is True
    assert session.runtime.provider.closed is False


def test_provider_is_cached_and_reused_in_vram(monkeypatch, tmp_path):
    import yaml
    from integrations.comfyui_solarwm.factory import (
        clear_cached_provider,
        get_cached_provider,
        is_provider_cached,
    )

    clear_cached_provider()
    assert is_provider_cached() is False

    dummy_config = tmp_path / "test_config.yaml"
    dummy_config.write_text(
        yaml.dump(
            {
                "model": {"family": "wan22_ti2v_5b", "base_path": "/dummy"},
                "runtime": {},
                "checkpoint": {"path": "/dummy_ckpt"},
                "data": {"width": 16, "height": 16},
            }
        ),
        encoding="utf-8",
    )

    build_calls = []

    class MockProvider:
        def __init__(self, cfg):
            self.cfg = cfg
            self.loaded = False
            self.closed = False

        def _load_role(self, role):
            self.loaded = True

        def close(self):
            self.closed = True

    def mock_build(config):
        build_calls.append(config)
        return MockProvider(config)

    monkeypatch.setattr(
        "solarwm.backends.wan22.runtime.stage2.build_stage2_generation_provider",
        mock_build,
    )

    payload = {
        "config_path": str(dummy_config),
        "attention_backend": "sdpa",
        "linear_quant": "full_precision",
        "stage2_vae_quant": "full_precision",
    }

    try:
        p1, _ = get_cached_provider(payload)
        assert is_provider_cached() is True
        assert len(build_calls) == 1
        assert p1.loaded is True
        assert p1.closed is False

        # Second call with the same config should reuse the exact same provider in VRAM!
        p2, _ = get_cached_provider(payload)
        assert p2 is p1
        assert len(build_calls) == 1
        assert p1.closed is False

        # Calling clear_cached_provider releases it from VRAM
        clear_cached_provider()
        assert is_provider_cached() is False
        assert p1.closed is True
    finally:
        clear_cached_provider()


def test_session_chunk0_aligns_initial_anchor_pose(monkeypatch):
    recorded_poses = []

    def mock_tokens_from_poses(poses, **kwargs):
        recorded_poses.append([p.copy() for p in poses])
        return {}

    monkeypatch.setattr(
        "integrations.comfyui_solarwm.manager.camera_tokens_from_poses",
        mock_tokens_from_poses,
    )

    class MockStepFrame:
        index = 0
        pixels = torch.zeros((1, 3, 3, 16, 16))
        generated_at = 0.0
        diffusion_seconds = 0.01
        decode_seconds = 0.01
        pose = np.eye(4)

    class MockStepRuntime:
        done = False
        latent_index = 0
        emitted_pixel_frames = 0
        chunk = 3
        frame_sequence_length = 2
        device = "cpu"
        dtype = None
        provider = type("P", (), {"config": {"data": {"fps": 16.0}}})()

        def step(self, camera_tokens, pose=None):
            self.latent_index += self.chunk
            self.emitted_pixel_frames += 3
            if self.latent_index >= 6:
                self.done = True
            return MockStepFrame()

        def close(self):
            pass

    class DummyContext:
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    camera = FreeFlyCamera(translation_speed=0.5)
    session = SessionState("test_session", MockStepRuntime(), camera, DummyContext(), events=Queue())
    session._run()

    assert len(recorded_poses) == 2
    # Chunk 0: 3 poses
    chunk0_poses = recorded_poses[0]
    assert len(chunk0_poses) == 3
    # Latent 0 must match the initial anchor pose at t=0
    assert np.allclose(chunk0_poses[0], np.eye(4))
    # Latent 1 at t=0.25 (0.5 m/s * 0.25s = +0.125m)
    assert np.isclose(chunk0_poses[1][2, 3], 0.125, atol=1e-5)
    # Latent 2 at t=0.50 (+0.250m)
    assert np.isclose(chunk0_poses[2][2, 3], 0.250, atol=1e-5)

    # Chunk 1: 3 poses
    chunk1_poses = recorded_poses[1]
    assert len(chunk1_poses) == 3
    # Latent 3 at t=0.75 (+0.375m)
    assert np.isclose(chunk1_poses[0][2, 3], 0.375, atol=1e-5)
    # Latent 4 at t=1.00 (+0.500m)
    assert np.isclose(chunk1_poses[1][2, 3], 0.500, atol=1e-5)
    # Latent 5 at t=1.25 (+0.625m)
    assert np.isclose(chunk1_poses[2][2, 3], 0.625, atol=1e-5)
