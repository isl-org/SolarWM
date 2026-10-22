from __future__ import annotations

import json
import time
from pathlib import Path
from queue import Queue

import numpy as np
import pytest
import torch

from integrations.comfyui_solarwm.camera_path import (
    drawn_path_from_payload,
    drawn_path_trajectory,
    validate_drawn_path,
)
from integrations.comfyui_solarwm.factory import (
    _interactive_config,
    load_dataset_camera_metadata,
    resolve_image_path,
)
from integrations.comfyui_solarwm.manager import (
    SessionState,
    _VideoRecorder,
    _batch_to_rgb_frames,
    _jpeg_bytes,
    _rgb_frame,
)
from integrations.comfyui_solarwm.node import (
    NODE_CLASS_MAPPINGS,
    SolarWMCameraPath,
    SolarWMInteractive,
)
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
    assert required["attention_backend"][1]["options"] == ["sdpa", "sageattnv1", "sageattnv2"]
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
        "dataset_trajectory",
        "manual_only",
    ]
    assert required["camera_motion"][1]["default"] == "straight_about_turn_10s"
    assert required["translation_speed"][1]["default"] == 0.1
    assert required["rotation_speed"][1]["default"] == 0.1
    assert required["display_scale"][1]["options"] == ["1x", "2x", "fit"]
    assert required["display_scale"][1]["default"] == "2x"
    assert required["display_fps"][1]["options"] == [4, 8, 12, 16, 24, 30]
    assert required["display_fps"][1]["default"] == 8
    assert required["resolution"][1]["options"] == ["480p (480x864)", "360p (320x576)"]
    assert required["resolution"][1]["default"] == "480p (480x864)"

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
        display_scale="2x",
        display_fps=8,
        resolution="360p (320x576)",
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
    assert payload["resolution"] == "360p (320x576)"


def test_interactive_config_applies_and_validates_stage2_options():
    base = {"model": {}, "runtime": {}, "checkpoint": {"path": "/checkpoint"}}
    config = _interactive_config(
        {
            "attention_backend": "sageattnv2",
            "linear_quant": "svdquant_k4",
            "svdquant_weights_root": "/weights",
            "stage2_vae_quant": "quarot_w8a8",
        },
        base,
    )
    assert config["model"] == {
        "attention_backend": "sageattnv2",
        "linear_quant": "svdquant_k4",
        "svdquant_weights_root": "/weights",
    }
    assert config["runtime"] == {"stage2_vae_quant": "quarot_w8a8"}

    config_360p = _interactive_config(
        {"resolution": "360p (320x576)"},
        {"model": {}, "runtime": {}, "data": {}},
    )
    assert config_360p["data"]["height"] == 320
    assert config_360p["data"]["width"] == 576
    assert config_360p["data"]["latent_shape"] == [21, 48, 20, 36]
    assert config_360p["model"]["frame_sequence_length"] == 180

    config_480p = _interactive_config(
        {"resolution": "480p (480x864)"},
        {"model": {}, "runtime": {}, "data": {}},
    )
    assert config_480p["data"]["height"] == 480
    assert config_480p["data"]["width"] == 864
    assert config_480p["data"]["latent_shape"] == [21, 48, 30, 54]
    assert config_480p["model"]["frame_sequence_length"] == 405

    with pytest.raises(ValueError, match="unsupported linear_quant"):
        _interactive_config({"linear_quant": "int8"}, base)


def test_interactive_config_resolves_relative_model_paths():
    base = {
        "model": {"base_path": "models/SolarWM-5B-base"},
        "runtime": {},
        "checkpoint": {"path": "models/SolarWM-5B-sgf-stage2-81f"},
    }
    config = _interactive_config(
        {
            "linear_quant": "svdquant_k1",
            "svdquant_weights_root": "models/SolarWM-5B-sgf-stage2-81f/svdquant_int4",
        },
        base,
    )
    repo_root = Path(__file__).resolve().parents[2]
    assert config["model"]["base_path"] == str(repo_root / "models/SolarWM-5B-base")
    assert config["checkpoint"]["path"] == str(
        repo_root / "models/SolarWM-5B-sgf-stage2-81f"
    )
    assert config["model"]["svdquant_weights_root"] == str(
        repo_root / "models/SolarWM-5B-sgf-stage2-81f/svdquant_int4"
    )


def test_dataset_sidecars_load_for_selected_image(tmp_path):
    image_path = tmp_path / "sample.webp"
    prompt_path = image_path.with_suffix(".prompt.txt")
    camera_path = image_path.with_suffix(".camera.npz")
    trajectory = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    prompt_path.write_text("A test prompt\n", encoding="utf-8")
    np.savez_compressed(
        camera_path,
        c2w=trajectory,
        fps=np.float32(16.0),
        sample_id=np.asarray("dataset/sample"),
    )

    metadata = load_dataset_camera_metadata(image_path)

    assert metadata is not None
    assert metadata["prompt"] == "A test prompt"
    assert metadata["sample_id"] == "dataset/sample"
    assert metadata["fps"] == 16.0
    assert np.array_equal(metadata["trajectory"], trajectory)


def test_repository_image_lookup_strips_comfyui_annotation():
    image_path = next(Path("assets/realcam_vid").glob("*.webp")).resolve()

    assert resolve_image_path(f"{image_path} [input]") == image_path


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
        # Exactly 16 widgets:
        # [config_path, initial_image, prompt, seed, control_after_generate,
        #  max_latent_frames, attention_backend, linear_quant, svdquant_weights_root,
        #  stage2_vae_quant, camera_motion, translation_speed, rotation_speed,
        #  display_scale, display_fps, resolution]
        assert len(wv) == 16, f"{path} widgets_values length mismatch: {len(wv)} != 16"
        assert isinstance(wv[0], str) and wv[0].endswith(".yaml")
        assert isinstance(wv[1], str) and ("eastshade" in wv[1] or wv[1].endswith((".webp", ".png", ".jpg")))
        assert isinstance(wv[2], str)
        assert isinstance(wv[3], int) and wv[3] == 42
        assert wv[4] in ("fixed", "increment", "decrement", "randomize"), f"control_after_generate: {wv[4]}"
        assert isinstance(wv[5], int) and wv[5] == 120, f"max_latent_frames: {wv[5]}"
        assert wv[6] in ("sdpa", "sageattnv1", "sageattnv2"), f"attention_backend: {wv[6]}"
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
        assert wv[15] in ("480p (480x864)", "360p (320x576)") and wv[15] == "480p (480x864)"

    for path in template_paths[:2]:
        workflow = json.loads(path.read_text(encoding="utf-8"))
        node = next(n for n in workflow["nodes"] if n.get("type") == "SolarWMInteractive")
        for value in (node["widgets_values"][0], node["widgets_values"][1], node["widgets_values"][8]):
            assert not value.startswith(("/", "\\")) and not (
                len(value) > 1 and value[1] == ":"
            ), f"template path is not portable: {value}"


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


def test_batch_to_rgb_frames_parity_and_shapes():
    # 1. 5D tensor in [-1.0, 1.0] range
    torch.manual_seed(42)
    pixels = (torch.rand((1, 4, 3, 16, 24), dtype=torch.float32) * 2.0) - 1.0
    batch = _batch_to_rgb_frames(pixels)
    assert batch.shape == (4, 16, 24, 3)
    assert batch.dtype == np.uint8

    # Exact parity against standard formula
    for t in range(4):
        ref = (
            np.clip(
                (pixels[0, t].permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5,
                0,
                255,
            )
            .round()
            .astype(np.uint8)
        )
        assert np.array_equal(batch[t], ref)
        assert np.array_equal(_rgb_frame(pixels, t), ref)

    # 2. 1-channel grayscale expansion
    mono = torch.zeros((1, 2, 1, 8, 8), dtype=torch.float32)
    mono_rgb = _batch_to_rgb_frames(mono)
    assert mono_rgb.shape == (2, 8, 8, 3)
    assert np.all(mono_rgb == 128)

    # 3. 4-channel RGBA truncation
    rgba = torch.ones((1, 2, 4, 8, 8), dtype=torch.float32)
    rgba_rgb = _batch_to_rgb_frames(rgba)
    assert rgba_rgb.shape == (2, 8, 8, 3)
    assert np.all(rgba_rgb == 255)

    # 4. uint8 tensor passthrough
    u8 = torch.randint(0, 256, (2, 8, 8, 3), dtype=torch.uint8)
    u8_out = _batch_to_rgb_frames(u8)
    assert np.array_equal(u8.numpy(), u8_out)

    # 5. Pre-converted 3D uint8 array identity in _rgb_frame
    single_frame = batch[0]
    assert _rgb_frame(single_frame) is single_frame


def test_background_encode_worker_recording_and_streaming(tmp_path):
    import av
    from PIL import Image
    from io import BytesIO

    record_file = tmp_path / "test_async_record.mp4"

    class StepFrame:
        def __init__(self, step_idx):
            self.index = step_idx
            self.pixels = (
                torch.rand((1, 3, 3, 16, 24), dtype=torch.float32) * 2.0 - 1.0
            )
            self.generated_at = time.perf_counter()
            self.diffusion_seconds = 0.05
            self.decode_seconds = 0.02
            self.pose = np.eye(4)
            self.horizon_ratio = 0.5
            self.horizon_step = 1
            self.horizon_limit = 10

    class MockAsyncRuntime:
        done = False
        latent_index = 0
        emitted_pixel_frames = 0
        chunk = 3
        frame_sequence_length = 2
        device = "cpu"
        dtype = None
        provider = type("P", (), {"config": {"data": {"fps": 16.0}}})()

        def step(self, camera_tokens, pose=None):
            step_frame = StepFrame(self.latent_index)
            self.latent_index += self.chunk
            self.emitted_pixel_frames += 3
            if self.latent_index >= 6:
                self.done = True
            return step_frame

        def close(self):
            pass

    class DummyContext:
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    session = SessionState(
        session_id="test_async_session",
        runtime=MockAsyncRuntime(),
        camera=FreeFlyCamera(),
        decode_context=DummyContext(),
        recording_path=record_file,
        recording_fps=16.0,
        events=Queue(maxsize=100),
    )

    session._run()

    # Worker thread must be cleanly stopped and joined
    assert session._encode_worker is not None
    assert not session._encode_worker.is_alive()
    assert session._recorder is None

    # Collect all events emitted through the session
    events = []
    while not session.events.empty():
        events.append(session.events.get_nowait())

    # Check event order and contents:
    # 1 "ready" event
    # 6 pairs of ("frame" dict, ("jpeg", bytes))
    # 1 "stopped" event
    assert events[0] == {"type": "ready", "session_id": "test_async_session"}

    frame_events = [e for e in events if isinstance(e, dict) and e.get("type") == "frame"]
    jpeg_events = [e for e in events if isinstance(e, tuple) and e[0] == "jpeg"]
    stopped_events = [e for e in events if isinstance(e, dict) and e.get("type") == "stopped"]

    assert len(frame_events) == 6
    assert len(jpeg_events) == 6
    assert len(stopped_events) == 1
    assert stopped_events[0]["reason"] == "horizon"
    assert stopped_events[0]["video_path"] == str(record_file)

    # Check JPEG bytes validity
    for _, jbytes in jpeg_events:
        with Image.open(BytesIO(jbytes)) as img:
            assert img.size == (24, 16)
            assert img.format == "JPEG"

    # Check MP4 recording validity
    assert record_file.exists()
    with av.open(str(record_file), mode="r") as container:
        stream = container.streams.video[0]
        assert stream.width == 24
        assert stream.height == 16
        frames = list(container.decode(video=0))
        assert len(frames) == 6


def test_checkpoint_file_prefers_ema_bf16_transaction(tmp_path: Path) -> None:
    import hashlib
    from solarwm.backends.wan22.runtime.inference import _checkpoint_file
    from solarwm.runtime.serialization import canonical_json_bytes

    base_dir = tmp_path / "models"
    raw_dir = base_dir / "SolarWM-5B-sgf-stage2-81f"
    ema_dir = base_dir / "SolarWM-5B-sgf-stage2-81f-ema-bf16"
    raw_dir.mkdir(parents=True)
    ema_dir.mkdir(parents=True)

    contract = {
        "family": "wan22_ti2v_5b",
        "stage": "stage2",
        "causal_mode": "self_gradient_forcing",
        "objective": "flow_matching",
        "objective_variant": "sgf_v1",
        "camera_translation_transform": "linear",
        "parameterization": "full-parameter-live-ema",
        "sp_size": 1,
        "data_generation": "inference-portable:v1",
    }
    for d, is_ema in ((raw_dir, False), (ema_dir, True)):
        (d / "model.pt").write_bytes(b"dummy")
        manifest = {
            "schema": "solarwm.checkpoint.v2",
            "step": 1,
            "contract": contract,
            "required_components": ["model.pt"],
            "files": [{"path": "model.pt", "size": 5}],
            "metadata": {"available_weights": ["ema"] if is_ema else ["live", "ema"]},
        }
        data = canonical_json_bytes(manifest)
        (d / "checkpoint-manifest.json").write_bytes(data)
        (d / "COMPLETE.json").write_bytes(canonical_json_bytes({
            "schema": "solarwm.checkpoint-complete.v2",
            "step": 1,
            "manifest_digest": hashlib.blake2s(data).hexdigest(),
        }))

    config = {
        "checkpoint": {"path": str(raw_dir)},
        "model": {"family": "wan22_ti2v_5b", "camera_translation_transform": "linear"},
        "train": {"stage": "stage2", "objective": "flow_matching"},
    }
    resolved = _checkpoint_file(config)
    assert resolved == ema_dir / "model.pt"


def test_drawn_path_trajectory_translates_along_wasd_plane_at_speed():
    path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5]]
    poses = drawn_path_trajectory(path, speed=0.1, fps=16.0)

    np.testing.assert_allclose(poses[0], np.eye(4))
    np.testing.assert_allclose(poses[-1, [0, 2], 3], [0.5, 0.5], atol=1e-9)
    np.testing.assert_allclose(poses[:, :3, :3], np.broadcast_to(np.eye(3), (len(poses), 3, 3)))
    np.testing.assert_allclose(poses[:, 1, 3], 0.0)
    step = np.linalg.norm(np.diff(poses[:, [0, 2], 3], axis=0), axis=1) * 16.0
    assert step.max() <= 0.1 + 1e-6
    assert np.isclose(np.median(step), 0.1, rtol=0.05)
    assert 9.0 <= len(poses) / 16.0 <= 11.0
    rounded = poses[:, [0, 2], 3]
    assert np.min(np.linalg.norm(rounded - [0.0, 0.5], axis=1)) > 0.01


def test_drawn_path_plays_through_dataset_trajectory_camera():
    poses = drawn_path_trajectory([[0.0, 0.0], [0.0, 1.0]], speed=0.2, fps=16.0)
    camera = FreeFlyCamera(
        default_motion="dataset_trajectory", dataset_trajectory=poses, dataset_fps=16.0
    )
    for _ in range(40):
        camera.update(set(), 0.25)
    np.testing.assert_allclose(camera.pose[:3, 3], [0.0, 0.0, 1.0], atol=1e-9)


def test_camera_path_node_scales_canvas_points_and_feeds_interactive_node():
    assert NODE_CLASS_MAPPINGS["SolarWMCameraPath"] is SolarWMCameraPath
    assert list(SolarWMCameraPath.INPUT_TYPES()["required"]) == ["extent", "path_json"]
    (camera_path,) = SolarWMCameraPath().build_path(
        2.0, json.dumps({"points": [[0, 0], [0, 0.5], [0.25, 0.5]]})
    )
    assert camera_path["points"] == [[0.0, 0.0], [0.0, 1.0], [0.5, 1.0]]

    optional = SolarWMInteractive.INPUT_TYPES()["optional"]
    assert optional["camera_path"][0] == SolarWMCameraPath.RETURN_TYPES[0]
    (config,) = SolarWMInteractive().build_config(
        "config.yaml",
        "anchor.webp",
        "prompt",
        42,
        120,
        "sdpa",
        "full_precision",
        "/weights",
        "full_precision",
        camera_path=camera_path,
    )
    payload = json.loads(config)
    assert payload["camera_motion"] == "dataset_trajectory"
    np.testing.assert_allclose(drawn_path_from_payload(payload), camera_path["points"])


def test_drawn_path_validation():
    assert drawn_path_from_payload({"camera_motion": "straight_about_turn_10s"}) is None
    np.testing.assert_allclose(
        validate_drawn_path([[1.0, 1.0], [1.0, 1.0], [1.0, 2.0]]), [[0.0, 0.0], [0.0, 1.0]]
    )
    for bad in ([], [[0.0, 0.0]], [[0.0, 0.0, 1.0]], [[0.0, np.nan]]):
        with pytest.raises(ValueError):
            validate_drawn_path(bad)
    with pytest.raises(ValueError):
        drawn_path_from_payload({"camera_path": [[0, 0], [0, 1]]})
    with pytest.raises(ValueError):
        drawn_path_trajectory([[0, 0], [0, 1]], speed=0.0)
