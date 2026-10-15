"""Threaded session broker for the ComfyUI interactive viewer."""

from __future__ import annotations

import io
import queue
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from solarwm.backends.wan22.runtime.interactive import Stage2InteractiveSession
from solarwm.backends.wan22.runtime.interactive_camera import (
    FreeFlyCamera,
    camera_tokens_from_poses,
)


ProviderFactory = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _rgb_frame(pixels: Any, frame_index: int = 0) -> Any:
    import numpy as np

    frame = pixels[0, int(frame_index)].detach().float().cpu().numpy()
    if frame.ndim != 3:
        raise ValueError(f"decoded frame must have three dimensions, got {frame.shape}")
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        raise ValueError(f"decoded frame must have 1, 3, or 4 channels, got {frame.shape}")
    return np.clip((frame + 1.0) * 127.5, 0, 255).round().astype(np.uint8)


def _jpeg_bytes(pixels: Any, frame_index: int = 0) -> bytes:
    """Encode one decoded frame in a viewer-friendly format."""

    from PIL import Image

    frame = _rgb_frame(pixels, frame_index)
    return _encode_image(Image.fromarray(frame), "JPEG")


def _encode_image(image: Any, format_name: str) -> bytes:
    output = io.BytesIO()
    image.save(output, format=format_name, quality=85)
    return output.getvalue()


class _VideoRecorder:
    """Incrementally write decoded interactive frames to an H.264 MP4."""

    def __init__(self, path: Path, fps: float) -> None:
        self.path = path
        self.fps = max(1, int(round(fps)))
        self.container: Any | None = None
        self.stream: Any | None = None

    def write(self, pixels: Any, frame_index: int) -> None:
        import av

        frame = _rgb_frame(pixels, frame_index)
        if self.container is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.container = av.open(str(self.path), mode="w")
            self.stream = self.container.add_stream("libx264", rate=self.fps)
            self.stream.width = int(frame.shape[1])
            self.stream.height = int(frame.shape[0])
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": "18", "preset": "medium"}
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self.container is None:
            return
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
            self.container = None


@dataclass
class SessionState:
    session_id: str
    runtime: Stage2InteractiveSession
    camera: FreeFlyCamera
    decode_context: Any
    recording_path: Path | None = None
    recording_fps: float = 16.0
    device_lock: threading.Lock = field(default_factory=threading.Lock)
    worker: threading.Thread | None = None
    controls: set[str] = field(default_factory=set)
    control_sequence: int = -1
    control_received_at: float | None = None
    control_sent_at: float | None = None
    running: bool = True
    single_step: bool = False
    stop_requested: bool = False
    events: queue.Queue[dict[str, Any] | tuple[str, bytes]] = field(
        default_factory=lambda: queue.Queue(maxsize=8)
    )
    lock: threading.Lock = field(default_factory=threading.Lock)
    _recorder: _VideoRecorder | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        self.worker = threading.Thread(
            target=self._run,
            name=f"solarwm-interactive-{self.session_id[:8]}",
            daemon=True,
        )
        self.worker.start()

    def _run(self) -> None:
        last_step = time.perf_counter()
        try:
            self._publish({"type": "ready", "session_id": self.session_id})
            while not self.stop_requested and not self.runtime.done:
                with self.lock:
                    should_run = self.running or self.single_step
                    self.single_step = False
                    keys = frozenset(self.controls)
                if not should_run:
                    time.sleep(0.02)
                    continue
                now = time.perf_counter()
                fps = 16.0
                if hasattr(self.runtime, "provider") and hasattr(self.runtime.provider, "config"):
                    fps = float(self.runtime.provider.config.get("data", {}).get("fps", 16.0))
                dt_sub = 4.0 / max(fps, 1.0)
                chunk = int(getattr(self.runtime, "chunk", 1))
                poses = []
                if self.runtime.latent_index == 0:
                    poses.append(self.camera.pose.copy())
                    for _ in range(chunk - 1):
                        pose = self.camera.update(keys, dt_sub)
                        poses.append(pose)
                else:
                    for _ in range(chunk):
                        pose = self.camera.update(keys, dt_sub)
                        poses.append(pose)
                camera_tokens = camera_tokens_from_poses(
                    poses,
                    origin=self.camera._origin,
                    frame_sequence_length=self.runtime.frame_sequence_length,
                    device=self.runtime.device,
                    dtype=self.runtime.dtype,
                )
                with self.device_lock:
                    frame = self.runtime.step(camera_tokens, pose=poses[-1])
                first_pixel = self.runtime.emitted_pixel_frames - int(frame.pixels.shape[1])
                for offset in range(int(frame.pixels.shape[1])):
                    if self.recording_path is not None:
                        if self._recorder is None:
                            self._recorder = _VideoRecorder(
                                self.recording_path,
                                self.recording_fps,
                            )
                        self._recorder.write(frame.pixels, offset)
                    self._publish(
                        {
                            "type": "frame",
                            "session_id": self.session_id,
                            "index": frame.index,
                            "pixel_index": first_pixel + offset,
                            "pixel_count": int(frame.pixels.shape[1]),
                            "generated_at": frame.generated_at,
                            "diffusion_seconds": frame.diffusion_seconds,
                            "decode_seconds": frame.decode_seconds,
                            "latency_seconds": frame.diffusion_seconds + frame.decode_seconds,
                            "latent_index": self.runtime.latent_index,
                            "pixel_frames": self.runtime.emitted_pixel_frames,
                            "control_sequence": self.control_sequence,
                            "control_sent_at": self.control_sent_at,
                            "control_received_at": self.control_received_at,
                            "pose": frame.pose.tolist() if frame.pose is not None else None,
                        }
                    )
                    self._publish(("jpeg", _jpeg_bytes(frame.pixels, offset)))
            self._publish(
                {
                    "type": "stopped",
                    "reason": "horizon" if self.runtime.done else "cancelled",
                    "latent_index": self.runtime.latent_index,
                    "video_path": (
                        str(self.recording_path)
                        if self.recording_path is not None
                        else None
                    ),
                }
            )
        except Exception as exc:
            self._publish(
                {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            if self._recorder is not None:
                self._recorder.close()
            self.runtime.close()
            self.decode_context.__exit__(None, None, None)
            try:
                import torch

                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _publish(self, event: dict[str, Any] | tuple[str, bytes]) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(event)

    def control(self, payload: Mapping[str, Any]) -> None:
        sequence = int(payload.get("sequence", -1))
        with self.lock:
            if sequence <= self.control_sequence:
                return
            self.control_sequence = sequence
            sent_at = payload.get("sent_at_epoch")
            self.control_sent_at = float(sent_at) if sent_at is not None else None
            self.control_received_at = time.time()
            self.controls = {
                str(key)
                for key in payload.get("pressed", [])
                if isinstance(key, str)
            }
            action = str(payload.get("action", ""))
            if action == "stop":
                self.stop_requested = True
            elif action == "pause":
                self.running = False
            elif action == "resume":
                self.running = True
            elif action == "step":
                self.single_step = True
            elif action == "brake":
                self.camera.brake()
            elif action == "drop":
                self.controls.clear()
                self.camera.brake()
            elif action == "level_roll":
                self.camera.level_roll()
            elif action == "speed_down":
                self.camera.translation_speed = max(
                    0.01, self.camera.translation_speed / 1.2
                )
            elif action == "speed_up":
                self.camera.translation_speed = min(
                    20.0, self.camera.translation_speed * 1.2
                )
            elif action == "look_down":
                self.camera.rotation_speed = max(0.01, self.camera.rotation_speed / 1.2)
            elif action == "look_up":
                self.camera.rotation_speed = min(6.28, self.camera.rotation_speed * 1.2)
            elif action == "defaults":
                self.camera.translation_speed = 0.1
                self.camera.rotation_speed = 0.1
                self.camera.default_motion = "straight_about_turn_10s"
                self.camera.turn_delay_seconds = 4.0
                self.camera.turn_duration_seconds = 2.0
                self.camera.turn_angle_rad = 3.141592653589793
                self.camera.acceleration = 2.5
                self.camera.damping = 3.0
                self.camera.reset()

    def stop(self) -> None:
        with self.lock:
            self.stop_requested = True


class SolarWMSessionManager:
    """Own sessions outside ComfyUI's single prompt worker."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._device_lock = threading.Lock()
        self._provider_factory: ProviderFactory | None = None

    def set_provider_factory(self, factory: ProviderFactory) -> None:
        self._provider_factory = factory

    def create(self, payload: Mapping[str, Any]) -> SessionState:
        if self._provider_factory is None:
            raise RuntimeError(
                "SolarWM provider factory is not configured; call "
                "configure_provider_factory() from the ComfyUI deployment"
            )
        # Ensure any existing active sessions are terminated before starting a new one
        with self._lock:
            stale_ids = list(self._sessions.keys())
        for sid in stale_ids:
            self.delete(sid)

        values = dict(self._provider_factory(payload))
        provider = values["provider"]
        decode_context = provider.vae.streaming_decode_session()
        decode_tile = decode_context.__enter__()
        try:
            inf_cfg = getattr(provider, "config", {}).get("inference", {})
            camera_motion = str(
                payload.get(
                    "camera_motion",
                    inf_cfg.get("default_camera_motion", "straight_about_turn_10s"),
                )
            )
            is_circle = camera_motion == "circle_twice_10s"
            camera = FreeFlyCamera(
                translation_speed=float(
                    payload.get(
                        "translation_speed",
                        inf_cfg.get("default_translation_speed", 0.1),
                    )
                ),
                rotation_speed=float(
                    payload.get(
                        "rotation_speed",
                        inf_cfg.get("default_rotation_speed", 0.1),
                    )
                ),
                default_motion=camera_motion,
                turn_delay_seconds=0.0 if is_circle else 4.0,
                turn_duration_seconds=10.0 if is_circle else 2.0,
                turn_angle_rad=4.0 * 3.141592653589793 if is_circle else 3.141592653589793,
                acceleration=float(inf_cfg.get("default_acceleration", 2.5)),
                damping=float(inf_cfg.get("default_damping", 3.0)),
            )
            runtime = Stage2InteractiveSession.from_provider(
                provider,
                first_latent=values["first_latent"],
                condition=values["condition"],
                seed=int(payload.get("seed", 42)),
                max_latent_frames=int(payload.get("max_latent_frames", 120)),
                decode_tile=decode_tile,
                first_pose=camera.pose,
            )
            session_id = str(uuid.uuid4())
            configured_output = str(
                getattr(provider, "config", {})
                .get("runtime", {})
                .get("output_dir", "")
            )
            if not configured_output or configured_output.startswith("/path/to/"):
                output_root = (
                    Path(__file__).resolve().parents[2]
                    / "outputs"
                    / "solarwm_interactive"
                )
            else:
                output_root = Path(configured_output).expanduser()
            recording_path = output_root / f"{session_id}.mp4"
            recording_fps = float(
                getattr(provider, "config", {}).get("data", {}).get("fps", 16.0)
            )
            session = SessionState(
                session_id,
                runtime,
                camera,
                decode_context,
                recording_path=recording_path,
                recording_fps=recording_fps,
            )
        except Exception:
            decode_context.__exit__(None, None, None)
            raise
        with self._lock:
            session.device_lock = self._device_lock
            self._sessions[session.session_id] = session
        session.start()
        return session

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            session = self._sessions.get(str(session_id))
        if session is None:
            raise KeyError(session_id)
        return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(str(session_id), None)
        if session is not None:
            session.stop()
            if session.worker is not None:
                session.worker.join(timeout=10.0)


manager = SolarWMSessionManager()


def configure_provider_factory(factory: ProviderFactory) -> None:
    manager.set_provider_factory(factory)


__all__ = [
    "SessionState",
    "SolarWMSessionManager",
    "configure_provider_factory",
    "manager",
]
