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

import numpy as np
import torch

from solarwm.backends.wan22.runtime.interactive import Stage2InteractiveSession
from solarwm.backends.wan22.runtime.interactive_camera import (
    FreeFlyCamera,
    camera_tokens_from_poses,
)
from .camera_path import drawn_path_from_payload, drawn_path_trajectory
from .factory import load_dataset_camera_metadata


ProviderFactory = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _batch_to_rgb_frames(pixels: Any) -> np.ndarray:
    """Vectorized un-normalization, clamp, and uint8 conversion with single D2H transfer.

    Returns a contiguous CPU ndarray of shape [T, H, W, 3] with dtype uint8.
    """
    if isinstance(pixels, torch.Tensor):
        p = pixels.detach()
        if p.ndim == 5:
            p = p[0]
        elif p.ndim == 3:
            p = p.unsqueeze(0)
        if p.ndim != 4:
            raise ValueError(
                f"expected 3D, 4D, or 5D pixels tensor, got shape {tuple(pixels.shape)}"
            )

        if p.shape[1] in (1, 3, 4) and p.shape[-1] not in (1, 3, 4):
            p = p.permute(0, 2, 3, 1)

        if p.shape[-1] == 1:
            p = p.repeat(1, 1, 1, 3)
        elif p.shape[-1] == 4:
            p = p[..., :3]

        if p.shape[-1] != 3:
            raise ValueError(
                f"expected 1, 3, or 4 channels, got shape {tuple(pixels.shape)}"
            )

        if p.dtype == torch.uint8:
            return p.contiguous().cpu().numpy()

        return (
            ((p.float() + 1.0) * 127.5)
            .clamp(0, 255)
            .round()
            .to(torch.uint8)
            .contiguous()
            .cpu()
            .numpy()
        )

    if not isinstance(pixels, np.ndarray):
        pixels = np.asarray(pixels)

    arr = pixels[0] if pixels.ndim == 5 else pixels
    if arr.ndim == 3:
        arr = np.expand_dims(arr, 0)
    if arr.ndim != 4:
        raise ValueError(
            f"expected 3D, 4D, or 5D pixels array, got shape {pixels.shape}"
        )

    if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.shape[-1] != 3:
        raise ValueError(
            f"expected 1, 3, or 4 channels, got shape {pixels.shape}"
        )

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    return np.ascontiguousarray(
        np.clip((arr.astype(np.float32) + 1.0) * 127.5, 0, 255)
        .round()
        .astype(np.uint8)
    )


def _rgb_frame(pixels: Any, frame_index: int = 0) -> np.ndarray:
    if (
        isinstance(pixels, np.ndarray)
        and pixels.ndim == 3
        and pixels.shape[-1] == 3
        and pixels.dtype == np.uint8
    ):
        return pixels
    batch = _batch_to_rgb_frames(pixels)
    if frame_index >= len(batch):
        raise IndexError(
            f"frame_index {frame_index} out of range for batch of size {len(batch)}"
        )
    return batch[frame_index]


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

    def write(self, pixels: Any, frame_index: int = 0) -> None:
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
            self.stream = None


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
    _encode_queue: queue.Queue[tuple[np.ndarray, list[dict[str, Any]]] | None] = field(
        default_factory=lambda: queue.Queue(maxsize=2), init=False, repr=False
    )
    _encode_worker: threading.Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        self._start_encode_worker()
        self.worker = threading.Thread(
            target=self._run,
            name=f"solarwm-interactive-{self.session_id[:8]}",
            daemon=True,
        )
        self.worker.start()

    def _start_encode_worker(self) -> None:
        if self._encode_worker is None or not self._encode_worker.is_alive():
            self._encode_worker = threading.Thread(
                target=self._process_encode_queue,
                name=f"solarwm-encode-{self.session_id[:8]}",
                daemon=True,
            )
            self._encode_worker.start()

    def _process_encode_queue(self) -> None:
        try:
            while True:
                item = self._encode_queue.get()
                if item is None:
                    self._encode_queue.task_done()
                    break
                try:
                    rgb_frames, events_to_publish = item
                    num_frames = int(rgb_frames.shape[0])
                    for offset in range(num_frames):
                        frame = rgb_frames[offset]
                        if self.recording_path is not None:
                            if self._recorder is None:
                                self._recorder = _VideoRecorder(
                                    self.recording_path,
                                    self.recording_fps,
                                )
                            self._recorder.write(frame)
                        self._publish(events_to_publish[offset])
                        self._publish(("jpeg", _jpeg_bytes(frame)))
                except Exception as exc:
                    self._publish(
                        {"type": "error", "error": f"Encoding error: {exc}"}
                    )
                finally:
                    self._encode_queue.task_done()
        finally:
            if self._recorder is not None:
                self._recorder.close()
                self._recorder = None

    def _run(self) -> None:
        last_step = time.perf_counter()
        self._start_encode_worker()
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
                    rgb_frames = (
                        _batch_to_rgb_frames(frame.pixels)
                        if getattr(frame, "pixels", None) is not None
                        else np.zeros((0, 0, 0, 3), dtype=np.uint8)
                    )

                num_frames = int(rgb_frames.shape[0])
                first_pixel = self.runtime.emitted_pixel_frames - num_frames
                events_to_publish: list[dict[str, Any]] = []
                for offset in range(num_frames):
                    events_to_publish.append(
                        {
                            "type": "frame",
                            "session_id": self.session_id,
                            "index": frame.index,
                            "pixel_index": first_pixel + offset,
                            "pixel_count": num_frames,
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
                self._encode_queue.put((rgb_frames, events_to_publish))

            self._encode_queue.put(None)
            if self._encode_worker is not None:
                self._encode_worker.join()

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
            if self._encode_worker is not None and self._encode_worker.is_alive():
                try:
                    self._encode_queue.put_nowait(None)
                except Exception:
                    pass
                self._encode_worker.join(timeout=5.0)
            if self._recorder is not None:
                self._recorder.close()
                self._recorder = None
            self.runtime.close()
            self.decode_context.__exit__(None, None, None)
            try:
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
        drawn_path = drawn_path_from_payload(payload)
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
            dataset_metadata = None
            camera_pose = None
            if drawn_path is not None:
                camera_motion = "dataset_trajectory"
                dataset_metadata = {
                    "trajectory": drawn_path_trajectory(
                        drawn_path,
                        speed=float(
                            payload.get(
                                "translation_speed",
                                inf_cfg.get("default_translation_speed", 0.1),
                            )
                        ),
                    ),
                    "fps": 16.0,
                }
                camera_pose = dataset_metadata["trajectory"][0]
            elif camera_motion == "dataset_trajectory":
                dataset_metadata = load_dataset_camera_metadata(
                    str(payload.get("initial_image", ""))
                )
                if dataset_metadata is None:
                    raise ValueError(
                        "dataset_trajectory requires prompt/camera sidecars for initial_image"
                    )
                camera_pose = dataset_metadata["trajectory"][0]
            camera = FreeFlyCamera(
                pose=camera_pose if camera_pose is not None else np.eye(4),
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
                dataset_trajectory=(
                    None if dataset_metadata is None else dataset_metadata["trajectory"]
                ),
                dataset_fps=(
                    16.0 if dataset_metadata is None else dataset_metadata["fps"]
                ),
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
