"""Threaded session broker for the ComfyUI interactive viewer."""

from __future__ import annotations

import io
import queue
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from solarwm.backends.wan22.runtime.interactive import Stage2InteractiveSession
from solarwm.backends.wan22.runtime.interactive_camera import FreeFlyCamera


ProviderFactory = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _jpeg_bytes(pixels: Any, frame_index: int = 0) -> bytes:
    """Encode one decoded frame in a viewer-friendly format."""

    import numpy as np
    from PIL import Image

    frame = pixels[0, int(frame_index)].detach().float().cpu().numpy()
    frame = np.clip((frame + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return _encode_image(Image.fromarray(frame), "JPEG")


def _encode_image(image: Any, format_name: str) -> bytes:
    output = io.BytesIO()
    image.save(output, format=format_name, quality=85)
    return output.getvalue()


@dataclass
class SessionState:
    session_id: str
    runtime: Stage2InteractiveSession
    camera: FreeFlyCamera
    decode_context: Any
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
                dt = min(max(now - last_step, 0.0), 0.25)
                last_step = now
                pose = self.camera.update(keys, dt)
                camera_tokens = self.camera.model_tokens(
                    frame_sequence_length=self.runtime.frame_sequence_length,
                    device=self.runtime.device,
                    dtype=self.runtime.dtype,
                )
                with self.device_lock:
                    frame = self.runtime.step(camera_tokens, pose=pose)
                first_pixel = self.runtime.emitted_pixel_frames - int(frame.pixels.shape[1])
                for offset in range(int(frame.pixels.shape[1])):
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
                }
            )
        except Exception as exc:
            self._publish(
                {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            self.runtime.close()
            self.decode_context.__exit__(None, None, None)
            close = getattr(self.runtime.provider, "close", None)
            if callable(close):
                close()

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
                self.camera.translation_speed = 1.0
                self.camera.rotation_speed = 0.7

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
        values = dict(self._provider_factory(payload))
        provider = values["provider"]
        decode_context = provider.vae.streaming_decode_session()
        decode_tile = decode_context.__enter__()
        try:
            camera = FreeFlyCamera(
                translation_speed=float(payload.get("translation_speed", 1.0)),
                rotation_speed=float(payload.get("rotation_speed", 0.7)),
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
            session = SessionState(str(uuid.uuid4()), runtime, camera, decode_context)
        except Exception:
            decode_context.__exit__(None, None, None)
            provider.close()
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
