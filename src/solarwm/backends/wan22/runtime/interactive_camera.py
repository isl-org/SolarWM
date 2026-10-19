"""Keyboard free-fly camera state and Wan camera-token conversion."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data import FX_NORM, FY_NORM


_TRANSLATION_KEYS = {
    "KeyW": np.array([0.0, 0.0, -1.0]),
    "KeyS": np.array([0.0, 0.0, 1.0]),
    "KeyA": np.array([-1.0, 0.0, 0.0]),
    "KeyD": np.array([1.0, 0.0, 0.0]),
    "KeyQ": np.array([0.0, -1.0, 0.0]),
    "KeyE": np.array([0.0, 1.0, 0.0]),
}
_LOOK_KEYS = {
    "ArrowLeft": np.array([-1.0, 0.0, 0.0]),
    "ArrowRight": np.array([1.0, 0.0, 0.0]),
    "ArrowUp": np.array([0.0, 1.0, 0.0]),
    "ArrowDown": np.array([0.0, -1.0, 0.0]),
    "KeyZ": np.array([0.0, 0.0, 1.0]),
    "KeyC": np.array([0.0, 0.0, -1.0]),
}
_STRAIGHT_ABOUT_TURN = "straight_about_turn_10s"
_CIRCLE_TWICE = "circle_twice_10s"
_DATASET_TRAJECTORY = "dataset_trajectory"
_AUTONOMOUS_MOTIONS = {_STRAIGHT_ABOUT_TURN, _CIRCLE_TWICE}


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0.0 or angle == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ],
        dtype=np.float64,
    )


def _validate_pose(pose: np.ndarray) -> None:
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("camera pose must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("camera pose must be homogeneous")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("camera rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("camera rotation must have determinant +1")


@dataclass
class FreeFlyCamera:
    """Integrate keyboard intent in camera-local coordinates.

    Manual controls use the conventional optical frame: forward is local
    ``-Z``, right is ``+X`` and up is ``+Y``.  The autonomous review trajectory
    intentionally follows ``+Z`` to match the model's desired visual direction.
    Positions are model-space units and angles are radians per second.
    """

    pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    translation_speed: float = 0.1
    rotation_speed: float = 0.1
    acceleration: float = 2.5
    damping: float = 3.0
    boost_multiplier: float = 3.0
    precision_multiplier: float = 0.25
    default_motion: str = _STRAIGHT_ABOUT_TURN
    turn_delay_seconds: float = 4.0
    turn_duration_seconds: float = 2.0
    turn_angle_rad: float = np.pi
    dataset_trajectory: np.ndarray | None = field(default=None, repr=False)
    dataset_fps: float = 16.0
    _velocity: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    _angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    _elapsed_time: float = field(default=0.0, init=False)
    _turn_accumulated: float = field(default=0.0, init=False)
    _dataset_offset: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64),
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.pose = np.asarray(self.pose, dtype=np.float64).copy()
        _validate_pose(self.pose)
        if self.dataset_trajectory is not None:
            trajectory = np.asarray(self.dataset_trajectory, dtype=np.float64)
            if trajectory.ndim != 3 or trajectory.shape[1:] != (4, 4):
                raise ValueError("dataset camera trajectory must have shape [N, 4, 4]")
            for pose in trajectory:
                _validate_pose(pose)
            if not np.isfinite(self.dataset_fps) or self.dataset_fps <= 0:
                raise ValueError("dataset camera trajectory fps must be positive")
            self.dataset_trajectory = trajectory.copy()
        if self.default_motion == _CIRCLE_TWICE:
            self.turn_delay_seconds = 0.0
            self.turn_duration_seconds = 10.0
            self.turn_angle_rad = 4.0 * np.pi
        self._origin = self.pose.copy()
        self._dataset_offset = np.eye(4, dtype=np.float64)
        if self.default_motion in _AUTONOMOUS_MOTIONS:
            self._velocity = np.array([0.0, 0.0, self.translation_speed], dtype=np.float64)

    def reset(self, pose: np.ndarray | None = None) -> None:
        reset_pose = (
            self.dataset_trajectory[0]
            if pose is None and self.dataset_trajectory is not None
            else np.eye(4, dtype=np.float64) if pose is None else pose
        )
        self.pose = np.asarray(
            reset_pose,
            dtype=np.float64,
        ).copy()
        _validate_pose(self.pose)
        self._origin = self.pose.copy()
        self._dataset_offset = np.eye(4, dtype=np.float64)
        self._angular_velocity.fill(0.0)
        self._elapsed_time = 0.0
        self._turn_accumulated = 0.0
        if self.default_motion == "none":
            self.default_motion = _STRAIGHT_ABOUT_TURN
            self.turn_delay_seconds = 4.0
            self.turn_duration_seconds = 2.0
            self.turn_angle_rad = np.pi
        if self.default_motion in _AUTONOMOUS_MOTIONS:
            self._velocity = np.array([0.0, 0.0, self.translation_speed], dtype=np.float64)

    def _dataset_pose_at(self, seconds: float) -> np.ndarray:
        assert self.dataset_trajectory is not None
        position = max(0.0, float(seconds)) * float(self.dataset_fps)
        if position >= len(self.dataset_trajectory) - 1:
            return self.dataset_trajectory[-1].copy()
        lower = int(np.floor(position))
        alpha = position - lower
        upper = lower + 1
        translation = (
            (1.0 - alpha) * self.dataset_trajectory[lower, :3, 3]
            + alpha * self.dataset_trajectory[upper, :3, 3]
        )
        rotation = _orthonormalize(
            (1.0 - alpha) * self.dataset_trajectory[lower, :3, :3]
            + alpha * self.dataset_trajectory[upper, :3, :3]
        )
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = rotation
        result[:3, 3] = translation
        return result

    def brake(self) -> None:
        self._velocity.fill(0.0)
        self._angular_velocity.fill(0.0)
        self.default_motion = "none"

    def update(self, pressed: Iterable[str] | Mapping[str, bool], dt: float) -> np.ndarray:
        if dt < 0 or not np.isfinite(dt):
            raise ValueError("camera integration dt must be finite and non-negative")
        keys = (
            {key for key, value in pressed.items() if value}
            if isinstance(pressed, Mapping)
            else set(pressed)
        )
        has_manual_translation = any(key in keys for key in _TRANSLATION_KEYS)
        has_manual_rotation = any(key in keys for key in _LOOK_KEYS)
        if (
            self.default_motion == _DATASET_TRAJECTORY
            and self.dataset_trajectory is not None
            and not has_manual_translation
            and not has_manual_rotation
        ):
            self._elapsed_time += dt
            self.pose = self._dataset_offset @ self._dataset_pose_at(self._elapsed_time)
            self._velocity.fill(0.0)
            self._angular_velocity.fill(0.0)
            return self.pose.copy()

        if has_manual_translation:
            translation = sum(
                (value for key, value in _TRANSLATION_KEYS.items() if key in keys),
                start=np.zeros(3, dtype=np.float64),
            )
        elif (
            self.default_motion in _AUTONOMOUS_MOTIONS
            and self._elapsed_time < 10.0
        ):
            # Autonomous motion follows the model's +Z trajectory direction.
            translation = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            translation = np.zeros(3, dtype=np.float64)

        turn_d_yaw = 0.0
        if has_manual_rotation:
            rotation = sum(
                (value for key, value in _LOOK_KEYS.items() if key in keys),
                start=np.zeros(3, dtype=np.float64),
            )
        elif (
            self.default_motion in _AUTONOMOUS_MOTIONS
            and self._elapsed_time < 10.0
            and self._elapsed_time + dt > self.turn_delay_seconds
            and self._turn_accumulated < self.turn_angle_rad
        ):
            dur = max(self.turn_duration_seconds, 1e-6)
            if self.default_motion == _CIRCLE_TWICE:
                turn_d_yaw = min(
                    float(self.turn_angle_rad - self._turn_accumulated),
                    float(self.turn_angle_rad * dt / dur),
                )
            else:
                # Smooth C2-continuous about-turn using smootherstep.
                tau0 = np.clip(
                    (self._elapsed_time - self.turn_delay_seconds) / dur, 0.0, 1.0
                )
                tau1 = np.clip(
                    (self._elapsed_time + dt - self.turn_delay_seconds) / dur,
                    0.0,
                    1.0,
                )
                s0 = 6.0 * (tau0**5) - 15.0 * (tau0**4) + 10.0 * (tau0**3)
                s1 = 6.0 * (tau1**5) - 15.0 * (tau1**4) + 10.0 * (tau1**3)
                turn_d_yaw = float(self.turn_angle_rad * (s1 - s0))
            self._turn_accumulated = min(float(self.turn_angle_rad), self._turn_accumulated + turn_d_yaw)
            rotation = np.zeros(3, dtype=np.float64)
            if (
                self._turn_accumulated >= self.turn_angle_rad
                or (
                    self.default_motion != _CIRCLE_TWICE
                    and tau1 >= 1.0
                )
            ):
                self._angular_velocity[0] = 0.0
            elif dt > 0:
                self._angular_velocity[0] = -turn_d_yaw / dt
        else:
            rotation = np.zeros(3, dtype=np.float64)

        if np.linalg.norm(translation) > 1.0:
            translation /= np.linalg.norm(translation)
        if np.linalg.norm(rotation) > 1.0:
            rotation /= np.linalg.norm(rotation)
        multiplier = self.boost_multiplier if "ShiftLeft" in keys or "ShiftRight" in keys else 1.0
        if "KeyV" in keys:
            multiplier *= self.precision_multiplier
        target_velocity = translation * self.translation_speed * multiplier
        target_angular = rotation * self.rotation_speed * multiplier
        alpha = 1.0 - np.exp(-self.acceleration * dt)
        turning_in_place = (
            self.default_motion == _STRAIGHT_ABOUT_TURN
            and self.turn_delay_seconds < self._elapsed_time + dt
            and self._elapsed_time < self.turn_delay_seconds + self.turn_duration_seconds
        )
        if turning_in_place and not has_manual_translation:
            self._velocity.fill(0.0)
        else:
            self._velocity += (target_velocity - self._velocity) * alpha
        if (
            self.default_motion in _AUTONOMOUS_MOTIONS
            and self._elapsed_time >= 10.0
            and not has_manual_translation
        ):
            self._velocity.fill(0.0)
        if has_manual_rotation:
            self._angular_velocity += (target_angular - self._angular_velocity) * alpha
        else:
            if turn_d_yaw == 0.0:
                self._angular_velocity *= np.exp(-self.damping * dt)

        if not np.any(translation):
            self._velocity *= np.exp(-self.damping * dt)

        if turn_d_yaw > 0.0:
            yaw = -turn_d_yaw
            pitch = float(self._angular_velocity[1] * dt)
            roll = float(self._angular_velocity[2] * dt)
        else:
            yaw, pitch, roll = self._angular_velocity * dt

        half_rot = (
            _rotation(np.array([0.0, 1.0, 0.0]), 0.5 * yaw)
            @ _rotation(np.array([1.0, 0.0, 0.0]), 0.5 * pitch)
            @ _rotation(np.array([0.0, 0.0, 1.0]), 0.5 * roll)
        )
        mid_rot = self.pose[:3, :3] @ half_rot
        self.pose[:3, 3] += mid_rot @ (self._velocity * dt)
        self.pose[:3, :3] = _orthonormalize(mid_rot @ half_rot)
        self._elapsed_time += dt
        if (
            self.default_motion == _DATASET_TRAJECTORY
            and self.dataset_trajectory is not None
            and (has_manual_translation or has_manual_rotation)
        ):
            self._dataset_offset = self.pose @ np.linalg.inv(
                self._dataset_pose_at(self._elapsed_time)
            )
        return self.pose.copy()

    def level_roll(self, amount: float = 0.2) -> np.ndarray:
        right = self.pose[:3, 0]
        up = self.pose[:3, 1]
        world_up = np.array([0.0, 1.0, 0.0])
        correction = np.dot(np.cross(up, world_up), right)
        self.pose[:3, :3] = self.pose[:3, :3] @ _rotation(
            np.array([0.0, 0.0, 1.0]), -float(correction) * float(amount)
        )
        self.pose[:3, :3] = _orthonormalize(self.pose[:3, :3])
        return self.pose.copy()

    def model_tokens(
        self,
        *,
        frame_sequence_length: int,
        device: Any,
        dtype: Any = None,
        chunk: int = 1,
    ) -> dict[str, Any]:
        return camera_tokens_from_poses(
            [self.pose] * int(chunk),
            origin=self._origin,
            frame_sequence_length=frame_sequence_length,
            device=device,
            dtype=dtype,
        )


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(rotation)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def camera_tokens_from_poses(
    poses: Iterable[np.ndarray],
    *,
    origin: np.ndarray,
    frame_sequence_length: int,
    device: Any,
    dtype: Any = None,
) -> dict[str, Any]:
    """Convert absolute C2W poses into repeated Wan relative-W2C tokens."""

    import torch

    origin = np.asarray(origin, dtype=np.float64)
    _validate_pose(origin)
    matrices = [np.asarray(pose, dtype=np.float64) for pose in poses]
    if not matrices:
        raise ValueError("at least one camera pose is required")
    relative_w2c = []
    origin_inverse = np.linalg.inv(origin)
    for pose in matrices:
        _validate_pose(pose)
        relative_c2w = origin_inverse @ pose
        relative_w2c.append(np.linalg.inv(relative_c2w).astype(np.float32))
    viewmats = np.repeat(
        np.asarray(relative_w2c, dtype=np.float32),
        int(frame_sequence_length),
        axis=0,
    )
    intrinsics = np.zeros((len(viewmats), 3, 3), dtype=np.float32)
    intrinsics[:, 0, 0] = FX_NORM
    intrinsics[:, 1, 1] = FY_NORM
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5
    intrinsics[:, 2, 2] = 1.0
    target_dtype = dtype or torch.float32
    return {
        "viewmats": torch.as_tensor(viewmats, device=device, dtype=target_dtype).unsqueeze(0),
        "K": torch.as_tensor(intrinsics, device=device, dtype=target_dtype).unsqueeze(0),
    }


__all__ = ["FreeFlyCamera", "camera_tokens_from_poses"]
