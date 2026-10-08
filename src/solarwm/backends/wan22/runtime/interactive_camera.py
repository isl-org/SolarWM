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
    "ArrowLeft": np.array([0.0, -1.0, 0.0]),
    "ArrowRight": np.array([0.0, 1.0, 0.0]),
    "ArrowUp": np.array([-1.0, 0.0, 0.0]),
    "ArrowDown": np.array([1.0, 0.0, 0.0]),
    "KeyZ": np.array([0.0, 0.0, 1.0]),
    "KeyC": np.array([0.0, 0.0, -1.0]),
}


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

    The camera uses the conventional optical frame: forward is local ``-Z``,
    right is ``+X`` and up is ``+Y``.  Positions are model-space units and
    angles are radians per second.
    """

    pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    translation_speed: float = 1.0
    rotation_speed: float = 0.7
    acceleration: float = 8.0
    damping: float = 10.0
    boost_multiplier: float = 3.0
    precision_multiplier: float = 0.25
    _velocity: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    _angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)

    def __post_init__(self) -> None:
        self.pose = np.asarray(self.pose, dtype=np.float64).copy()
        _validate_pose(self.pose)
        self._origin = self.pose.copy()

    def reset(self, pose: np.ndarray | None = None) -> None:
        self.pose = np.asarray(
            np.eye(4, dtype=np.float64) if pose is None else pose,
            dtype=np.float64,
        ).copy()
        _validate_pose(self.pose)
        self._origin = self.pose.copy()
        self._velocity.fill(0.0)
        self._angular_velocity.fill(0.0)

    def brake(self) -> None:
        self._velocity.fill(0.0)
        self._angular_velocity.fill(0.0)

    def update(self, pressed: Iterable[str] | Mapping[str, bool], dt: float) -> np.ndarray:
        if dt < 0 or not np.isfinite(dt):
            raise ValueError("camera integration dt must be finite and non-negative")
        keys = (
            {key for key, value in pressed.items() if value}
            if isinstance(pressed, Mapping)
            else set(pressed)
        )
        translation = sum(
            (value for key, value in _TRANSLATION_KEYS.items() if key in keys),
            start=np.zeros(3, dtype=np.float64),
        )
        rotation = sum(
            (value for key, value in _LOOK_KEYS.items() if key in keys),
            start=np.zeros(3, dtype=np.float64),
        )
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
        self._velocity += (target_velocity - self._velocity) * alpha
        self._angular_velocity += (target_angular - self._angular_velocity) * alpha
        if not np.any(translation):
            self._velocity *= np.exp(-self.damping * dt)
        if not np.any(rotation):
            self._angular_velocity *= np.exp(-self.damping * dt)
        self.pose[:3, 3] += self.pose[:3, :3] @ (self._velocity * dt)
        yaw, pitch, roll = self._angular_velocity * dt
        self.pose[:3, :3] = (
            self.pose[:3, :3]
            @ _rotation(np.array([0.0, 1.0, 0.0]), yaw)
            @ _rotation(np.array([1.0, 0.0, 0.0]), pitch)
            @ _rotation(np.array([0.0, 0.0, 1.0]), roll)
        )
        self.pose[:3, :3] = _orthonormalize(self.pose[:3, :3])
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

    def model_tokens(self, *, frame_sequence_length: int, device: Any, dtype: Any = None) -> dict[str, Any]:
        return camera_tokens_from_poses(
            [self.pose],
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
