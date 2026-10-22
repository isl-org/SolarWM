"""Turn a drawn top-down W/A/S/D path into a camera trajectory for the viewer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

CAMERA_PATH_SCHEMA = "solarwm.comfyui.camera-path.v1"
MAX_PATH_POINTS = 4096
RAMP_SECONDS = 0.5
SMOOTHING_PASSES = 2


def validate_drawn_path(points: Any) -> np.ndarray:
    """Return ``[N, 2]`` float64 ``(right, forward)`` points starting at the origin.

    Coordinates are model-space units relative to the start pose: ``right`` is
    the A/D axis and ``forward`` the W/S axis, i.e. the model's ``+Z`` view
    direction.  Consecutive duplicate points are dropped.
    """

    path = np.asarray(points, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("drawn camera path must be a list of [right, forward] points")
    if len(path) > MAX_PATH_POINTS:
        raise ValueError(f"drawn camera path has more than {MAX_PATH_POINTS} points")
    if not np.isfinite(path).all():
        raise ValueError("drawn camera path must contain finite coordinates")
    if len(path) == 0:
        raise ValueError("drawn camera path is empty")
    keep = np.ones(len(path), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9
    path = path[keep]
    if len(path) < 2:
        raise ValueError("drawn camera path needs at least two distinct points")
    return path - path[0]


def drawn_path_from_payload(payload: Mapping[str, Any]) -> np.ndarray | None:
    """Return the validated drawn path of a session payload, if one is connected."""

    camera_path = payload.get("camera_path")
    if camera_path is None:
        return None
    if not isinstance(camera_path, Mapping):
        raise ValueError("camera_path must be an object with a 'points' list")
    return validate_drawn_path(camera_path.get("points", []))


def _smooth(path: np.ndarray) -> np.ndarray:
    """Round corners with Chaikin corner cutting while keeping both end points."""

    for _ in range(SMOOTHING_PASSES):
        if len(path) < 3:
            break
        start, end = path[:-1], path[1:]
        cut = np.empty((2 * len(start), 2), dtype=np.float64)
        cut[0::2] = 0.75 * start + 0.25 * end
        cut[1::2] = 0.25 * start + 0.75 * end
        path = np.vstack([path[:1], cut[1:-1], path[-1:]])
    return path


def drawn_path_trajectory(
    points: Any,
    *,
    speed: float,
    fps: float = 16.0,
) -> np.ndarray:
    """Sample absolute C2W poses along a drawn path at ``speed`` units per second.

    The camera keeps the start orientation (pure W/A/S/D translation), eases in
    and out over ``RAMP_SECONDS`` and otherwise moves at constant speed.
    """

    if not np.isfinite(speed) or speed <= 0:
        raise ValueError("camera path speed must be positive")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("camera path fps must be positive")
    path = _smooth(validate_drawn_path(points))
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    length = float(arc[-1])
    duration = length / speed + RAMP_SECONDS
    steps = max(2, int(np.ceil(duration * fps)) + 1)
    times = np.arange(steps) / fps
    profile = np.clip(
        np.minimum.reduce(
            [np.ones_like(times), times / RAMP_SECONDS, (duration - times) / RAMP_SECONDS]
        ),
        0.0,
        1.0,
    )
    travelled = np.concatenate([[0.0], np.cumsum(0.5 * (profile[1:] + profile[:-1]))])
    travelled *= length / travelled[-1]
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], steps, axis=0)
    poses[:, 0, 3] = np.interp(travelled, arc, path[:, 0])
    poses[:, 2, 3] = np.interp(travelled, arc, path[:, 1])
    return poses


__all__ = [
    "CAMERA_PATH_SCHEMA",
    "drawn_path_from_payload",
    "drawn_path_trajectory",
    "validate_drawn_path",
]
