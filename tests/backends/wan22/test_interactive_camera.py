from __future__ import annotations

import numpy as np
import torch

from solarwm.backends.wan22.runtime.interactive_camera import (
    FreeFlyCamera,
    camera_tokens_from_poses,
)


def test_free_fly_diagonal_motion_is_normalized_and_pose_stays_se3():
    camera = FreeFlyCamera(translation_speed=2.0, acceleration=100.0, damping=100.0)
    camera.update({"KeyW", "KeyD"}, 0.01)
    displacement = np.linalg.norm(camera.pose[:3, 3])
    assert displacement <= 2.0 * 0.01 + 1e-6
    rotation = camera.pose[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)


def test_camera_tokens_rebase_to_initial_absolute_pose():
    origin = np.eye(4)
    moved = np.eye(4)
    moved[0, 3] = 2.0
    tokens = camera_tokens_from_poses(
        [origin, moved],
        origin=origin,
        frame_sequence_length=3,
        device=torch.device("cpu"),
    )
    assert tuple(tokens["viewmats"].shape) == (1, 6, 4, 4)
    assert torch.allclose(tokens["viewmats"][0, :3], torch.eye(4))
    assert torch.allclose(tokens["viewmats"][0, 3, :3, 3], torch.tensor([-2.0, 0.0, 0.0]))
    assert torch.allclose(tokens["K"][0, 0], tokens["K"][0, 5])
