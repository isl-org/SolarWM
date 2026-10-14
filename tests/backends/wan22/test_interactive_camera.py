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


def test_horizontal_look_keys_match_viewer_direction():
    left_camera = FreeFlyCamera(
        default_motion="manual_only",
        acceleration=100.0,
        damping=100.0,
    )
    right_camera = FreeFlyCamera(
        default_motion="manual_only",
        acceleration=100.0,
        damping=100.0,
    )
    left_camera.update({"ArrowLeft"}, 0.1)
    right_camera.update({"ArrowRight"}, 0.1)

    left_heading = left_camera.pose[:3, :3] @ np.array([0.0, 0.0, -1.0])
    right_heading = right_camera.pose[:3, :3] @ np.array([0.0, 0.0, -1.0])
    assert left_heading[0] > 0.0
    assert right_heading[0] < 0.0


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


def test_default_motion_forward_turn_right_4s_then_forward():
    camera = FreeFlyCamera(
        translation_speed=0.5,
        default_motion="forward_turn_right_4s",
        turn_delay_seconds=4.0,
        turn_duration_seconds=2.0,
    )
    dt = 0.25  # 1 latent frame duration at 16 fps

    # 0 to 4 seconds: 16 steps
    for _ in range(16):
        camera.update(set(), dt)

    # At 4 seconds: camera moved forward along +Z, no turn
    fwd_4s = camera.pose[:3, :3] @ np.array([0.0, 0.0, -1.0])
    assert camera.pose[2, 3] > 0.8  # moved forward along +Z
    assert np.isclose(camera.pose[0, 3], 0.0, atol=1e-3)
    assert np.isclose(fwd_4s[0], 0.0, atol=1e-3)
    assert np.isclose(fwd_4s[2], -1.0, atol=1e-3)

    # Next 10 steps (2.5 seconds): turn right to 90 degrees
    for _ in range(10):
        camera.update(set(), dt)

    # Turn should be completed (90 deg / pi/2 right turn)
    assert np.isclose(camera._turn_accumulated, np.pi / 2.0, atol=1e-3)
    fwd_turn = camera.pose[:3, :3] @ np.array([0.0, 0.0, -1.0])
    assert np.isclose(fwd_turn[0], 1.0, atol=1e-2)  # now pointing right (+X)
    assert np.isclose(fwd_turn[2], 0.0, atol=1e-2)

    z_after_turn = camera.pose[2, 3]
    x_after_turn = camera.pose[0, 3]

    # Next 8 steps (2.0 seconds): continuing moving slowly forward in new heading (-X)
    for _ in range(8):
        camera.update(set(), dt)

    assert camera.pose[0, 3] < x_after_turn - 0.5  # moving forward along -X
    assert np.isclose(camera.pose[2, 3], z_after_turn, atol=0.05)  # Z stayed steady


def test_default_camera_uses_smooth_4s_turn_defaults():
    camera = FreeFlyCamera()
    assert camera.turn_delay_seconds == 4.0
    assert camera.turn_duration_seconds == 4.0
    assert np.isclose(camera.rotation_speed, 0.1, atol=1e-4)
    assert camera.acceleration == 2.5
    assert camera.damping == 3.0
    assert camera.default_motion == "forward_turn_right_4s"

    dt = 0.25  # 1 latent frame duration at 16 fps

    # 0 to 4s: 16 steps forward
    for _ in range(16):
        camera.update(set(), dt)
    assert camera._turn_accumulated == 0.0
    assert camera.pose[2, 3] > 0.3

    # 4 to 8s: 16 steps smooth turn
    prev_turn = 0.0
    turn_deltas = []
    for _ in range(16):
        camera.update(set(), dt)
        delta = camera._turn_accumulated - prev_turn
        turn_deltas.append(delta)
        prev_turn = camera._turn_accumulated

    # Turn should be completed to exactly 90 degrees (pi/2)
    assert np.isclose(camera._turn_accumulated, np.pi / 2.0, atol=1e-3)
    # Smooth start: first step delta is gentle (< 1 deg)
    assert np.degrees(turn_deltas[0]) < 1.0
    # Smooth end: last step delta is gentle (< 1 deg)
    assert np.degrees(turn_deltas[-1]) < 1.0
    # Heading points right along +X
    fwd = camera.pose[:3, :3] @ np.array([0.0, 0.0, -1.0])
    assert np.isclose(fwd[0], 1.0, atol=1e-2)
    assert np.isclose(fwd[2], 0.0, atol=1e-2)


def test_default_camera_initial_velocity_is_constant_speed():
    camera = FreeFlyCamera()
    assert np.allclose(camera._velocity, [0.0, 0.0, 0.1])

    # First step should move exactly 0.1 * 0.25 = 0.025m without acceleration lag
    camera.update(set(), 0.25)
    assert np.isclose(camera.pose[2, 3], 0.025, atol=1e-5)

    # Calling reset should restore constant cruising velocity
    camera.brake()
    assert np.allclose(camera._velocity, 0.0)
    camera.reset()
    assert np.allclose(camera._velocity, [0.0, 0.0, 0.1])


def test_midpoint_integration_keeps_trajectory_tangent_to_heading():
    camera = FreeFlyCamera(
        translation_speed=0.5,
        turn_delay_seconds=0.0,
        turn_duration_seconds=4.0,
    )
    dt = 0.25
    prev_pos = camera.pose[:3, 3].copy()
    prev_rot = camera.pose[:3, :3].copy()

    for _ in range(16):
        camera.update(set(), dt)
        curr_pos = camera.pose[:3, 3].copy()
        curr_rot = camera.pose[:3, :3].copy()

        disp = curr_pos - prev_pos
        disp_dir = disp / np.linalg.norm(disp)

        # Midpoint orientation forward vector
        trajectory_prev = prev_rot @ np.array([0.0, 0.0, 1.0])
        trajectory_curr = curr_rot @ np.array([0.0, 0.0, 1.0])
        trajectory_mid = trajectory_prev + trajectory_curr
        trajectory_mid /= np.linalg.norm(trajectory_mid)

        # Displacement must be tangent to midpoint heading within 0.01 deg
        cos_angle = np.clip(np.dot(disp_dir, trajectory_mid), -1.0, 1.0)
        assert np.degrees(np.arccos(cos_angle)) < 0.01

        prev_pos = curr_pos
        prev_rot = curr_rot
