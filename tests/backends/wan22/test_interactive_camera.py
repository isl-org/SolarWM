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


def test_straight_about_turn_trajectory_walks_back():
    camera = FreeFlyCamera(
        translation_speed=0.5,
        default_motion="straight_about_turn_10s",
        turn_delay_seconds=4.0,
        turn_duration_seconds=2.0,
        turn_angle_rad=np.pi,
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

    # Next 8 steps (2 seconds): turn in place through 180 degrees.
    z_before_turn = camera.pose[2, 3]
    for _ in range(8):
        camera.update(set(), dt)

    assert np.isclose(camera._turn_accumulated, np.pi, atol=1e-3)
    assert np.isclose(camera.pose[2, 3], z_before_turn, atol=0.02)
    fwd_turn = camera.pose[:3, :3] @ np.array([0.0, 0.0, -1.0])
    assert np.isclose(fwd_turn[0], 0.0, atol=1e-2)
    assert np.isclose(fwd_turn[2], 1.0, atol=1e-2)

    # Next 16 steps (4 seconds): walk back along -Z.
    for _ in range(16):
        camera.update(set(), dt)

    assert camera.pose[2, 3] < z_before_turn - 1.5
    assert np.isclose(camera.pose[0, 3], 0.0, atol=0.02)


def test_default_camera_uses_smooth_about_turn_defaults():
    camera = FreeFlyCamera()
    assert camera.turn_delay_seconds == 4.0
    assert camera.turn_duration_seconds == 2.0
    assert np.isclose(camera.rotation_speed, 0.1, atol=1e-4)
    assert camera.acceleration == 2.5
    assert camera.damping == 3.0
    assert camera.default_motion == "straight_about_turn_10s"

    dt = 0.25  # 1 latent frame duration at 16 fps

    # 0 to 4s: 16 steps forward
    for _ in range(16):
        camera.update(set(), dt)
    assert camera._turn_accumulated == 0.0
    assert camera.pose[2, 3] > 0.3

    # 4 to 6s: 8 steps smooth about-turn
    prev_turn = 0.0
    turn_deltas = []
    for _ in range(8):
        camera.update(set(), dt)
        delta = camera._turn_accumulated - prev_turn
        turn_deltas.append(delta)
        prev_turn = camera._turn_accumulated

    assert np.isclose(camera._turn_accumulated, np.pi, atol=1e-3)
    # Smooth start: first step delta is gentle (< 1 deg)
    assert np.degrees(turn_deltas[0]) < 4.0
    # Smooth end: last step delta is gentle (< 1 deg)
    assert np.degrees(turn_deltas[-1]) < 4.0
    # Heading has reversed along Z.
    fwd = camera.pose[:3, :3] @ np.array([0.0, 0.0, -1.0])
    assert np.isclose(fwd[0], 0.0, atol=1e-2)
    assert np.isclose(fwd[2], 1.0, atol=1e-2)


def test_circle_twice_trajectory_closes_after_ten_seconds():
    camera = FreeFlyCamera(
        translation_speed=0.1,
        default_motion="circle_twice_10s",
        turn_delay_seconds=0.0,
        turn_duration_seconds=10.0,
        turn_angle_rad=4.0 * np.pi,
    )
    for _ in range(40):
        camera.update(set(), 0.25)

    assert np.isclose(camera._turn_accumulated, 4.0 * np.pi, atol=1e-3)
    assert np.linalg.norm(camera.pose[:3, 3]) < 0.02
    assert np.allclose(camera.pose[:3, :3], np.eye(3), atol=1e-3)
    end_pose = camera.pose.copy()
    for _ in range(4):
        camera.update(set(), 0.25)
    assert np.allclose(camera.pose, end_pose, atol=1e-6)


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


def test_dataset_trajectory_motion_follows_saved_poses():
    trajectory = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    trajectory[1, 2, 3] = 1.0
    trajectory[2, 2, 3] = 2.0
    camera = FreeFlyCamera(
        default_motion="dataset_trajectory",
        dataset_trajectory=trajectory,
        dataset_fps=4.0,
    )

    assert np.allclose(camera.pose, trajectory[0])
    camera.update(set(), 0.25)
    assert np.allclose(camera.pose, trajectory[1])
    camera.update(set(), 0.25)
    assert np.allclose(camera.pose, trajectory[2])


def test_dataset_trajectory_manual_override_resumes_without_jump():
    trajectory = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    trajectory[1, 2, 3] = 1.0
    trajectory[2, 2, 3] = 2.0
    camera = FreeFlyCamera(
        default_motion="dataset_trajectory",
        dataset_trajectory=trajectory,
        dataset_fps=4.0,
        translation_speed=1.0,
        acceleration=100.0,
        damping=100.0,
    )

    camera.update(set(), 0.25)
    camera.update({"KeyD"}, 0.25)
    manual_pose = camera.pose.copy()
    resumed_pose = camera.update(set(), 0.0)

    assert manual_pose[0, 3] > 0.0
    assert np.allclose(resumed_pose, manual_pose, atol=1e-6)
    assert np.isclose(resumed_pose[2, 3], 1.0, atol=1e-6)


def test_midpoint_integration_keeps_trajectory_tangent_to_heading():
    camera = FreeFlyCamera(
        translation_speed=0.5,
        default_motion="circle_twice_10s",
        turn_delay_seconds=0.0,
        turn_duration_seconds=4.0,
        turn_angle_rad=2.0 * np.pi,
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
