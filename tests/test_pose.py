"""
Tests for Hand Pose Estimation Module.
Verifies keypoint schemas, pinch distance, and kinematic hand tracking.
"""

import numpy as np
import pytest

from modules.pose.pose_base import HandPose, HandKeypoint, HAND_KEYPOINT_NAMES
from modules.pose.kinematic_pose import KinematicHandTracker
from modules.pose.mock_pose import MockPoseEstimator, create_synthetic_hand_pose
from scripts.generate_synthetic_video import render_scene


def test_hand_pose_schema_and_landmarks():
    pose = create_synthetic_hand_pose(
        wrist=(100.0, 200.0),
        palm=(100.0, 150.0),
        index_tip=(90.0, 100.0),
        thumb_tip=(70.0, 120.0)
    )
    assert len(pose.landmarks) == 21
    assert pose.landmarks[0].name == "WRIST"
    assert pose.landmarks[4].name == "THUMB_TIP"
    assert pose.landmarks[8].name == "INDEX_FINGER_TIP"
    assert pose.wrist == (100.0, 200.0)
    assert pose.palm_center == (100.0, 150.0)
    assert pose.pinch_distance > 0.0


def test_kinematic_hand_tracker():
    tracker = KinematicHandTracker(min_hand_area_pct=0.005)
    assert tracker.is_operational() is True

    # Render scene containing skin-tone hand
    frame = render_scene(frame_idx=20, total_frames=180)
    poses = tracker.estimate(frame)

    assert len(poses) >= 1
    p = poses[0]
    assert p.handedness in ["Left", "Right"]
    assert len(p.landmarks) == 21
    assert p.bbox[2] > p.bbox[0]
    assert p.bbox[3] > p.bbox[1]


def test_mock_pose_estimator():
    sample = create_synthetic_hand_pose()
    mock = MockPoseEstimator([sample])
    assert mock.is_operational() is True
    res = mock.estimate(np.zeros((50, 50, 3), dtype=np.uint8))
    assert len(res) == 1
    assert res[0].handedness == "Right"

