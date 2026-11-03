"""
Mock Hand Pose Estimator for Deterministic Unit Tests.
"""

from typing import Callable, Optional
import numpy as np

from modules.pose.pose_base import BasePoseEstimator, HandPose, HandKeypoint, HAND_KEYPOINT_NAMES


class MockPoseEstimator(BasePoseEstimator):
    """Provides deterministic hand pose predictions for test harnesses."""

    def __init__(self, predefined_poses: Optional[list[HandPose]] = None):
        self._poses = predefined_poses or []
        self._provider_fn: Optional[Callable[[np.ndarray], list[HandPose]]] = None

    def set_poses(self, poses: list[HandPose]) -> None:
        self._poses = list(poses)

    def set_provider(self, provider_fn: Callable[[np.ndarray], list[HandPose]]) -> None:
        self._provider_fn = provider_fn

    def is_operational(self) -> bool:
        return True

    def estimate(self, frame: np.ndarray) -> list[HandPose]:
        if self._provider_fn is not None:
            return self._provider_fn(frame)
        return list(self._poses)


def create_synthetic_hand_pose(
    wrist: tuple[float, float] = (320.0, 400.0),
    palm: tuple[float, float] = (320.0, 320.0),
    index_tip: tuple[float, float] = (300.0, 220.0),
    thumb_tip: tuple[float, float] = (250.0, 280.0),
    handedness: str = "Right"
) -> HandPose:
    """Helper to generate a structurally valid HandPose object for tests."""
    keypoints = []
    # 0: wrist
    keypoints.append(HandKeypoint(id=0, name="WRIST", x=wrist[0], y=wrist[1]))
    # fill other 20 landmarks with linearly interpolated positions
    for i in range(1, 21):
        name = HAND_KEYPOINT_NAMES[i]
        keypoints.append(HandKeypoint(id=i, name=name, x=palm[0] + (i * 2.0), y=palm[1] - (i * 5.0)))

    xs = [wrist[0], palm[0], index_tip[0], thumb_tip[0]]
    ys = [wrist[1], palm[1], index_tip[1], thumb_tip[1]]
    bbox = [min(xs) - 20.0, min(ys) - 20.0, max(xs) + 20.0, max(ys) + 20.0]

    return HandPose(
        handedness=handedness,
        confidence=0.95,
        bbox=bbox,
        landmarks=keypoints,
        wrist=wrist,
        palm_center=palm,
        index_tip=index_tip,
        thumb_tip=thumb_tip
    )

