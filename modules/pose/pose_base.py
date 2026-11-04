"""
Hand Pose and Keypoint Models and Interfaces for SUTRA Vision Engine.
"""

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, Field
import numpy as np

# Standard 21 Hand Keypoint Names (MediaPipe / OpenPose standard)
HAND_KEYPOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]


class HandKeypoint(BaseModel):
    """Normalized or pixel keypoint."""
    id: int = 0
    name: str = "wrist"
    x: float
    y: float
    z: float = 0.0
    x_px: float = 0.0
    y_px: float = 0.0
    confidence: float = 1.0
    visibility: float = 1.0


class HandPose(BaseModel):
    """Estimated hand structure with anatomically critical landmarks."""
    handedness: str = Field("Right", description="'Left' or 'Right'")
    confidence: float = 0.85
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2] bounding hand")
    landmarks: list[HandKeypoint] = Field(default_factory=list)
    wrist: tuple[float, float]
    palm_center: tuple[float, float]
    index_tip: tuple[float, float]
    thumb_tip: tuple[float, float]

    @property
    def pinch_distance(self) -> float:
        """Euclidean distance between thumb tip and index finger tip."""
        dx = self.thumb_tip[0] - self.index_tip[0]
        dy = self.thumb_tip[1] - self.index_tip[1]
        return float(np.hypot(dx, dy))


class BasePoseEstimator(ABC):
    """Abstract interface for all hand pose estimation backends."""

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> list[HandPose]:
        """
        Extract hand poses and 21 keypoints from a single BGR frame.
        """
        pass

    @abstractmethod
    def is_operational(self) -> bool:
        """Check if the pose engine is operational."""
        pass
