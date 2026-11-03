"""
Object Detection Base Classes and Data Models for SUTRA Vision Engine.
"""

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, Field
import numpy as np

# Canonical lab and astronaut classes for the SUTRA on-board experiment assistant
MVP_CLASSES = [
    "hand",
    "vial_sealed",
    "vial_open",
    "pipette",
    "tip_box",
    "waste_bin",
    "astronaut",
    "apparatus"
]


class BoundingBox(BaseModel):
    """Normalized or absolute bounding box coordinates [x1, y1, x2, y2]."""
    x1: float = Field(..., description="Top-left X")
    y1: float = Field(..., description="Top-left Y")
    x2: float = Field(..., description="Bottom-right X")
    y2: float = Field(..., description="Bottom-right Y")

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection over Union with another box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


class DetectedObject(BaseModel):
    """Standardized detection result for a localized object."""
    class_name: str
    confidence: float
    bbox: BoundingBox
    center: tuple[float, float]
    tracking_id: Optional[int] = None
    timestamp: float
    source_model: str = "unknown"

    model_config = {
        "arbitrary_types_allowed": True
    }


class BaseObjectDetector(ABC):
    """Abstract interface for all SUTRA object detection backends."""

    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp: float) -> list[DetectedObject]:
        """
        Run inference on a single BGR frame.
        Returns a list of DetectedObject instances.
        """
        pass

    @abstractmethod
    def get_supported_classes(self) -> list[str]:
        """Return the list of class labels supported by this detector."""
        pass

    @abstractmethod
    def is_operational(self) -> bool:
        """Check if weights are loaded and the engine is ready for inference."""
        pass
