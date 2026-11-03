"""
Mock Object Detector for Deterministic Testing and Regression Suites.
"""

from typing import Callable, Optional
import numpy as np

from modules.detection.detector_base import BaseObjectDetector, DetectedObject, BoundingBox, MVP_CLASSES


class MockObjectDetector(BaseObjectDetector):
    """
    Deterministic mock detector whose output can be programmed or driven
    by a generator function for testing edge cases, skipped steps, and occlusions.
    """

    def __init__(self, predefined_detections: Optional[list[DetectedObject]] = None):
        self._detections = predefined_detections or []
        self._provider_fn: Optional[Callable[[float], list[DetectedObject]]] = None

    def set_detections(self, detections: list[DetectedObject]) -> None:
        self._detections = list(detections)

    def set_provider(self, provider_fn: Callable[[float], list[DetectedObject]]) -> None:
        self._provider_fn = provider_fn

    def is_operational(self) -> bool:
        return True

    def get_supported_classes(self) -> list[str]:
        return list(MVP_CLASSES)

    def detect(self, frame: np.ndarray, timestamp: float) -> list[DetectedObject]:
        if self._provider_fn is not None:
            return self._provider_fn(timestamp)
        # Return static detections with updated timestamp
        res = []
        for d in self._detections:
            d_copy = d.model_copy()
            d_copy.timestamp = timestamp
            res.append(d_copy)
        return res

