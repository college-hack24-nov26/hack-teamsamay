"""
Lightweight Color & Geometry Heuristic Detector for SUTRA Vision Engine.
Provides CPU-efficient fallback localization of lab items without requiring
large deep-learning weights. Fully transparent about its heuristic nature.
"""

from typing import Optional
import cv2
import numpy as np

from modules.detection.detector_base import BaseObjectDetector, DetectedObject, BoundingBox, MVP_CLASSES


class HeuristicLabDetector(BaseObjectDetector):
    """
    Computer-vision heuristic detector based on color thresholds,
    aspect ratios, and contour geometry.
    Useful for standalone demo pipelines and offline testing.
    """

    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = min_confidence
        self.classes = MVP_CLASSES

    def is_operational(self) -> bool:
        return True

    def get_supported_classes(self) -> list[str]:
        return list(self.classes)

    def detect(self, frame: np.ndarray, timestamp: float) -> list[DetectedObject]:
        results: list[DetectedObject] = []
        if frame is None or frame.size == 0:
            return results

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Detect Hand (Skin tone range in HSV)
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([25, 200, 255], dtype=np.uint8)
        mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morphological clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            area = cv2.contourArea(c)
            if area > (w * h * 0.015):  # At least 1.5% of frame
                x, y, bw, bh = cv2.boundingRect(c)
                bbox = BoundingBox(x1=float(x), y1=float(y), x2=float(x + bw), y2=float(y + bh))
                results.append(DetectedObject(
                    class_name="hand",
                    confidence=0.82,
                    bbox=bbox,
                    center=bbox.center,
                    timestamp=timestamp,
                    source_model="heuristic:skin_geometry"
                ))

        # 2. Detect Pipette / Cylinder (Blue or Dark Cyan accents common on lab tools)
        lower_blue = np.array([90, 70, 70], dtype=np.uint8)
        upper_blue = np.array([130, 255, 255], dtype=np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours_blue:
            area = cv2.contourArea(c)
            if area > 400:
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bh / float(bw) if bw > 0 else 1.0
                if aspect > 1.8 or aspect < 0.5: # Elongated tool shape
                    bbox = BoundingBox(x1=float(x), y1=float(y), x2=float(x + bw), y2=float(y + bh))
                    results.append(DetectedObject(
                        class_name="pipette",
                        confidence=0.78,
                        bbox=bbox,
                        center=bbox.center,
                        timestamp=timestamp,
                        source_model="heuristic:tool_aspect"
                    ))

        # 3. Detect Containers / Vials (Yellow/Orange liquid or vial caps)
        lower_amber = np.array([15, 100, 100], dtype=np.uint8)
        upper_amber = np.array([35, 255, 255], dtype=np.uint8)
        mask_amber = cv2.inRange(hsv, lower_amber, upper_amber)
        contours_amber, _ = cv2.findContours(mask_amber, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours_amber:
            area = cv2.contourArea(c)
            if 150 < area < (w * h * 0.08):
                x, y, bw, bh = cv2.boundingRect(c)
                bbox = BoundingBox(x1=float(x), y1=float(y), x2=float(x + bw), y2=float(y + bh))
                # Classify vial state
                cls_name = "vial_sealed" if bh > bw * 1.2 else "vial_open"
                results.append(DetectedObject(
                    class_name=cls_name,
                    confidence=0.74,
                    bbox=bbox,
                    center=bbox.center,
                    timestamp=timestamp,
                    source_model="heuristic:amber_vial"
                ))

        return results

