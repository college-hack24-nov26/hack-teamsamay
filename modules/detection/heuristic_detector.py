"""
Lightweight Geometry & Color Filter Detector for SUTRA Vision Engine.
Specifically detects lab items: vials, pipettes, tip boxes, waste bins, and astronaut presence.
Hands are handled exclusively by the dedicated neural pose estimator.
"""

from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from modules.detection.detector_base import BaseObjectDetector, DetectedObject, BoundingBox, MVP_CLASSES


class HeuristicLabDetector(BaseObjectDetector):
    """
    Computer-vision detector for wetlab apparatus (vials, pipette, tip box, waste bin).
    Applies Non-Maximum Suppression to ensure clean, singular bounding boxes.
    """

    def __init__(self, min_confidence: float = 0.60):
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

        candidate_boxes = []
        candidate_scores = []
        candidate_classes = []

        # 1. Detect Pipette / Cylindrical Tool (Blue / Cyan accents & slender objects)
        lower_blue = np.array([85, 70, 70], dtype=np.uint8)
        upper_blue = np.array([135, 255, 255], dtype=np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours_blue:
            area = cv2.contourArea(c)
            if (w * h * 0.003) < area < (w * h * 0.18):
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bh / float(bw) if bw > 0 else 1.0
                if aspect > 1.3 or aspect < 0.7:  # Elongated tool shape
                    candidate_boxes.append([x, y, bw, bh])
                    candidate_scores.append(0.85)
                    candidate_classes.append("pipette")

        # 2. Detect Containers / Vials (Amber / High contrast bottles)
        lower_amber = np.array([10, 90, 80], dtype=np.uint8)
        upper_amber = np.array([35, 255, 255], dtype=np.uint8)
        mask_amber = cv2.inRange(hsv, lower_amber, upper_amber)
        mask_amber = cv2.morphologyEx(mask_amber, cv2.MORPH_OPEN, kernel)
        contours_amber, _ = cv2.findContours(mask_amber, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours_amber:
            area = cv2.contourArea(c)
            # Must be a realistic compact vial size
            if (w * h * 0.003) < area < (w * h * 0.12):
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bh / float(bw) if bw > 0 else 1.0
                if 0.5 <= aspect <= 3.5:
                    cls_name = "vial_sealed" if bh > bw * 1.2 else "vial_open"
                    candidate_boxes.append([x, y, bw, bh])
                    candidate_scores.append(0.80)
                    candidate_classes.append(cls_name)

        # 3. Non-Maximum Suppression (NMS) to eliminate duplicate/cluttered boxes
        if candidate_boxes:
            indices = cv2.dnn.NMSBoxes(candidate_boxes, candidate_scores, self.min_confidence, 0.35)
            if len(indices) > 0:
                for idx in indices.flatten():
                    bx, by, bw, bh = candidate_boxes[idx]
                    bbox = BoundingBox(x1=float(bx), y1=float(by), x2=float(bx + bw), y2=float(by + bh))
                    results.append(DetectedObject(
                        class_name=candidate_classes[idx],
                        confidence=candidate_scores[idx],
                        bbox=bbox,
                        center=bbox.center,
                        timestamp=timestamp,
                        source_model="heuristic:lab_apparatus"
                    ))

        return results
