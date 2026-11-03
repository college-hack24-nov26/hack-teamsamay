"""
Lightweight Color & Geometry Heuristic Detector for SUTRA Vision Engine.
Includes Face Suppression to prevent human faces from being misclassified as hands,
and Non-Maximum Suppression (NMS) to eliminate duplicate/cluttered bounding boxes.
"""

import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from modules.detection.detector_base import BaseObjectDetector, DetectedObject, BoundingBox, MVP_CLASSES


class HeuristicLabDetector(BaseObjectDetector):
    """
    Computer-vision detector based on color thresholds, aspect ratios,
    face-suppression masking, and Non-Maximum Suppression.
    """

    def __init__(self, min_confidence: float = 0.65, face_cascade_path: Optional[str] = None):
        self.min_confidence = min_confidence
        self.classes = MVP_CLASSES

        if face_cascade_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            face_cascade_path = str(base_dir / "models" / "haarcascade_frontalface_default.xml")

        self.face_cascade = None
        if os.path.isfile(face_cascade_path):
            try:
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            except Exception:
                self.face_cascade = None

    def is_operational(self) -> bool:
        return True

    def get_supported_classes(self) -> list[str]:
        return list(self.classes)

    def detect(self, frame: np.ndarray, timestamp: float) -> list[DetectedObject]:
        results: list[DetectedObject] = []
        if frame is None or frame.size == 0:
            return results

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Detect and Mask Out Faces (Crucial to prevent face misclassification as hand)
        face_boxes = []
        if self.face_cascade is not None and not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))
            for (fx, fy, fw, fh) in faces:
                face_boxes.append((fx, fy, fw, fh))

        # 2. Detect Hand (Skin tone range in HSV)
        lower_skin = np.array([0, 35, 60], dtype=np.uint8)
        upper_skin = np.array([25, 185, 255], dtype=np.uint8)
        mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)

        # Suppress face regions from skin mask with generous margin
        for (fx, fy, fw, fh) in face_boxes:
            pad_w = int(fw * 0.2)
            pad_h = int(fh * 0.3)
            x_start = max(0, fx - pad_w)
            y_start = max(0, fy - pad_h)
            x_end = min(w, fx + fw + pad_w)
            y_end = min(h, fy + fh + int(pad_h * 1.5))
            cv2.rectangle(mask_skin, (x_start, y_start), (x_end, y_end), 0, -1)

        # Morphological clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_OPEN, kernel)
        mask_skin = cv2.morphologyEx(mask_skin, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_boxes = []
        candidate_scores = []
        candidate_classes = []

        for c in contours:
            area = cv2.contourArea(c)
            # Acceptable hand area range: at least 1.5% and at most 35% of workspace
            if (w * h * 0.015) < area < (w * h * 0.35):
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bh / float(bw) if bw > 0 else 1.0
                # Hands/forearms are usually oriented or compact, not gigantic blocks
                if 0.35 <= aspect <= 3.2:
                    candidate_boxes.append([x, y, bw, bh])
                    candidate_scores.append(0.85)
                    candidate_classes.append("hand")

        # 3. Detect Pipette / Cylindrical Tool (Blue/Cyan accents)
        lower_blue = np.array([90, 70, 70], dtype=np.uint8)
        upper_blue = np.array([130, 255, 255], dtype=np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours_blue:
            area = cv2.contourArea(c)
            if area > 450:
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bh / float(bw) if bw > 0 else 1.0
                if aspect > 1.8 or aspect < 0.5:
                    candidate_boxes.append([x, y, bw, bh])
                    candidate_scores.append(0.82)
                    candidate_classes.append("pipette")

        # 4. Detect Containers / Vials (Amber / High contrast bottles)
        lower_amber = np.array([15, 100, 100], dtype=np.uint8)
        upper_amber = np.array([35, 255, 255], dtype=np.uint8)
        mask_amber = cv2.inRange(hsv, lower_amber, upper_amber)
        contours_amber, _ = cv2.findContours(mask_amber, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours_amber:
            area = cv2.contourArea(c)
            if 200 < area < (w * h * 0.08):
                x, y, bw, bh = cv2.boundingRect(c)
                cls_name = "vial_sealed" if bh > bw * 1.2 else "vial_open"
                candidate_boxes.append([x, y, bw, bh])
                candidate_scores.append(0.78)
                candidate_classes.append(cls_name)

        # 5. Non-Maximum Suppression (NMS) to eliminate multiple/overlapping duplicate boxes
        if candidate_boxes:
            indices = cv2.dnn.NMSBoxes(candidate_boxes, candidate_scores, self.min_confidence, 0.40)
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
                        source_model="heuristic:filtered"
                    ))

        return results
