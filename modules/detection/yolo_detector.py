"""
Ultralytics YOLOv8 Deep Learning Detector for SUTRA Vision Engine.
Provides high-speed neural object detection on CPU with COCO-to-Lab class mapping,
universal apparatus detection for any held item, and face suppression.
"""

import logging
import os
from pathlib import Path
from typing import Optional
import numpy as np

from modules.detection.detector_base import BaseObjectDetector, DetectedObject, BoundingBox, MVP_CLASSES

logger = logging.getLogger("sutra.detection.yolo")

# Mapping from COCO classes to SUTRA wetlab experiment targets
COCO_TO_LAB_MAPPING = {
    # Vials & Containers
    "bottle": "vial_sealed",
    "wine glass": "vial_open",
    "cup": "vial_open",
    # Tools & Pipettes
    "scissors": "pipette",
    "knife": "pipette",
    "fork": "pipette",
    "spoon": "pipette",
    "toothbrush": "pipette",
    "pen": "pipette",
    # Apparatus & Racks & Tip boxes
    "bowl": "tip_box",
    "cell phone": "apparatus",
    "book": "apparatus",
    "mouse": "apparatus",
    "remote": "pipette",
    "keyboard": "apparatus",
    "laptop": "apparatus",
    "box": "tip_box",
    "clock": "timer_gauge",
}


class UltralyticsYOLODetector(BaseObjectDetector):
    """
    State-of-the-art YOLOv8 deep learning detector.
    Runs on CPU with optimized 320x320 inference (~25ms).
    Detects both standard lab items and generic handheld tools/apparatus.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.20,
        input_size: tuple[int, int] = (320, 320)
    ):
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self._is_operational = False
        self._model = None

        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            model_path = str(base_dir / "models" / "yolov8n.pt")

        self.model_path = model_path
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.isfile(self.model_path):
            logger.warning(f"YOLOv8 weights not found at {self.model_path}")
            self._is_operational = False
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._is_operational = True
            logger.info(f"Ultralytics YOLOv8 detector loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load Ultralytics YOLO: {e}")
            self._is_operational = False

    def is_operational(self) -> bool:
        return self._is_operational

    def get_supported_classes(self) -> list[str]:
        return list(MVP_CLASSES) + ["apparatus"]

    def detect(self, frame: np.ndarray, timestamp: float) -> list[DetectedObject]:
        if not self._is_operational or self._model is None or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        try:
            # High-speed inference at 320x320 resolution
            results = self._model(frame, imgsz=320, conf=self.conf_threshold, verbose=False)
        except Exception as e:
            logger.error(f"Error during YOLOv8 inference: {e}")
            return []

        if not results or len(results) == 0:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detected_objects: list[DetectedObject] = []
        names = results[0].names

        # First pass: find person bounding boxes to isolate head region
        person_head_zones = []
        for box in boxes:
            cls_id = int(box.cls[0])
            name = names.get(cls_id, "")
            if name == "person":
                xyxy = box.xyxy[0].tolist()
                px1, py1, px2, py2 = xyxy
                pw = px2 - px1
                ph = py2 - py1
                # Head zone is top 30% and centered horizontally
                hx1 = px1 + pw * 0.20
                hx2 = px2 - pw * 0.20
                hy1 = py1
                hy2 = py1 + ph * 0.35
                person_head_zones.append((hx1, hy1, hx2, hy2))

        # Second pass: extract lab objects and apparatus
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name_raw = names.get(cls_id, "")
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy

            # Skip person class itself
            if cls_name_raw == "person":
                continue

            # Check if object is inside face/head zone
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            is_on_face = False
            for (hx1, hy1, hx2, hy2) in person_head_zones:
                if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
                    is_on_face = True
                    break
            if is_on_face:
                continue

            # Map COCO class to Lab Class, or default to apparatus
            lab_class = COCO_TO_LAB_MAPPING.get(cls_name_raw, "apparatus")

            bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            detected_objects.append(DetectedObject(
                class_name=lab_class,
                confidence=round(conf, 2),
                bbox=bbox,
                center=bbox.center,
                timestamp=timestamp,
                source_model=f"yolo:{cls_name_raw}"
            ))

        return detected_objects
