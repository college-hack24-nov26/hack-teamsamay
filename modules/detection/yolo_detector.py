"""
Ultralytics YOLOv8 Deep Learning Detector for SUTRA Vision Engine.
Provides real-world neural object detection on CPU with COCO-to-Lab class mapping
and astronaut head/face suppression to eliminate misclassifications.
"""

import logging
import os
from pathlib import Path
from typing import Optional
import numpy as np

from modules.detection.detector_base import BaseObjectDetector, DetectedObject, BoundingBox, MVP_CLASSES

logger = logging.getLogger("sutra.detection.yolo")

# Mapping from common COCO classes to SUTRA wetlab experiment targets
COCO_TO_LAB_MAPPING = {
    "bottle": "vial_sealed",
    "cup": "waste_bin",
    "bowl": "tip_box",
    "scissors": "pipette",
    "knife": "pipette",
    "spoon": "pipette",
    "toothbrush": "pipette",
    "remote": "pipette",
    "cell phone": "tip_box",
}


class UltralyticsYOLODetector(BaseObjectDetector):
    """
    State-of-the-art YOLOv8 deep learning detector.
    Runs on CPU without GPU requirements.
    Detects real lab items and accurately localizes human presence.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.40,
        input_size: tuple[int, int] = (640, 480)
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
        return list(MVP_CLASSES)

    def detect(self, frame: np.ndarray, timestamp: float) -> list[DetectedObject]:
        if not self._is_operational or self._model is None or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        try:
            results = self._model(frame, conf=self.conf_threshold, verbose=False)
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
                # Approximate upper 35% of person as head/face zone
                head_y2 = py1 + (py2 - py1) * 0.40
                person_head_zones.append((px1, py1, px2, head_y2))

        # Second pass: extract lab objects (strictly suppressing false objects on the head)
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name_raw = names.get(cls_id, "")
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = xyxy

            # Map COCO class to Lab Class
            lab_class = COCO_TO_LAB_MAPPING.get(cls_name_raw)
            if not lab_class:
                continue

            # Ensure object is not inside the user's face/head!
            is_on_head = False
            for (hx1, hy1, hx2, hy2) in person_head_zones:
                if x1 >= hx1 - 20 and x2 <= hx2 + 20 and y1 >= hy1 - 20 and y2 <= hy2:
                    is_on_head = True
                    break
            if is_on_head:
                continue

            bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            detected_objects.append(DetectedObject(
                class_name=lab_class,
                confidence=round(conf, 2),
                bbox=bbox,
                center=bbox.center,
                timestamp=timestamp,
                source_model="yolov8n"
            ))

        return detected_objects
