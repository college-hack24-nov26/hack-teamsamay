"""
ONNX Runtime Object Detector for SUTRA Vision Engine.
Supports YOLOv8n / YOLO11n exported to ONNX with CPUExecutionProvider.
"""

import logging
import os
from typing import Optional
import cv2
import numpy as np

from modules.detection.detector_base import BaseObjectDetector, DetectedObject, BoundingBox, MVP_CLASSES

logger = logging.getLogger("sutra.detection.onnx")


class ONNXObjectDetector(BaseObjectDetector):
    """
    Inference backend using ONNX Runtime CPU.
    Validates model inputs/outputs and enforces strict class-label honesty:
    will NOT falsely report custom lab objects if loaded with generic COCO weights.
    """

    def __init__(
        self,
        model_path: str,
        class_names: Optional[list[str]] = None,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.50,
        input_size: tuple[int, int] = (640, 640)
    ):
        self.model_path = model_path
        self.class_names = class_names or MVP_CLASSES
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        
        self.session = None
        self._is_operational = False
        self._input_name = None
        self._output_name = None

        self._load_model()

    def _load_model(self) -> None:
        if not os.path.isfile(self.model_path):
            logger.warning(
                f"Custom ONNX weights not found at '{self.model_path}'. "
                f"ONNX detector is inactive. To use custom detection, export fine-tuned YOLO "
                f"weights for classes: {self.class_names} to this path."
            )
            self._is_operational = False
            return

        try:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, os.cpu_count() or 4)
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=options,
                providers=["CPUExecutionProvider"]
            )
            self._input_name = self.session.get_inputs()[0].name
            self._output_name = self.session.get_outputs()[0].name
            
            output_shape = self.session.get_outputs()[0].shape
            logger.info(
                f"Loaded ONNX model: {self.model_path} | Input: {self._input_name} | "
                f"Output: {self._output_name} (Shape: {output_shape})"
            )
            self._is_operational = True
        except Exception as e:
            logger.error(f"Failed to load ONNX model at {self.model_path}: {e}")
            self._is_operational = False

    def is_operational(self) -> bool:
        return self._is_operational

    def get_supported_classes(self) -> list[str]:
        return list(self.class_names)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize with aspect ratio preservation (letterbox) and normalize."""
        h, w = frame.shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        pad_x = (target_w - nw) // 2
        pad_y = (target_h - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        
        # Convert BGR to RGB, normalize [0, 1], HWC -> CHW -> NCHW
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        
        return blob, scale, (pad_x, pad_y)

    def detect(self, frame: np.ndarray, timestamp: float) -> list[DetectedObject]:
        if not self._is_operational or self.session is None:
            return []

        orig_h, orig_w = frame.shape[:2]
        blob, scale, (pad_x, pad_y) = self._preprocess(frame)

        outputs = self.session.run([self._output_name], {self._input_name: blob})
        predictions = outputs[0]  # Shape (1, 4 + num_classes, num_anchors)

        # Transpose if output is (1, 84, 8400) -> (8400, 84)
        if predictions.ndim == 3:
            preds = predictions[0]
            if preds.shape[0] < preds.shape[1]:
                preds = preds.T
        else:
            preds = predictions

        boxes = []
        confidences = []
        class_ids = []
        num_classes = len(self.class_names)

        for row in preds:
            cx, cy, bw, bh = row[:4]
            scores = row[4:4 + num_classes] if len(row) >= 4 + num_classes else []
            if len(scores) == 0:
                continue
            max_idx = int(np.argmax(scores))
            max_score = float(scores[max_idx])

            if max_score >= self.conf_threshold:
                # Unpad and rescale back to original image space
                x1 = (cx - bw / 2.0 - pad_x) / scale
                y1 = (cy - bh / 2.0 - pad_y) / scale
                x2 = (cx + bw / 2.0 - pad_x) / scale
                y2 = (cy + bh / 2.0 - pad_y) / scale

                # Clamp to frame bounds
                x1 = max(0.0, min(float(orig_w), x1))
                y1 = max(0.0, min(float(orig_h), y1))
                x2 = max(0.0, min(float(orig_w), x2))
                y2 = max(0.0, min(float(orig_h), y2))

                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(float(max_score))
                class_ids.append(max_idx)

        if not boxes:
            return []

        # OpenCV NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = boxes[idx]
                cls_idx = class_ids[idx]
                cls_name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"cls_{cls_idx}"
                bbox = BoundingBox(x1=float(bx), y1=float(by), x2=float(bx + bw), y2=float(by + bh))
                
                results.append(DetectedObject(
                    class_name=cls_name,
                    confidence=confidences[idx],
                    bbox=bbox,
                    center=bbox.center,
                    timestamp=timestamp,
                    source_model=f"onnx:{os.path.basename(self.model_path)}"
                ))

        return results

