"""
Tests for Object Detection Module.
Verifies BoundingBox math, IoU calculations, detector interfaces, and fallback behaviors.
"""

import numpy as np
import pytest

from modules.detection.detector_base import BoundingBox, DetectedObject, MVP_CLASSES
from modules.detection.heuristic_detector import HeuristicLabDetector
from modules.detection.mock_detector import MockObjectDetector
from modules.detection.onnx_detector import ONNXObjectDetector
from scripts.generate_synthetic_video import render_scene


def test_bounding_box_geometry_and_iou():
    box1 = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0)
    assert box1.width == 40.0
    assert box1.height == 40.0
    assert box1.area == 1600.0
    assert box1.center == (30.0, 30.0)

    # Identical box -> IoU = 1.0
    assert box1.iou(box1) == 1.0

    # Non-overlapping box -> IoU = 0.0
    box2 = BoundingBox(x1=100.0, y1=100.0, x2=150.0, y2=150.0)
    assert box1.iou(box2) == 0.0

    # 50% overlap box
    box3 = BoundingBox(x1=30.0, y1=10.0, x2=70.0, y2=50.0)
    # intersection: [30, 10] to [50, 50] = 20 * 40 = 800
    # union: 1600 + 1600 - 800 = 2400 -> IoU = 800 / 2400 = 1/3
    assert abs(box1.iou(box3) - (1.0 / 3.0)) < 0.01


def test_heuristic_detector_identifies_synthetic_lab_items():
    detector = HeuristicLabDetector()
    assert detector.is_operational() is True
    assert set(detector.get_supported_classes()) == set(MVP_CLASSES)

    # Render scene where pipette and vial are present
    frame = render_scene(frame_idx=80, total_frames=180)
    detections = detector.detect(frame, timestamp=1.0)

    class_names = [d.class_name for d in detections]
    assert len(detections) > 0
    # Must identify at least one lab item or hand
    assert any(c in ["hand", "pipette", "vial_sealed", "vial_open"] for c in class_names)

    for d in detections:
        assert 0.0 <= d.confidence <= 1.0
        assert d.bbox.width > 0
        assert d.bbox.height > 0


def test_mock_detector():
    box = BoundingBox(x1=20.0, y1=20.0, x2=40.0, y2=40.0)
    det = DetectedObject(
        class_name="pipette",
        confidence=0.92,
        bbox=box,
        center=box.center,
        timestamp=0.0
    )
    mock = MockObjectDetector([det])
    out = mock.detect(np.zeros((100, 100, 3), dtype=np.uint8), timestamp=2.5)
    assert len(out) == 1
    assert out[0].class_name == "pipette"
    assert out[0].timestamp == 2.5


def test_onnx_detector_graceful_missing_weights():
    """Verify that ONNX detector does not crash or claim operation if file is missing."""
    onnx_det = ONNXObjectDetector(model_path="non_existent_weights.onnx")
    assert onnx_det.is_operational() is False
    assert onnx_det.detect(np.zeros((100, 100, 3), dtype=np.uint8), 0.0) == []
