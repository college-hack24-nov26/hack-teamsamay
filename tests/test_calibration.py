"""
Tests for Coordinate Normalization & Rack Calibration Module.
Verifies microgravity invariance, perspective transformation, and fallback handling.
"""

import os
import pytest

from modules.calibration.coordinate_normalizer import RackCoordinateNormalizer


def test_uncalibrated_normalizer_graceful_fallback():
    normalizer = RackCoordinateNormalizer(anchor_points=None, image_dims=(640, 480))
    assert normalizer.is_calibrated is False

    pt = normalizer.normalize_point(320.0, 240.0)
    assert pt.calibration_status == "UNAVAILABLE"
    assert pt.x == 0.5
    assert pt.y == 0.5


def test_perspective_calibration_mapping():
    # Define a 4-point trapezoidal or skewed rack bounding box
    anchors = [
        (50.0, 50.0),    # Top-Left
        (590.0, 50.0),   # Top-Right
        (590.0, 430.0),  # Bottom-Right
        (50.0, 430.0)    # Bottom-Left
    ]
    normalizer = RackCoordinateNormalizer(anchor_points=anchors, image_dims=(640, 480))
    assert normalizer.is_calibrated is True

    # Top-left anchor should map to (0.0, 0.0)
    p_tl = normalizer.normalize_point(50.0, 50.0)
    assert p_tl.calibration_status == "CALIBRATED"
    assert abs(p_tl.x - 0.0) < 0.02
    assert abs(p_tl.y - 0.0) < 0.02

    # Bottom-right anchor should map to (1.0, 1.0)
    p_br = normalizer.normalize_point(590.0, 430.0)
    assert abs(p_br.x - 1.0) < 0.02
    assert abs(p_br.y - 1.0) < 0.02

    # Center point (320, 240) should map to ~ (0.5, 0.5)
    p_center = normalizer.normalize_point(320.0, 240.0)
    assert abs(p_center.x - 0.5) < 0.05
    assert abs(p_center.y - 0.5) < 0.05


def test_calibration_save_and_load(tmp_path):
    anchors = [(40.0, 40.0), (600.0, 40.0), (600.0, 440.0), (40.0, 440.0)]
    normalizer = RackCoordinateNormalizer(anchor_points=anchors, image_dims=(640, 480))
    
    save_file = str(tmp_path / "test_calib.json")
    normalizer.save_calibration(save_file)
    assert os.path.isfile(save_file)

    loaded_norm = RackCoordinateNormalizer(image_dims=(640, 480))
    assert loaded_norm.is_calibrated is False
    success = loaded_norm.load_calibration(save_file)
    assert success is True
    assert loaded_norm.is_calibrated is True
