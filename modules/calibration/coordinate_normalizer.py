"""
Rack-Relative Coordinate Normalizer for Microgravity Invariance.
Transforms camera pixel coordinates into normalized [0.0, 1.0] workspace rack coordinates
using 4-point perspective transformation. Fully independent of gravity vectors.
"""

import json
import logging
from typing import Optional, Union
import cv2
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger("sutra.calibration")


class RackPoint(BaseModel):
    """Normalized coordinate [0.0, 1.0] within the payload experiment rack."""
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized X [0.0, 1.0]")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized Y [0.0, 1.0]")
    z: float = Field(0.0, description="Optional normalized depth [0.0, 1.0]")
    calibration_status: str = Field("CALIBRATED", description="'CALIBRATED' or 'UNAVAILABLE'")


class RackCoordinateNormalizer:
    """
    Handles perspective homography transformation from raw camera pixel coordinates
    to a standardized [0.0, 1.0] x [0.0, 1.0] payload rack reference frame.
    """

    def __init__(
        self,
        anchor_points: Optional[list[tuple[float, float]]] = None,
        image_dims: tuple[int, int] = (640, 480)
    ):
        """
        anchor_points: 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] in pixel space:
        Order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left] of the experiment rack.
        """
        self.image_dims = image_dims
        self._anchor_points = anchor_points
        self._transform_matrix: Optional[np.ndarray] = None
        self._is_calibrated = False

        if anchor_points and len(anchor_points) == 4:
            self.calibrate(anchor_points)

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def calibrate(self, points: list[tuple[float, float]]) -> bool:
        """
        Compute perspective homography matrix using 4 corners of payload rack.
        """
        if len(points) != 4:
            logger.error("Calibration requires exactly 4 anchor points.")
            self._is_calibrated = False
            return False

        src = np.array(points, dtype=np.float32)
        # Target canonical rack coordinates: (0,0), (1,0), (1,1), (0,1)
        dst = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ], dtype=np.float32)

        try:
            self._transform_matrix = cv2.getPerspectiveTransform(src, dst)
            self._anchor_points = points
            self._is_calibrated = True
            logger.info(f"Rack coordinate normalizer successfully calibrated with anchors: {points}")
            return True
        except Exception as e:
            logger.error(f"Failed to calculate perspective transform: {e}")
            self._is_calibrated = False
            self._transform_matrix = None
            return False

    def normalize_point(self, px: float, py: float) -> RackPoint:
        """
        Transform a single pixel point (px, py) to normalized rack coordinates.
        If calibration is unavailable, falls back gracefully to image-relative [0, 1].
        """
        if self._is_calibrated and self._transform_matrix is not None:
            pt = np.array([[[px, py]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self._transform_matrix)
            rx = float(transformed[0, 0, 0])
            ry = float(transformed[0, 0, 1])
            # Clamp to [0.0, 1.0]
            rx = max(0.0, min(1.0, rx))
            ry = max(0.0, min(1.0, ry))
            return RackPoint(x=round(rx, 4), y=round(ry, 4), calibration_status="CALIBRATED")
        else:
            # Fallback: simple image dimension normalization
            w, h = self.image_dims
            rx = max(0.0, min(1.0, px / float(w))) if w > 0 else 0.0
            ry = max(0.0, min(1.0, py / float(h))) if h > 0 else 0.0
            return RackPoint(x=round(rx, 4), y=round(ry, 4), calibration_status="UNAVAILABLE")

    def normalize_bbox(self, bbox: list[float]) -> list[float]:
        """Convert [x1, y1, x2, y2] to normalized rack coordinates."""
        p1 = self.normalize_point(bbox[0], bbox[1])
        p2 = self.normalize_point(bbox[2], bbox[3])
        return [p1.x, p1.y, p2.x, p2.y]

    def save_calibration(self, filepath: str) -> None:
        """Save anchor points to a JSON file."""
        data = {
            "image_dims": self.image_dims,
            "anchor_points": self._anchor_points,
            "is_calibrated": self._is_calibrated
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_calibration(self, filepath: str) -> bool:
        """Load anchor points from a JSON file and recalculate homography."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            pts = [tuple(p) for p in data.get("anchor_points", [])]
            dims = tuple(data.get("image_dims", (640, 480)))
            self.image_dims = dims
            if len(pts) == 4:
                return self.calibrate(pts)
        except Exception as e:
            logger.warning(f"Could not load calibration file '{filepath}': {e}")
        return False
