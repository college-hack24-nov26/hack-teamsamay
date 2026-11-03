"""
Tests for Hand-Object Interaction (HOI) Engine.
Verifies spatial proximity, verb classification (APPROACHES, GRASPS, ENTERS),
duration timers, and evidence payloads.
"""

import pytest

from modules.interaction.interaction_engine import InteractionEngine
from modules.detection.detector_base import DetectedObject, BoundingBox
from modules.pose.mock_pose import create_synthetic_hand_pose
from modules.calibration.coordinate_normalizer import RackCoordinateNormalizer


def test_hoi_grasp_and_duration_accumulation():
    engine = InteractionEngine(grasp_distance_threshold=0.15)
    normalizer = RackCoordinateNormalizer(image_dims=(640, 480))

    # Place pipette at (300, 200)
    pipette_box = BoundingBox(x1=280.0, y1=180.0, x2=320.0, y2=260.0)
    pipette = DetectedObject(
        class_name="pipette",
        confidence=0.90,
        bbox=pipette_box,
        center=pipette_box.center,
        timestamp=1.0
    )

    # Place hand directly at pipette
    hand = create_synthetic_hand_pose(
        wrist=(300.0, 320.0),
        palm=(300.0, 240.0),
        index_tip=(295.0, 200.0),
        thumb_tip=(285.0, 210.0),
        handedness="Right"
    )

    # First frame at t=1.0s
    rels_t1 = engine.evaluate([hand], [pipette], normalizer, timestamp=1.0)
    assert len(rels_t1) == 1
    rel = rels_t1[0]
    assert rel.subject == "Hand_R"
    assert rel.verb == "GRASPS"
    assert rel.target == "pipette"
    assert rel.duration_seconds == 0.0
    assert "disclaimer" in rel.model_dump()

    # Second frame at t=3.5s
    rels_t2 = engine.evaluate([hand], [pipette], normalizer, timestamp=3.5)
    assert len(rels_t2) == 1
    assert rels_t2[0].duration_seconds == 2.5
    assert rels_t2[0].verb == "GRASPS"


def test_hoi_tool_enters_vial():
    engine = InteractionEngine(grasp_distance_threshold=0.15)
    normalizer = RackCoordinateNormalizer(image_dims=(640, 480))

    pip_box = BoundingBox(x1=400.0, y1=200.0, x2=420.0, y2=300.0)
    pipette = DetectedObject(class_name="pipette", confidence=0.88, bbox=pip_box, center=pip_box.center, timestamp=1.0)

    vial_box = BoundingBox(x1=410.0, y1=280.0, x2=440.0, y2=360.0)
    vial = DetectedObject(class_name="vial_open", confidence=0.85, bbox=vial_box, center=vial_box.center, timestamp=1.0)

    rels = engine.evaluate([], [pipette, vial], normalizer, timestamp=1.0)
    tool_rels = [r for r in rels if r.verb == "ENTERS"]
    assert len(tool_rels) == 1
    assert tool_rels[0].subject == "Pipette"
    assert tool_rels[0].target == "vial_open"

