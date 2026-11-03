"""
Tests for Temporal HAR & Debounce Engine.
Verifies sliding-window consensus, hysteresis debounce, and short occlusion tolerance.
"""

import pytest

from modules.temporal.temporal_engine import DeterministicTemporalEngine
from modules.interaction.interaction_engine import InteractionRelation
from modules.detection.detector_base import DetectedObject, BoundingBox
from modules.pose.mock_pose import create_synthetic_hand_pose


def test_temporal_debounce_requires_consensus():
    """Verify that a single-frame flicker does not prematurely confirm an action."""
    engine = DeterministicTemporalEngine(window_size=10, consensus_threshold=5)

    inter = InteractionRelation(
        subject="Hand_R",
        verb="GRASPS",
        target="pipette",
        confidence=0.88,
        duration_seconds=1.0,
        distance_norm=0.05
    )

    dummy_box = BoundingBox(x1=10, y1=10, x2=50, y2=50)
    pip = DetectedObject(class_name="pipette", confidence=0.9, bbox=dummy_box, center=dummy_box.center, timestamp=1.0)
    vial = DetectedObject(class_name="vial_open", confidence=0.8, bbox=dummy_box, center=dummy_box.center, timestamp=1.0)
    hand = create_synthetic_hand_pose()

    # Frame 1: instant candidate appears, but not confirmed yet
    st1 = engine.push_frame([pip, vial], [hand], [inter], timestamp=1.0)
    assert st1.candidate_activity == "DISPENSING_LIQUID"
    assert st1.confirmed_activity is None  # Not enough consensus frames!

    # Push frames 2, 3, 4
    for i in range(2, 5):
        st = engine.push_frame([pip, vial], [hand], [inter], timestamp=1.0 + i * 0.1)
        assert st.confirmed_activity is None

    # Frame 5: consensus threshold met -> confirmed!
    st5 = engine.push_frame([pip, vial], [hand], [inter], timestamp=1.5)
    assert st5.confirmed_activity == "DISPENSING_LIQUID"
    assert st5.duration_active >= 0.0


def test_short_occlusion_tolerance():
    """Verify that a brief visual occlusion (<= 0.6s) bridges state continuity."""
    engine = DeterministicTemporalEngine(window_size=15, consensus_threshold=3, max_occlusion_seconds=0.6)

    inter = InteractionRelation(
        subject="Hand_R",
        verb="GRASPS",
        target="pipette",
        confidence=0.88,
        duration_seconds=1.0,
        distance_norm=0.05
    )
    dummy_box = BoundingBox(x1=10, y1=10, x2=50, y2=50)
    pip = DetectedObject(class_name="pipette", confidence=0.9, bbox=dummy_box, center=dummy_box.center, timestamp=1.0)
    vial = DetectedObject(class_name="vial_open", confidence=0.8, bbox=dummy_box, center=dummy_box.center, timestamp=1.0)
    hand = create_synthetic_hand_pose()

    # Confirm action over 3 frames
    for t in [1.0, 1.1, 1.2]:
        engine.push_frame([pip, vial], [hand], [inter], timestamp=t)

    # Frame at t=1.4s: complete occlusion (nothing detected)
    st_occ = engine.push_frame([], [], [], timestamp=1.4)
    assert st_occ.is_occluded is True
    assert st_occ.confirmed_activity == "DISPENSING_LIQUID"

    # Frame at t=2.5s: occlusion exceeded max_occlusion_seconds (0.6s) -> dropped
    for t in [1.5, 1.7, 1.9, 2.1]:
        engine.push_frame([], [], [], timestamp=t)
    st_dropped = engine.push_frame([], [], [], timestamp=2.5)
    assert st_dropped.confirmed_activity is None
