"""
Tests for Protocol DAG & Finite State Machine Engine.
Verifies step progression, duration enforcement, deviation detection for skipped/out-of-order steps,
and checkpoint save/restore for camera reconnection recovery.
"""

import os
import pytest

from modules.protocol.fsm_engine import ProtocolEngine, StepStatus
from modules.temporal.temporal_engine import TemporalActivityState
from modules.interaction.interaction_engine import InteractionRelation
from modules.detection.detector_base import DetectedObject, BoundingBox

PROTOCOL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "protocols", "demo_wetlab.yaml")


def test_protocol_loading():
    engine = ProtocolEngine(PROTOCOL_FILE)
    assert engine.protocol.protocol_id == "DEMO-WETLAB-01"
    assert len(engine.protocol.steps) == 5
    assert engine.protocol.steps[0].id == "STEP_01"
    assert engine.protocol.steps[4].id == "STEP_05"


def test_step_requires_minimum_duration():
    """Verify that a step will NOT complete prematurely simply on detection."""
    engine = ProtocolEngine(PROTOCOL_FILE)
    box = BoundingBox(x1=10, y1=10, x2=50, y2=50)
    hand_det = DetectedObject(class_name="hand", confidence=0.9, bbox=box, center=box.center, timestamp=1.0)

    temp_state = TemporalActivityState(
        confirmed_activity="INSPECTING_WORKSPACE",
        duration_active=0.2,
        confidence=0.85
    )

    # First call at t=1.0: triggers STEP_STARTED
    trigger, payload = engine.evaluate_step(temp_state, [], [hand_det], timestamp=1.0)
    assert trigger == "STEP_STARTED"
    assert engine.step_statuses["STEP_01"] == StepStatus.IN_PROGRESS

    # Second call at t=1.5s (only 0.5s elapsed, min duration is 1.5s): should NOT complete
    trigger2, _ = engine.evaluate_step(temp_state, [], [hand_det], timestamp=1.5)
    assert trigger2 is None
    assert engine.step_statuses["STEP_01"] == StepStatus.IN_PROGRESS

    # Third call at t=3.0s (2.0s elapsed >= 1.5s min): should complete!
    trigger3, payload3 = engine.evaluate_step(temp_state, [], [hand_det], timestamp=3.0)
    assert trigger3 == "STEP_COMPLETED"
    assert payload3["step_id"] == "STEP_01"
    assert engine.step_statuses["STEP_01"] == StepStatus.COMPLETED
    assert engine.current_step_idx == 1  # Advanced to STEP_02


def test_skipped_step_deviation():
    """Verify that jumping ahead in the protocol triggers DEVIATION_DETECTED."""
    engine = ProtocolEngine(PROTOCOL_FILE)
    box = BoundingBox(x1=10, y1=10, x2=50, y2=50)
    pip_det = DetectedObject(class_name="pipette", confidence=0.9, bbox=box, center=box.center, timestamp=1.0)

    # Currently at STEP_01 (Workspace Inspection)
    # Simulate astronaut directly performing DISPENSING_LIQUID (STEP_04 activity)
    temp_state = TemporalActivityState(
        confirmed_activity="DISPENSING_LIQUID",
        duration_active=1.0,
        confidence=0.88
    )

    trigger, payload = engine.evaluate_step(temp_state, [], [pip_det], timestamp=1.0)
    assert trigger == "DEVIATION_DETECTED"
    assert payload["deviation_type"] == "SKIPPED_STEP"
    assert payload["uncompleted_step"] == "STEP_01"
    assert payload["attempted_step"] == "STEP_04"
    assert engine.active_deviation is not None

    # Operator acknowledgement
    engine.acknowledge_deviation()
    assert engine.active_deviation is None


def test_checkpoint_save_and_restore():
    """Verify state recovery across camera disconnections."""
    engine = ProtocolEngine(PROTOCOL_FILE)
    engine.current_step_idx = 2
    engine.step_statuses["STEP_01"] = StepStatus.COMPLETED
    engine.step_statuses["STEP_02"] = StepStatus.COMPLETED
    engine.step_statuses["STEP_03"] = StepStatus.IN_PROGRESS

    checkpoint = engine.save_checkpoint()
    assert checkpoint["current_step_idx"] == 2

    # Reset
    engine.reset()
    assert engine.current_step_idx == 0

    # Restore
    success = engine.restore_checkpoint(checkpoint)
    assert success is True
    assert engine.current_step_idx == 2
    assert engine.step_statuses["STEP_02"] == StepStatus.COMPLETED

