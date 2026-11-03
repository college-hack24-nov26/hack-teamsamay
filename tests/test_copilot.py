"""
Unit tests for Autonomous Flight Copilot Engine.
Verifies autonomous state evaluation, directive generation, and deviation warnings.
"""

import pytest
from modules.assistant.copilot_engine import AutonomousCopilotEngine, CopilotDirective


def test_copilot_deterministic_rules():
    copilot = AutonomousCopilotEngine(api_key=None)  # Offline deterministic rules

    # 1. Step 1 Active
    ctx = {
        "protocol_name": "Demo Assay",
        "current_step_id": "STEP_01",
        "current_step_name": "Workspace Inspection",
        "step_status": "IN_PROGRESS",
        "active_deviation": None,
        "detected_objects": ["hand"],
        "active_action": "INSPECTING_WORKSPACE",
        "action_duration": 2.5
    }

    directive = copilot.evaluate_state_autonomously(ctx, force_urgent=True)
    assert directive is not None
    assert isinstance(directive, CopilotDirective)
    assert directive.alert_level == "INFO"
    assert "Step 1 active" in directive.current_status
    assert len(directive.spoken_announcement) > 0

    # 2. Procedural Deviation Detected
    ctx["active_deviation"] = {
        "type": "SKIPPED_STEP",
        "description": "Attempted STEP_03 before STEP_02",
        "uncompleted_step": "STEP_02",
        "attempted_step": "STEP_03"
    }
    copilot._last_deviation_hash = None  # Reset hash to trigger
    dev_directive = copilot.evaluate_state_autonomously(ctx, force_urgent=True)
    assert dev_directive is not None
    assert dev_directive.alert_level == "CRITICAL"
    assert "Procedural violation" in dev_directive.current_status
    assert "STEP_02" in dev_directive.missing_or_deviation
    assert "Caution!" in dev_directive.spoken_announcement

