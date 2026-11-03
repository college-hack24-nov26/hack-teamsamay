"""
Tests for Telemetry Event Schema and Local Storage.
Verifies JSON serialization, schema versioning, append-only JSONL recording, and export.
"""

import json
import os
import pytest

from modules.telemetry.event_schema import TelemetryEvent, TelemetryEventType, RackCoordinates
from modules.telemetry.event_storage import EventStorage


def test_telemetry_event_schema_and_serialization():
    coord = RackCoordinates(x=0.45, y=0.62, calibration_status="CALIBRATED")
    event = TelemetryEvent(
        device_id="ai-device-01",
        experiment_id="DEMO-WETLAB-01",
        event_type=TelemetryEventType.STEP_COMPLETED,
        step_id="STEP_03",
        step_name="Reagent Aspiration",
        status="COMPLETED",
        confidence=0.91,
        evidence={"duration_seconds": 3.5, "object_classes": ["pipette", "vial_open"]},
        rack_coordinates=coord,
        requires_review=False
    )

    json_str = event.to_json_str()
    data = json.loads(json_str)

    assert data["schema_version"] == "1.0"
    assert data["source"] == "sutra-ai-engine"
    assert data["device_id"] == "ai-device-01"
    assert data["event_type"] == "STEP_COMPLETED"
    assert data["confidence"] == 0.91
    assert data["rack_coordinates"]["x"] == 0.45


def test_event_storage_jsonl_and_export(tmp_path):
    log_file = str(tmp_path / "test_events.jsonl")
    export_file = str(tmp_path / "test_export.json")

    storage = EventStorage(log_file)

    e1 = TelemetryEvent(event_type=TelemetryEventType.SYSTEM_STATUS, confidence=1.0)
    e2 = TelemetryEvent(event_type=TelemetryEventType.STEP_STARTED, step_id="STEP_01", confidence=0.85)
    e3 = TelemetryEvent(event_type=TelemetryEventType.DEVIATION_DETECTED, step_id="STEP_02", confidence=0.95)

    storage.record_event(e1)
    storage.record_event(e2)
    storage.record_event(e3)

    assert storage.get_stats()["total_events"] == 3
    assert storage.get_stats()["total_bytes_written"] > 0

    read_back = storage.read_events()
    assert len(read_back) == 3
    assert read_back[1].step_id == "STEP_01"

    count = storage.export_to_json(export_file)
    assert count == 3
    assert os.path.isfile(export_file)

    with open(export_file, "r") as f:
        exported_array = json.load(f)
    assert len(exported_array) == 3
    assert exported_array[2]["event_type"] == "DEVIATION_DETECTED"

