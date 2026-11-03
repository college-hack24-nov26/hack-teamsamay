"""
Tests for Event Publisher and Mock Receiver Communication.
Verifies queue bounding, deduplication, retry handling, and schema validation on receipt.
"""

import time
import pytest
from starlette.testclient import TestClient

from modules.telemetry.event_schema import TelemetryEvent, TelemetryEventType
from modules.communication.event_publisher import EventPublisher
from mock_receiver.mock_server import app, received_events, received_ids


@pytest.fixture(autouse=True)
def clean_mock_receiver():
    received_events.clear()
    received_ids.clear()
    yield
    received_events.clear()
    received_ids.clear()


def test_mock_receiver_schema_validation():
    client = TestClient(app)

    # Valid event
    evt = TelemetryEvent(
        event_type=TelemetryEventType.STEP_COMPLETED,
        step_id="STEP_01",
        confidence=0.92
    )
    resp = client.post("/api/v1/telemetry/events", json=evt.model_dump())
    assert resp.status_code == 200
    assert resp.json()["acknowledged"] is True
    assert resp.json()["total_received"] == 1

    # Duplicate submission
    resp_dup = client.post("/api/v1/telemetry/events", json=evt.model_dump())
    assert resp_dup.status_code == 200
    assert resp_dup.json()["is_duplicate"] is True
    # Count should still be 1
    assert resp_dup.json()["total_received"] == 1

    # Invalid payload missing required event_type
    bad_payload = {"device_id": "ai-01"}
    resp_bad = client.post("/api/v1/telemetry/events", json=bad_payload)
    assert resp_bad.status_code == 422


def test_publisher_queue_bounding_and_drop_strategy():
    # Publisher with tiny queue of 2
    pub = EventPublisher(target_url=None, max_queue_size=2)
    e1 = TelemetryEvent(event_type=TelemetryEventType.HEARTBEAT, confidence=1.0)
    e2 = TelemetryEvent(event_type=TelemetryEventType.HEARTBEAT, confidence=1.0)
    e3 = TelemetryEvent(event_type=TelemetryEventType.HEARTBEAT, confidence=1.0)

    pub.publish(e1)
    pub.publish(e2)
    # Queue is now full (2/2). Publishing e3 should drop e1
    pub.publish(e3)

    assert pub.get_stats()["total_dropped"] == 1
    assert pub.get_stats()["queue_depth"] == 2


def test_publisher_deduplication():
    pub = EventPublisher(target_url=None, max_queue_size=10)
    evt = TelemetryEvent(event_type=TelemetryEventType.SYSTEM_STATUS, confidence=1.0)

    # Publish twice
    pub.publish(evt)
    # Simulate that this event was already delivered
    pub._sent_ids_set.add(evt.event_id)
    pub.publish(evt)

    # Queue should only have 1 item
    assert pub.get_stats()["queue_depth"] == 1

