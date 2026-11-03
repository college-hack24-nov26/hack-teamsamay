"""
Mock Telemetry Receiver Server for Testing SUTRA AI Engine Publisher.
Emulates the Laptop 2 backend ingestion endpoint for offline testing.
"""

import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import ValidationError

from modules.telemetry.event_schema import TelemetryEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sutra.mock_receiver")

app = FastAPI(
    title="SUTRA Mock Backend Telemetry Receiver",
    description="Stand-in test harness for Laptop 2 event ingestion.",
    version="1.0"
)

# In-memory event ledger
received_events: list[dict] = []
received_ids: set[str] = set()


@app.get("/")
def root():
    return {
        "status": "ONLINE",
        "service": "SUTRA Mock Telemetry Receiver",
        "total_events_received": len(received_events)
    }


@app.post("/api/v1/telemetry/events")
async def receive_event(request: Request):
    """Ingest, validate, and store outbound telemetry events."""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")

    try:
        # Validate against schema
        event = TelemetryEvent(**body)
    except ValidationError as ve:
        logger.error(f"Schema validation error: {ve}")
        raise HTTPException(status_code=422, detail=ve.errors())

    # Check for duplicate
    is_duplicate = event.event_id in received_ids
    if not is_duplicate:
        received_ids.add(event.event_id)
        received_events.append(event.model_dump())
        logger.info(f"Received [{event.event_type.value}] Step: {event.step_id} (ID: {event.event_id})")
    else:
        logger.warning(f"Duplicate event ignored: {event.event_id}")

    return JSONResponse(
        status_code=200,
        content={
            "acknowledged": True,
            "event_id": event.event_id,
            "is_duplicate": is_duplicate,
            "total_received": len(received_events)
        }
    )


@app.get("/api/v1/telemetry/events")
def get_all_events(limit: Optional[int] = 50):
    """Retrieve stored events for test assertions."""
    return {
        "count": len(received_events),
        "events": received_events[-limit:] if limit else received_events
    }


@app.post("/api/v1/telemetry/reset")
def reset_ledger():
    """Clear memory buffer between test runs."""
    received_events.clear()
    received_ids.clear()
    return {"status": "RESET", "total_events": 0}


def run_mock_receiver(host: str = "127.0.0.1", port: int = 8088):
    """Run server programmatically."""
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_mock_receiver()

