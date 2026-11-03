"""
Standard Telemetry Event Contract Schema (v1.0) for SUTRA Vision Engine.
Provides strict Pydantic validation for all outbound AI status, step verification,
and deviation events emitted by this node.
"""

from enum import Enum
from typing import Any, Optional
import uuid
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class TelemetryEventType(str, Enum):
    HEARTBEAT = "HEARTBEAT"
    STEP_STARTED = "STEP_STARTED"
    STEP_COMPLETED = "STEP_COMPLETED"
    DEVIATION_DETECTED = "DEVIATION_DETECTED"
    CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
    CAMERA_RECONNECTED = "CAMERA_RECONNECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    SYSTEM_STATUS = "SYSTEM_STATUS"


class RackCoordinates(BaseModel):
    """Normalized rack position [0.0, 1.0] where the event or interaction occurred."""
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized horizontal coordinate")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized vertical coordinate")
    calibration_status: str = Field("CALIBRATED", description="'CALIBRATED' or 'UNAVAILABLE'")


class TelemetryEvent(BaseModel):
    """
    Canonical JSON Event Payload for SUTRA telemetry downlinks and IPC.
    Version: 1.0
    """
    schema_version: str = Field("1.0", description="Contract schema version")
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Globally unique event UUID")
    source: str = Field("sutra-ai-engine", description="Originating node identifier")
    device_id: str = Field("ai-device-01", description="Physical machine or payload rack identifier")
    experiment_id: str = Field("DEMO-WETLAB-01", description="Active scientific experiment protocol ID")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp"
    )
    event_type: TelemetryEventType = Field(..., description="Categorical classification of the event")
    step_id: Optional[str] = Field(None, description="Active protocol step ID (e.g. 'STEP_03')")
    step_name: Optional[str] = Field(None, description="Human-readable protocol step name")
    status: Optional[str] = Field(None, description="'STARTED', 'COMPLETED', 'DEVIATED', 'NEEDS_REVIEW'")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="AI confidence score [0.0, 1.0]")
    evidence: dict[str, Any] = Field(default_factory=dict, description="Supporting visual/temporal evidence")
    rack_coordinates: Optional[RackCoordinates] = Field(
        None,
        description="Fiducial-normalized payload rack coordinates"
    )
    requires_review: bool = Field(False, description="Flag indicating human-in-the-loop review is required")

    def to_json_str(self) -> str:
        """Serialize to compact JSON string."""
        return self.model_dump_json()
