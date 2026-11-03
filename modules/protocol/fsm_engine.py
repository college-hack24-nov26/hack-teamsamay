"""
Protocol Directed Acyclic Graph (DAG) and Finite State Machine Engine for SUTRA.
Enforces multi-step scientific protocol compliance, detects deviations (skipped steps,
out-of-order steps, incorrect tools), and prevents false completions without temporal proof.
"""

import logging
from enum import Enum
from typing import Callable, Optional
import yaml
from pydantic import BaseModel, Field

from modules.temporal.temporal_engine import TemporalActivityState
from modules.interaction.interaction_engine import InteractionRelation
from modules.detection.detector_base import DetectedObject

logger = logging.getLogger("sutra.protocol")


class StepStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    DEVIATED = "DEVIATED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ProtocolStepConfig(BaseModel):
    id: str
    index: int
    name: str
    description: str
    required_activities: list[str] = Field(default_factory=list)
    required_objects: list[str] = Field(default_factory=list)
    forbidden_objects: list[str] = Field(default_factory=list)
    required_interactions: list[dict] = Field(default_factory=list)
    min_duration_seconds: float = 2.0
    max_duration_seconds: float = 60.0
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)


class ProtocolDefinition(BaseModel):
    protocol_id: str
    protocol_name: str
    version: str
    description: str
    steps: list[ProtocolStepConfig]


class ProtocolEngine:
    """
    State Machine evaluating temporal action progression against protocol rules.
    """

    def __init__(self, protocol_path: str):
        self.protocol_path = protocol_path
        self.protocol = self._load_protocol(protocol_path)
        
        # Runtime State
        self.current_step_idx = 0
        self.step_statuses: dict[str, StepStatus] = {s.id: StepStatus.PENDING for s in self.protocol.steps}
        self.step_start_times: dict[str, float] = {}
        self.step_completed_times: dict[str, float] = {}
        
        self.active_deviation: Optional[dict] = None
        self.is_protocol_completed = False
        
        # Saved state checkpoint for camera recovery
        self._checkpoint: Optional[dict] = None

    def _load_protocol(self, path: str) -> ProtocolDefinition:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return ProtocolDefinition(**data)

    def reset(self) -> None:
        """Reset state machine to initial unstarted state."""
        self.current_step_idx = 0
        self.step_statuses = {s.id: StepStatus.PENDING for s in self.protocol.steps}
        self.step_start_times.clear()
        self.step_completed_times.clear()
        self.active_deviation = None
        self.is_protocol_completed = False
        self._checkpoint = None
        logger.info(f"Protocol '{self.protocol.protocol_id}' reset.")

    def save_checkpoint(self) -> dict:
        """Serialize current state to recover after camera disconnect."""
        self._checkpoint = {
            "current_step_idx": self.current_step_idx,
            "step_statuses": {k: v.value for k, v in self.step_statuses.items()},
            "step_start_times": dict(self.step_start_times),
            "step_completed_times": dict(self.step_completed_times),
            "is_protocol_completed": self.is_protocol_completed
        }
        return self._checkpoint

    def restore_checkpoint(self, checkpoint: Optional[dict] = None) -> bool:
        """Restore state from checkpoint following camera re-acquisition."""
        state = checkpoint or self._checkpoint
        if not state:
            return False
        self.current_step_idx = state["current_step_idx"]
        self.step_statuses = {k: StepStatus(v) for k, v in state["step_statuses"].items()}
        self.step_start_times = dict(state["step_start_times"])
        self.step_completed_times = dict(state["step_completed_times"])
        self.is_protocol_completed = state["is_protocol_completed"]
        logger.info(f"Protocol state restored to step index {self.current_step_idx}.")
        return True

    def acknowledge_deviation(self, step_id: Optional[str] = None) -> None:
        """Manual operator override/acknowledgement of a triggered deviation."""
        if self.active_deviation:
            logger.info(f"Deviation acknowledged by operator: {self.active_deviation}")
            self.active_deviation = None

    def evaluate_step(
        self,
        temporal_state: TemporalActivityState,
        interactions: list[InteractionRelation],
        objects: list[DetectedObject],
        timestamp: float
    ) -> tuple[Optional[str], Optional[dict]]:
        """
        Evaluate current perception features against the active protocol step.
        Returns: (event_trigger_type, event_payload)
        Possible trigger types:
          - 'STEP_STARTED'
          - 'STEP_COMPLETED'
          - 'DEVIATION_DETECTED'
          - 'NEEDS_REVIEW'
          - None (no transition in this frame)
        """
        if self.is_protocol_completed or self.current_step_idx >= len(self.protocol.steps):
            return None, None

        current_step = self.protocol.steps[self.current_step_idx]
        current_status = self.step_statuses[current_step.id]
        detected_classes = {o.class_name for o in objects}
        active_activity = temporal_state.confirmed_activity

        # 1. Check for Out-of-Order / Skipped Step Attempt
        if active_activity is not None:
            for future_idx in range(self.current_step_idx + 1, len(self.protocol.steps)):
                future_step = self.protocol.steps[future_idx]
                if active_activity in future_step.required_activities:
                    # An action corresponding to a future step was executed before current step completed!
                    deviation = {
                        "deviation_type": "SKIPPED_STEP",
                        "attempted_step": future_step.id,
                        "uncompleted_step": current_step.id,
                        "description": f"Attempted '{future_step.name}' while '{current_step.name}' was not completed.",
                        "confidence": temporal_state.confidence,
                        "evidence": {
                            "active_activity": active_activity,
                            "missing_preconditions": [current_step.id]
                        }
                    }
                    self.active_deviation = deviation
                    self.step_statuses[current_step.id] = StepStatus.DEVIATED
                    return "DEVIATION_DETECTED", deviation

        # 2. Check for Incorrect Tool Usage
        if current_status == StepStatus.IN_PROGRESS:
            for forbidden in current_step.forbidden_objects:
                if forbidden in detected_classes:
                    deviation = {
                        "deviation_type": "INCORRECT_TOOL",
                        "step_id": current_step.id,
                        "description": f"Forbidden object '{forbidden}' detected during '{current_step.name}'.",
                        "confidence": 0.85,
                        "evidence": {"forbidden_tool": forbidden}
                    }
                    self.active_deviation = deviation
                    return "DEVIATION_DETECTED", deviation

        # 3. Step Progression Evaluation
        is_step_activity = active_activity in current_step.required_activities if active_activity else False

        # Transition: PENDING -> IN_PROGRESS
        if current_status == StepStatus.PENDING:
            if is_step_activity:
                # Check prerequisites
                for pre in current_step.preconditions:
                    if self.step_statuses.get(pre) != StepStatus.COMPLETED:
                        deviation = {
                            "deviation_type": "PRECONDITION_UNMET",
                            "step_id": current_step.id,
                            "description": f"Precondition '{pre}' not completed for step '{current_step.name}'.",
                            "confidence": 0.90,
                            "evidence": {"precondition": pre}
                        }
                        self.active_deviation = deviation
                        return "DEVIATION_DETECTED", deviation

                self.step_statuses[current_step.id] = StepStatus.IN_PROGRESS
                self.step_start_times[current_step.id] = timestamp
                logger.info(f"Step {current_step.id} ({current_step.name}) started.")
                return "STEP_STARTED", {
                    "step_id": current_step.id,
                    "step_name": current_step.name,
                    "confidence": temporal_state.confidence,
                    "start_time": timestamp
                }

        # Transition: IN_PROGRESS -> COMPLETED (or NEEDS_REVIEW)
        elif current_status == StepStatus.IN_PROGRESS:
            start_t = self.step_start_times.get(current_step.id, timestamp)
            elapsed = timestamp - start_t

            # Verify required objects are present
            missing_objects = [req for req in current_step.required_objects if req not in detected_classes]
            
            # Check duration limits
            if elapsed > current_step.max_duration_seconds:
                # Dwell time exceeded expected maximum -> mark for review
                self.step_statuses[current_step.id] = StepStatus.NEEDS_REVIEW
                return "NEEDS_REVIEW", {
                    "step_id": current_step.id,
                    "reason": "MAX_DURATION_EXCEEDED",
                    "elapsed_seconds": round(elapsed, 2),
                    "max_seconds": current_step.max_duration_seconds
                }

            # To complete: must meet minimum duration AND have verified temporal consensus
            if is_step_activity and elapsed >= current_step.min_duration_seconds:
                # Check if interaction requirements are met (if specified)
                interactions_satisfied = True
                if current_step.required_interactions:
                    active_tuples = {(r.subject, r.verb, r.target) for r in interactions}
                    for req_int in current_step.required_interactions:
                        expected = (req_int["subject"], req_int["verb"], req_int["target"])
                        if expected not in active_tuples:
                            # Could be momentary occlusion; if missing_objects is empty, give slight grace
                            if missing_objects:
                                interactions_satisfied = False

                if interactions_satisfied and not missing_objects:
                    self.step_statuses[current_step.id] = StepStatus.COMPLETED
                    self.step_completed_times[current_step.id] = timestamp
                    logger.info(f"Step {current_step.id} ({current_step.name}) successfully COMPLETED in {elapsed:.1f}s.")
                    
                    completed_payload = {
                        "step_id": current_step.id,
                        "step_name": current_step.name,
                        "confidence": temporal_state.confidence,
                        "duration_seconds": round(elapsed, 2),
                        "evidence": {
                            "objects": list(detected_classes),
                            "duration_active": temporal_state.duration_active
                        }
                    }

                    # Advance to next step
                    self.current_step_idx += 1
                    if self.current_step_idx >= len(self.protocol.steps):
                        self.is_protocol_completed = True
                        logger.info("All protocol steps COMPLETED successfully.")

                    self.save_checkpoint()
                    return "STEP_COMPLETED", completed_payload

        return None, None
