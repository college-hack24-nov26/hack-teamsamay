"""
Temporal Human Activity Recognition and Sequence Smoothing for SUTRA.
Evaluates multi-frame feature windows, provides temporal debounce (hysteresis),
consecutive-frame consensus, and short-term occlusion bridging.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional
from pydantic import BaseModel, Field

from modules.interaction.interaction_engine import InteractionRelation
from modules.detection.detector_base import DetectedObject
from modules.pose.pose_base import HandPose


class FrameSnapshot(BaseModel):
    """Temporal feature vector captured at a discrete frame timestamp."""
    timestamp: float
    detected_classes: set[str]
    interactions: list[tuple[str, str, str]]  # (subject, verb, target)
    max_interaction_duration: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class TemporalActivityState(BaseModel):
    """Smoothed, verified temporal activity status."""
    candidate_activity: Optional[str] = None
    confirmed_activity: Optional[str] = None
    consecutive_consensus_count: int = 0
    start_timestamp: Optional[float] = None
    duration_active: float = 0.0
    is_occluded: bool = False
    confidence: float = 0.0


class BaseTemporalModel(ABC):
    """Abstract interface for future deep temporal models (1D-TCN / Video-Swin)."""

    @abstractmethod
    def predict_window(self, feature_sequence: list[FrameSnapshot]) -> tuple[str, float]:
        """Returns (predicted_action_label, confidence)."""
        pass

    @abstractmethod
    def is_operational(self) -> bool:
        pass


class DeterministicTemporalEngine:
    """
    CPU-first temporal sequence validator and debounce engine.
    Filters momentary flickering, bridges brief physical occlusions (up to max_occlusion_s),
    and enforces consecutive frame consensus before triggering an activity candidate.
    """

    def __init__(
        self,
        window_size: int = 30,          # Approx 2.5 - 3 seconds at 10 FPS
        consensus_threshold: int = 5,   # Minimum consecutive matching frames
        max_occlusion_seconds: float = 0.6
    ):
        self.window_size = window_size
        self.consensus_threshold = consensus_threshold
        self.max_occlusion_seconds = max_occlusion_seconds

        self._history: deque[FrameSnapshot] = deque(maxlen=window_size)
        self._current_candidate: Optional[str] = None
        self._consecutive_count: int = 0
        self._activity_start_time: Optional[float] = None
        self._last_valid_time: Optional[float] = None
        self._confirmed_activity: Optional[str] = None
        self._confirmed_start_time: Optional[float] = None

    def reset(self) -> None:
        """Clear temporal history and reset states."""
        self._history.clear()
        self._current_candidate = None
        self._consecutive_count = 0
        self._activity_start_time = None
        self._last_valid_time = None
        self._confirmed_activity = None
        self._confirmed_start_time = None

    def push_frame(
        self,
        objects: list[DetectedObject],
        hands: list[HandPose],
        interactions: list[InteractionRelation],
        timestamp: float
    ) -> TemporalActivityState:
        """
        Record a frame's spatial features and evaluate temporal consensus.
        """
        classes = {o.class_name for o in objects}
        if hands:
            classes.add("hand")

        inter_tuples = [(r.subject, r.verb, r.target) for r in interactions]
        max_dur = max([r.duration_seconds for r in interactions], default=0.0)

        snapshot = FrameSnapshot(
            timestamp=timestamp,
            detected_classes=classes,
            interactions=inter_tuples,
            max_interaction_duration=max_dur
        )
        self._history.append(snapshot)

        # 1. Infer Instantaneous Frame Activity from Interactions
        instant_action = self._infer_instantaneous_action(classes, inter_tuples)

        # 2. Occlusion Tolerance Check
        is_occluded = False
        if instant_action is None and self._confirmed_activity is not None and self._last_valid_time is not None:
            time_since_valid = timestamp - self._last_valid_time
            if time_since_valid <= self.max_occlusion_seconds:
                # Bridge short visual dropouts (astronaut's arm or body briefly occluding tool)
                instant_action = self._confirmed_activity
                is_occluded = True

        # 3. Consensus & Debounce Logic
        if instant_action is not None and not is_occluded:
            self._last_valid_time = timestamp

        if instant_action == self._current_candidate:
            self._consecutive_count += 1
        else:
            self._current_candidate = instant_action
            self._consecutive_count = 1
            self._activity_start_time = timestamp

        # 4. Confirmation Threshold
        if self._consecutive_count >= self.consensus_threshold and self._current_candidate is not None:
            if self._confirmed_activity != self._current_candidate:
                self._confirmed_activity = self._current_candidate
                self._confirmed_start_time = self._activity_start_time
        elif instant_action is None and not is_occluded and self._consecutive_count >= self.consensus_threshold:
            self._confirmed_activity = None
            self._confirmed_start_time = None

        duration_active = 0.0
        if self._confirmed_activity and self._confirmed_start_time:
            duration_active = max(0.0, timestamp - self._confirmed_start_time)

        # Compute smoothed confidence
        conf = 0.88 if self._confirmed_activity else 0.0
        if is_occluded:
            conf *= 0.75  # Degrade confidence during occlusion bridging

        return TemporalActivityState(
            candidate_activity=self._current_candidate,
            confirmed_activity=self._confirmed_activity,
            consecutive_consensus_count=self._consecutive_count,
            start_timestamp=self._confirmed_start_time,
            duration_active=round(duration_active, 2),
            is_occluded=is_occluded,
            confidence=round(conf, 2)
        )

    def _infer_instantaneous_action(
        self,
        classes: set[str],
        interactions: list[tuple[str, str, str]]
    ) -> Optional[str]:
        """
        Maps low-level relational interactions to candidate experiment activities.
        """
        # Step 3: Aspirating reagent
        if any(target in ["vial_open", "vial_sealed"] and verb == "ENTERS" for _, verb, target in interactions):
            return "ASPIRATING_REAGENT"
        if any(target == "pipette" and verb == "GRASPS" for _, verb, target in interactions):
            # If holding pipette near open vial
            if "vial_open" in classes:
                return "DISPENSING_LIQUID"
            return "HOLDING_PIPETTE"

        # Step 2: Unsealing vial
        if any("vial" in target and verb == "GRASPS" for _, verb, target in interactions):
            if "vial_open" in classes:
                return "SEALING_CONTAINER"
            return "UNSEALING_CONTAINER"

        # Step 1: Workspace inspection
        if "hand" in classes and not interactions:
            return "INSPECTING_WORKSPACE"

        return None
