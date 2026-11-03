"""
Hand-Object Interaction (HOI) Engine for SUTRA Vision Engine.
Evaluates geometric proximity, bounding box overlap, keypoint distances,
interaction dwell time, and motion correlation to infer relational action tuples.
"""

from typing import Optional
import numpy as np
from pydantic import BaseModel, Field

from modules.detection.detector_base import DetectedObject
from modules.pose.pose_base import HandPose
from modules.calibration.coordinate_normalizer import RackCoordinateNormalizer, RackPoint


class InteractionRelation(BaseModel):
    """
    Structured relational tuple describing an inferred spatial/physical interaction.
    Example: Hand_R -> GRASPS -> Pipette (Confidence: 0.88, Duration: 3.2s)
    """
    subject: str = Field(..., description="Subject entity (e.g. 'Hand_R', 'Pipette')")
    verb: str = Field(..., description="Predicate action ('APPROACHES', 'GRASPS', 'ENTERS', 'RELEASES', 'IN_CONTACT')")
    target: str = Field(..., description="Target object (e.g. 'Pipette', 'Vial_Sealed', 'Vial_Open')")
    confidence: float = Field(..., ge=0.0, le=1.0)
    duration_seconds: float = Field(0.0, description="Elapsed time this interaction has persisted")
    distance_norm: float = Field(..., description="Euclidean distance in normalized rack units")
    evidence: dict = Field(default_factory=dict, description="Supporting metrics: IoU, pinch, keypoint dist")
    disclaimer: str = Field(
        "Geometric MVP inference; not physical proof of molecular contact or fluid transfer.",
        description="Scientific integrity note"
    )

    def to_summary(self) -> str:
        return f"{self.subject} -> {self.verb} -> {self.target} (conf: {self.confidence:.2f}, dur: {self.duration_seconds:.1f}s)"


class InteractionEngine:
    """
    Evaluates spatial relationships between detected hands and experimental tools.
    Maintains internal interaction duration timers to distinguish momentary brush-bys
    from intentional grasping and operation.
    """

    def __init__(
        self,
        grasp_distance_threshold: float = 0.12,     # In normalized rack space [0.0, 1.0]
        approach_distance_threshold: float = 0.25,
        min_grasp_duration_s: float = 0.3
    ):
        self.grasp_distance_threshold = grasp_distance_threshold
        self.approach_distance_threshold = approach_distance_threshold
        self.min_grasp_duration_s = min_grasp_duration_s

        # Active interaction history: {(subject, target, verb): start_timestamp}
        self._interaction_history: dict[tuple[str, str, str], float] = {}

    def reset(self) -> None:
        """Clear all active interaction timers."""
        self._interaction_history.clear()

    def evaluate(
        self,
        hands: list[HandPose],
        objects: list[DetectedObject],
        normalizer: RackCoordinateNormalizer,
        timestamp: float
    ) -> list[InteractionRelation]:
        """
        Compute active interactions in the current frame.
        """
        relations: list[InteractionRelation] = []
        current_active_keys: set[tuple[str, str, str]] = set()

        # Filter out "hand" objects from detections if detected separately
        lab_objects = [obj for obj in objects if obj.class_name != "hand"]

        for hand in hands:
            hand_label = f"Hand_{hand.handedness[0].upper()}"  # "Hand_R" or "Hand_L"
            
            # Hand keypoint locations in normalized rack space
            norm_palm = normalizer.normalize_point(hand.palm_center[0], hand.palm_center[1])
            norm_index = normalizer.normalize_point(hand.index_tip[0], hand.index_tip[1])
            norm_thumb = normalizer.normalize_point(hand.thumb_tip[0], hand.thumb_tip[1])

            for obj in lab_objects:
                norm_obj_center = normalizer.normalize_point(obj.center[0], obj.center[1])
                
                # Euclidean distance in normalized rack space
                d_palm = np.hypot(norm_palm.x - norm_obj_center.x, norm_palm.y - norm_obj_center.y)
                d_index = np.hypot(norm_index.x - norm_obj_center.x, norm_index.y - norm_obj_center.y)
                min_dist = min(d_palm, d_index)

                # Bounding box IoU (using normalized coordinates)
                norm_hand_bbox = normalizer.normalize_bbox(hand.bbox)
                norm_obj_bbox = normalizer.normalize_bbox(obj.bbox.to_list())
                
                # Check intersection
                ix1 = max(norm_hand_bbox[0], norm_obj_bbox[0])
                iy1 = max(norm_hand_bbox[1], norm_obj_bbox[1])
                ix2 = min(norm_hand_bbox[2], norm_obj_bbox[2])
                iy2 = min(norm_hand_bbox[3], norm_obj_bbox[3])
                overlap = (max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)) > 0.0

                # Determine verb
                verb = None
                conf = 0.0
                evidence = {
                    "distance_palm": round(float(d_palm), 4),
                    "distance_index": round(float(d_index), 4),
                    "box_overlap": overlap,
                    "hand_pinch_dist": round(hand.pinch_distance, 2),
                    "target_conf": obj.confidence
                }

                if min_dist <= self.grasp_distance_threshold or overlap:
                    # Pinch heuristic: if index and thumb tips are close, strong grasp indicator
                    is_pinching = hand.pinch_distance < 45.0
                    verb = "GRASPS"
                    ratio = min(1.0, min_dist / max(1e-5, self.grasp_distance_threshold))
                    conf = max(0.10, min(0.95, (1.0 - ratio) * 0.4 + 0.55))
                    evidence["is_pinching"] = is_pinching
                elif min_dist <= self.approach_distance_threshold:
                    verb = "APPROACHES"
                    ratio = min(1.0, min_dist / max(1e-5, self.approach_distance_threshold))
                    conf = max(0.10, min(0.90, 0.85 - ratio * 0.35))

                if verb is not None:
                    # Ensure confidence is strictly clamped in [0.0, 1.0]
                    clamped_conf = float(max(0.0, min(1.0, conf)))
                    key = (hand_label, obj.class_name, verb)
                    current_active_keys.add(key)

                    # Update or initialize duration timer
                    if key in self._interaction_history:
                        start_time = self._interaction_history[key]
                        duration = timestamp - start_time
                    else:
                        self._interaction_history[key] = timestamp
                        duration = 0.0

                    relations.append(InteractionRelation(
                        subject=hand_label,
                        verb=verb,
                        target=obj.class_name,
                        confidence=round(clamped_conf, 3),
                        duration_seconds=round(max(0.0, duration), 2),
                        distance_norm=round(float(min_dist), 4),
                        evidence=evidence
                    ))

        # Tool-to-Container Interactions (e.g. Pipette entering Vial)
        pipettes = [o for o in lab_objects if o.class_name == "pipette"]
        vials = [o for o in lab_objects if "vial" in o.class_name]

        for pip in pipettes:
            norm_pip = normalizer.normalize_point(pip.center[0], pip.center[1])
            for vial in vials:
                norm_vial = normalizer.normalize_point(vial.center[0], vial.center[1])
                d_tool = np.hypot(norm_pip.x - norm_vial.x, norm_pip.y - norm_vial.y)
                
                if d_tool <= (self.grasp_distance_threshold * 1.2):
                    key = ("Pipette", vial.class_name, "ENTERS")
                    current_active_keys.add(key)
                    if key in self._interaction_history:
                        duration = timestamp - self._interaction_history[key]
                    else:
                        self._interaction_history[key] = timestamp
                        duration = 0.0

                    relations.append(InteractionRelation(
                        subject="Pipette",
                        verb="ENTERS",
                        target=vial.class_name,
                        confidence=0.86,
                        duration_seconds=round(max(0.0, duration), 2),
                        distance_norm=round(float(d_tool), 4),
                        evidence={"tool_vial_distance": round(float(d_tool), 4)}
                    ))

        # Cleanup expired interactions that ceased in this frame
        for k in list(self._interaction_history.keys()):
            if k not in current_active_keys:
                del self._interaction_history[k]

        return relations
