"""
SUTRA Vision Pipeline Orchestrator.
Integrates Ingestion -> Detection -> Pose -> Calibration -> Interaction ->
Temporal HAR -> Protocol FSM -> Telemetry Storage & Event Publisher.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np
import psutil

from app.config import SutraConfig
from modules.ingestion.frame_buffer import LatestFrameBuffer, IngestedFrame
from modules.ingestion.ingestor import FrameIngestor
from modules.detection.detector_base import BaseObjectDetector, DetectedObject
from modules.detection.onnx_detector import ONNXObjectDetector
from modules.detection.heuristic_detector import HeuristicLabDetector
from modules.pose.pose_base import BasePoseEstimator, HandPose
from modules.pose.mediapipe_pose import MediaPipePoseEstimator
from modules.pose.kinematic_pose import KinematicHandTracker
from modules.calibration.coordinate_normalizer import RackCoordinateNormalizer
from modules.tracking.tracker import CentroidTracker
from modules.interaction.interaction_engine import InteractionEngine, InteractionRelation
from modules.temporal.temporal_engine import DeterministicTemporalEngine, TemporalActivityState
from modules.protocol.fsm_engine import ProtocolEngine, StepStatus
from modules.telemetry.event_schema import TelemetryEvent, TelemetryEventType, RackCoordinates
from modules.telemetry.event_storage import EventStorage
from modules.communication.event_publisher import EventPublisher

logger = logging.getLogger("sutra.engine")


@dataclass
class PipelineResult:
    """Output summary of processing a single video frame."""
    frame_id: int
    timestamp: float
    latency_ms: float
    fps: float
    objects: list[DetectedObject] = field(default_factory=list)
    hands: list[HandPose] = field(default_factory=list)
    interactions: list[InteractionRelation] = field(default_factory=list)
    temporal_state: Optional[TemporalActivityState] = None
    current_step_id: Optional[str] = None
    current_step_name: Optional[str] = None
    step_status: Optional[str] = None
    deviation: Optional[dict] = None
    annotated_frame: Optional[np.ndarray] = None


class SutraVisionEngine:
    """
    Autonomous, independent AI Vision & Activity Recognition Engine for SUTRA.
    Operates strictly within an 8 GB RAM CPU-only environment.
    """

    def __init__(self, config: SutraConfig):
        self.config = config
        self._running = False
        
        # 1. Performance Profiling
        self._latencies: list[float] = []
        self._processed_frames = 0
        self._start_time = time.monotonic()

        # 2. Storage & Communication
        self.storage = EventStorage(config.storage_log_path)
        self.publisher = EventPublisher(
            target_url=config.publisher_url if config.enable_http_publisher else None
        )

        # 3. Calibration
        self.normalizer = RackCoordinateNormalizer(image_dims=(config.frame_width, config.frame_height))
        if os.path.isfile(config.calibration_path):
            self.normalizer.load_calibration(config.calibration_path)

        # 4. Perception Backends
        self.detector = self._initialize_detector()
        self.pose_estimator = self._initialize_pose()
        self.tracker = CentroidTracker()

        # 5. Spatial & Temporal Reasoning
        self.interaction_engine = InteractionEngine()
        self.temporal_engine = DeterministicTemporalEngine()
        self.protocol_engine = ProtocolEngine(config.protocol_path)

        # 6. Ingestion Buffer & Worker
        self.frame_buffer = LatestFrameBuffer(max_capacity=2)
        self.ingestor = self._initialize_ingestor()

    def _initialize_detector(self) -> BaseObjectDetector:
        onnx_det = ONNXObjectDetector(
            model_path=self.config.model_path,
            conf_threshold=self.config.detection_conf_threshold,
            input_size=(self.config.frame_width, self.config.frame_height)
        )
        if onnx_det.is_operational():
            logger.info(f"Using ONNX Runtime detector with model: {self.config.model_path}")
            return onnx_det
        else:
            logger.info("Using lightweight HeuristicLabDetector fallback for lab items.")
            return HeuristicLabDetector(min_confidence=self.config.detection_conf_threshold)

    def _initialize_pose(self) -> BasePoseEstimator:
        if self.config.pose_backend == "mediapipe":
            mp_pose = MediaPipePoseEstimator()
            if mp_pose.is_operational():
                logger.info("Using MediaPipe Hands pose estimator.")
                return mp_pose
        logger.info("Using CPU-first KinematicHandTracker pose estimator.")
        return KinematicHandTracker()

    def _initialize_ingestor(self) -> FrameIngestor:
        src = self.config.video_source
        if src.isdigit():
            src_val = int(src)
        else:
            src_val = src

        return FrameIngestor(
            source=src_val,
            buffer=self.frame_buffer,
            target_fps=self.config.target_fps,
            target_resolution=(self.config.frame_width, self.config.frame_height),
            loop_video=self.config.loop_video,
            on_disconnect=self._on_camera_disconnect,
            on_reconnect=self._on_camera_reconnect
        )

    def _on_camera_disconnect(self) -> None:
        logger.warning("Event dispatched: CAMERA_DISCONNECTED")
        # Save protocol checkpoint to preserve mission state
        self.protocol_engine.save_checkpoint()
        evt = TelemetryEvent(
            device_id=self.config.device_id,
            experiment_id=self.config.experiment_id,
            event_type=TelemetryEventType.CAMERA_DISCONNECTED,
            confidence=1.0,
            evidence={"reason": "Video stream dropped or cable disconnected"}
        )
        self.storage.record_event(evt)
        self.publisher.publish(evt)

    def _on_camera_reconnect(self) -> None:
        logger.info("Event dispatched: CAMERA_RECONNECTED")
        # Restore protocol checkpoint
        self.protocol_engine.restore_checkpoint()
        evt = TelemetryEvent(
            device_id=self.config.device_id,
            experiment_id=self.config.experiment_id,
            event_type=TelemetryEventType.CAMERA_RECONNECTED,
            confidence=1.0,
            evidence={"status": "Stream resumed, state checkpoint recovered"}
        )
        self.storage.record_event(evt)
        self.publisher.publish(evt)

    def start(self) -> None:
        """Start ingestion and communication workers."""
        if self._running:
            return
        self._running = True
        self.publisher.start()
        self.ingestor.start()
        logger.info("SUTRA Vision Engine started.")

        # Emit initial system status event
        init_evt = TelemetryEvent(
            device_id=self.config.device_id,
            experiment_id=self.config.experiment_id,
            event_type=TelemetryEventType.SYSTEM_STATUS,
            confidence=1.0,
            evidence={
                "status": "INITIALIZED",
                "detector": type(self.detector).__name__,
                "pose": type(self.pose_estimator).__name__,
                "is_calibrated": self.normalizer.is_calibrated
            }
        )
        self.storage.record_event(init_evt)
        self.publisher.publish(init_evt)

    def stop(self) -> None:
        """Gracefully terminate engine and export final telemetry."""
        self._running = False
        self.ingestor.stop()
        self.publisher.stop()
        self.storage.export_to_json(self.config.storage_export_path)
        logger.info(f"SUTRA Vision Engine stopped. Telemetry exported to {self.config.storage_export_path}")

    def process_single_frame(self, ingested_frame: IngestedFrame, annotate: bool = True) -> PipelineResult:
        """
        Execute full vision and activity recognition pipeline on a single frame.
        """
        t0 = time.perf_counter()
        frame = ingested_frame.frame
        ts = ingested_frame.timestamp_wall
        f_id = ingested_frame.frame_id

        # 1. Perception Layer (Detection & Pose)
        raw_detections = self.detector.detect(frame, ts)
        tracked_objects = self.tracker.update(raw_detections, ts)
        hand_poses = self.pose_estimator.estimate(frame)

        # 2. Hand-Object Interaction (HOI)
        interactions = self.interaction_engine.evaluate(
            hands=hand_poses,
            objects=tracked_objects,
            normalizer=self.normalizer,
            timestamp=ts
        )

        # 3. Temporal Sequence Reasoning
        temporal_state = self.temporal_engine.push_frame(
            objects=tracked_objects,
            hands=hand_poses,
            interactions=interactions,
            timestamp=ts
        )

        # 4. Protocol DAG State Machine
        trigger_type, payload = self.protocol_engine.evaluate_step(
            temporal_state=temporal_state,
            interactions=interactions,
            objects=tracked_objects,
            timestamp=ts
        )

        # 5. Dispatch Telemetry Events on State Changes
        cur_step = None
        if self.protocol_engine.current_step_idx < len(self.protocol_engine.protocol.steps):
            cur_step = self.protocol_engine.protocol.steps[self.protocol_engine.current_step_idx]

        if trigger_type and payload:
            event_type_map = {
                "STEP_STARTED": TelemetryEventType.STEP_STARTED,
                "STEP_COMPLETED": TelemetryEventType.STEP_COMPLETED,
                "DEVIATION_DETECTED": TelemetryEventType.DEVIATION_DETECTED,
                "NEEDS_REVIEW": TelemetryEventType.NEEDS_REVIEW
            }
            evt_type = event_type_map.get(trigger_type, TelemetryEventType.SYSTEM_STATUS)

            # Extract rack coordinates of primary interaction if present
            norm_coord = None
            if interactions:
                norm_coord = RackCoordinates(
                    x=self.normalizer.normalize_point(tracked_objects[0].center[0], tracked_objects[0].center[1]).x if tracked_objects else 0.5,
                    y=self.normalizer.normalize_point(tracked_objects[0].center[0], tracked_objects[0].center[1]).y if tracked_objects else 0.5,
                    calibration_status="CALIBRATED" if self.normalizer.is_calibrated else "UNAVAILABLE"
                )

            evt = TelemetryEvent(
                device_id=self.config.device_id,
                experiment_id=self.config.experiment_id,
                event_type=evt_type,
                step_id=payload.get("step_id", cur_step.id if cur_step else None),
                step_name=payload.get("step_name", cur_step.name if cur_step else None),
                status=trigger_type,
                confidence=payload.get("confidence", temporal_state.confidence),
                evidence=payload.get("evidence", {}),
                rack_coordinates=norm_coord,
                requires_review=(evt_type in [TelemetryEventType.DEVIATION_DETECTED, TelemetryEventType.NEEDS_REVIEW])
            )
            self.storage.record_event(evt)
            self.publisher.publish(evt)

        # 6. Profiling & Measurement
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._latencies.append(elapsed_ms)
        if len(self._latencies) > 500:
            self._latencies.pop(0)
        self._processed_frames += 1

        # 7. Visual HUD Annotation
        annotated = None
        if annotate:
            annotated = self._render_hud(
                frame.copy(),
                tracked_objects,
                hand_poses,
                interactions,
                temporal_state,
                cur_step,
                elapsed_ms
            )

        return PipelineResult(
            frame_id=f_id,
            timestamp=ts,
            latency_ms=round(elapsed_ms, 2),
            fps=round(ingested_frame.fps_measured, 1),
            objects=tracked_objects,
            hands=hand_poses,
            interactions=interactions,
            temporal_state=temporal_state,
            current_step_id=cur_step.id if cur_step else "DONE",
            current_step_name=cur_step.name if cur_step else "Protocol Finished",
            step_status=self.protocol_engine.step_statuses.get(cur_step.id, StepStatus.COMPLETED).value if cur_step else "COMPLETED",
            deviation=self.protocol_engine.active_deviation,
            annotated_frame=annotated
        )

    def _render_hud(
        self,
        frame: np.ndarray,
        objects: list[DetectedObject],
        hands: list[HandPose],
        interactions: list[InteractionRelation],
        temporal_state: TemporalActivityState,
        cur_step,
        latency_ms: float
    ) -> np.ndarray:
        """Render high-contrast visual HUD on frame for live inspection."""
        h, w = frame.shape[:2]

        # 1. Draw Object Boxes
        for obj in objects:
            b = obj.bbox
            cv2.rectangle(frame, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 255, 128), 2)
            lbl = f"{obj.class_name} ({obj.confidence:.2f})"
            cv2.putText(frame, lbl, (int(b.x1), max(18, int(b.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA)

        # 2. Draw Hand Landmarks
        for hand in hands:
            b = hand.bbox
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 160, 0), 1)
            cv2.circle(frame, (int(hand.palm_center[0]), int(hand.palm_center[1])), 4, (0, 200, 255), -1)
            cv2.circle(frame, (int(hand.index_tip[0]), int(hand.index_tip[1])), 4, (0, 0, 255), -1)
            cv2.circle(frame, (int(hand.thumb_tip[0]), int(hand.thumb_tip[1])), 4, (255, 0, 255), -1)

        # 3. Top Banner: Current Step & Progress
        cv2.rectangle(frame, (0, 0), (w, 54), (20, 20, 25), -1)
        step_text = f"STEP: {cur_step.id if cur_step else 'COMPLETED'} - {cur_step.name if cur_step else 'All Steps Verified'}"
        cv2.putText(frame, step_text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        
        act_text = f"Action: {temporal_state.confirmed_activity or 'WAITING'} (Dur: {temporal_state.duration_active:.1f}s)"
        if temporal_state.is_occluded:
            act_text += " [OCCLUSION BRIDGED]"
        cv2.putText(frame, act_text, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 255), 1, cv2.LINE_AA)

        # 4. Diagnostics (Latency, RAM)
        ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        diag_text = f"{latency_ms:.1f}ms | RAM: {ram_mb:.0f}MB"
        cv2.putText(frame, diag_text, (w - 180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # 5. Deviation Banner (Red alert overlay if deviation active)
        if self.protocol_engine.active_deviation:
            cv2.rectangle(frame, (0, h - 36), (w, h), (0, 0, 180), -1)
            dev_text = f"ALERT: {self.protocol_engine.active_deviation.get('description', 'Deviation Detected')}"
            cv2.putText(frame, dev_text, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def get_diagnostics(self) -> dict:
        """Compute real, measured benchmark and runtime metrics."""
        ram_mb = round(psutil.Process().memory_info().rss / (1024.0 * 1024.0), 1)
        lats = self._latencies
        p50 = round(float(np.percentile(lats, 50)), 2) if lats else 0.0
        p95 = round(float(np.percentile(lats, 95)), 2) if lats else 0.0
        p99 = round(float(np.percentile(lats, 99)), 2) if lats else 0.0
        mean_lat = round(float(np.mean(lats)), 2) if lats else 0.0

        buf_stats = self.frame_buffer.get_stats()
        storage_stats = self.storage.get_stats()
        pub_stats = self.publisher.get_stats()

        return {
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "processed_frames": self._processed_frames,
            "memory_rss_mb": ram_mb,
            "latency_mean_ms": mean_lat,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
            "buffer": buf_stats,
            "telemetry": storage_stats,
            "publisher": pub_stats
        }

