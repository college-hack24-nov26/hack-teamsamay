"""
MediaPipe Hands Estimator for SUTRA Vision Engine.
Dynamically checks for mediapipe package availability.
"""

import logging
from typing import Optional
import cv2
import numpy as np

from modules.pose.pose_base import BasePoseEstimator, HandPose, HandKeypoint, HAND_KEYPOINT_NAMES

logger = logging.getLogger("sutra.pose.mediapipe")


class MediaPipePoseEstimator(BasePoseEstimator):
    """
    Standard Google MediaPipe hand landmark pipeline.
    Extracts 21 3D coordinates per hand when installed.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self._is_operational = False
        self._hands_module = None
        self._hands = None

        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self._is_operational = True
            logger.info("MediaPipe Hands pipeline successfully initialized.")
        except ImportError:
            logger.info("MediaPipe is not installed. Will use CPU kinematic hand tracker fallback.")
            self._is_operational = False
        except Exception as e:
            logger.warning(f"MediaPipe initialization failed: {e}")
            self._is_operational = False

    def is_operational(self) -> bool:
        return self._is_operational

    def estimate(self, frame: np.ndarray) -> list[HandPose]:
        if not self._is_operational or self._hands is None or frame is None:
            return []

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return []

        poses: list[HandPose] = []
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            handedness = "Right"
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness = results.multi_handedness[idx].classification[0].label

            keypoints: list[HandKeypoint] = []
            xs, ys = [], []

            for k_id, lm in enumerate(hand_lms.landmark):
                px = lm.x * w
                py = lm.y * h
                pz = lm.z * w
                name = HAND_KEYPOINT_NAMES[k_id] if k_id < len(HAND_KEYPOINT_NAMES) else f"KP_{k_id}"
                
                keypoints.append(HandKeypoint(
                    id=k_id,
                    name=name,
                    x=float(px),
                    y=float(py),
                    z=float(pz),
                    confidence=float(getattr(lm, "visibility", 1.0)),
                    visibility=float(getattr(lm, "visibility", 1.0))
                ))
                xs.append(px)
                ys.append(py)

            x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
            x2, y2 = min(float(w), max(xs)), min(float(h), max(ys))

            wrist = (keypoints[0].x, keypoints[0].y)
            thumb_tip = (keypoints[4].x, keypoints[4].y)
            index_tip = (keypoints[8].x, keypoints[8].y)
            # Palm center approximated from MCP joints and wrist
            palm_cx = float(np.mean([keypoints[i].x for i in [0, 5, 9, 13, 17]]))
            palm_cy = float(np.mean([keypoints[i].y for i in [0, 5, 9, 13, 17]]))

            poses.append(HandPose(
                handedness=handedness,
                confidence=0.90,
                bbox=[x1, y1, x2, y2],
                landmarks=keypoints,
                wrist=wrist,
                palm_center=(palm_cx, palm_cy),
                index_tip=index_tip,
                thumb_tip=thumb_tip
            ))

        return poses
