"""
CPU-First Kinematic Hand Tracker for SUTRA Vision Engine.
Includes face suppression to prevent facial features from being mistaken for hands.
Extracts wrist, palm, fingertips and 21 anatomical keypoints along kinematic chains.
"""

import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from modules.pose.pose_base import BasePoseEstimator, HandPose, HandKeypoint, HAND_KEYPOINT_NAMES


class KinematicHandTracker(BasePoseEstimator):
    """
    CPU-optimized hand pose estimator with facial masking.
    """

    def __init__(self, min_hand_area_pct: float = 0.015, face_cascade_path: Optional[str] = None):
        self.min_hand_area_pct = min_hand_area_pct

        if face_cascade_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            face_cascade_path = str(base_dir / "models" / "haarcascade_frontalface_default.xml")

        self.face_cascade = None
        if os.path.isfile(face_cascade_path):
            try:
                self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            except Exception:
                self.face_cascade = None

    def is_operational(self) -> bool:
        return True

    def estimate(self, frame: np.ndarray) -> list[HandPose]:
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        min_area = w * h * self.min_hand_area_pct
        max_area = w * h * 0.40

        # Detect face to mask out
        face_boxes = []
        if self.face_cascade is not None and not self.face_cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))
            for (fx, fy, fw, fh) in faces:
                face_boxes.append((fx, fy, fw, fh))

        # Skin tone segmentation in YCrCb color space
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        # Mask out face region completely
        for (fx, fy, fw, fh) in face_boxes:
            pad_w = int(fw * 0.25)
            pad_h = int(fh * 0.35)
            cv2.rectangle(
                mask,
                (max(0, fx - pad_w), max(0, fy - pad_h)),
                (min(w, fx + fw + pad_w), min(h, fy + fh + int(pad_h * 1.5))),
                0, -1
            )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poses: list[HandPose] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bh / float(bw) if bw > 0 else 1.0
            if aspect > 3.5 or aspect < 0.25:
                continue

            x1, y1 = float(bx), float(by)
            x2, y2 = float(bx + bw), float(by + bh)

            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])

            hull = cv2.convexHull(c, returnPoints=True)
            wrist = (float(cx), float(y2))
            palm_center = (cx, cy)

            tips = []
            for pt in hull:
                px, py = float(pt[0][0]), float(pt[0][1])
                if py < cy:
                    tips.append((px, py))

            tips = sorted(tips, key=lambda p: p[0])
            if len(tips) >= 2:
                thumb_tip = tips[0]
                index_tip = tips[min(1, len(tips) - 1)]
            elif len(tips) == 1:
                index_tip = tips[0]
                thumb_tip = (cx - bw * 0.2, cy - bh * 0.1)
            else:
                index_tip = (cx, y1)
                thumb_tip = (cx - bw * 0.2, cy)

            keypoints: list[HandKeypoint] = []
            keypoints.append(HandKeypoint(id=0, name="WRIST", x=wrist[0], y=wrist[1]))

            finger_targets = [
                thumb_tip,
                index_tip,
                (cx, y1),
                (cx + bw * 0.15, y1 + bh * 0.05),
                (cx + bw * 0.3, cy - bh * 0.1)
            ]

            kp_id = 1
            for f_idx, tip in enumerate(finger_targets):
                base_x = cx + (tip[0] - cx) * 0.2
                base_y = cy + (tip[1] - cy) * 0.2
                for joint_ratio in [0.25, 0.50, 0.75, 1.0]:
                    jx = base_x + (tip[0] - base_x) * joint_ratio
                    jy = base_y + (tip[1] - base_y) * joint_ratio
                    name = HAND_KEYPOINT_NAMES[kp_id] if kp_id < len(HAND_KEYPOINT_NAMES) else f"KP_{kp_id}"
                    keypoints.append(HandKeypoint(
                        id=kp_id,
                        name=name,
                        x=float(jx),
                        y=float(jy),
                        z=0.0,
                        confidence=0.80
                    ))
                    kp_id += 1

            handedness = "Right" if thumb_tip[0] < index_tip[0] else "Left"

            poses.append(HandPose(
                handedness=handedness,
                confidence=0.82,
                bbox=[x1, y1, x2, y2],
                landmarks=keypoints,
                wrist=wrist,
                palm_center=palm_center,
                index_tip=index_tip,
                thumb_tip=thumb_tip
            ))

        return poses
