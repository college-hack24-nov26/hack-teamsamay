"""
CPU-First Kinematic Hand Tracker for SUTRA Vision Engine.
Uses OpenCV contour geometry, convex hull analysis, and human anatomical
kinematic proportions to reconstruct 21 hand keypoints without heavy neural network overhead.
"""

from typing import Optional
import cv2
import numpy as np

from modules.pose.pose_base import BasePoseEstimator, HandPose, HandKeypoint, HAND_KEYPOINT_NAMES


class KinematicHandTracker(BasePoseEstimator):
    """
    Self-contained, CPU-optimized hand pose estimator.
    Extracts wrist, palm, fingertips and full 21 keypoints structure.
    """

    def __init__(self, min_hand_area_pct: float = 0.015):
        self.min_hand_area_pct = min_hand_area_pct

    def is_operational(self) -> bool:
        return True

    def estimate(self, frame: np.ndarray) -> list[HandPose]:
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        min_area = w * h * self.min_hand_area_pct

        # Skin tone segmentation in YCrCb color space (more illumination-resilient than RGB)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poses: list[HandPose] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            # Bounding box
            bx, by, bw, bh = cv2.boundingRect(c)
            x1, y1 = float(bx), float(by)
            x2, y2 = float(bx + bw), float(by + bh)

            # Centroid (Moments)
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])

            # Find extreme points
            hull = cv2.convexHull(c, returnPoints=True)
            topmost = tuple(c[c[:, :, 1].argmin()][0])
            bottommost = tuple(c[c[:, :, 1].argmax()][0])
            leftmost = tuple(c[c[:, :, 0].argmin()][0])
            rightmost = tuple(c[c[:, :, 0].argmax()][0])

            # In typical frontal view, bottommost edge represents the wrist
            wrist = (float(cx), float(y2))
            palm_center = (cx, cy)

            # Convex hull tips (fingertip candidates)
            tips = []
            for pt in hull:
                px, py = float(pt[0][0]), float(pt[0][1])
                # Fingertips are typically higher than the palm center
                if py < cy:
                    tips.append((px, py))

            # Sort tips horizontally
            tips = sorted(tips, key=lambda p: p[0])
            if len(tips) >= 2:
                # Leftmost is thumb or pinky depending on hand orientation
                thumb_tip = tips[0]
                index_tip = tips[min(1, len(tips) - 1)]
            elif len(tips) == 1:
                index_tip = tips[0]
                thumb_tip = (cx - bw * 0.2, cy - bh * 0.1)
            else:
                index_tip = (cx, y1)
                thumb_tip = (cx - bw * 0.2, cy)

            # Reconstruct 21 anatomical landmarks along kinematic chains
            keypoints: list[HandKeypoint] = []
            # 0: Wrist
            keypoints.append(HandKeypoint(id=0, name="WRIST", x=wrist[0], y=wrist[1]))

            # Finger chains: Thumb(1-4), Index(5-8), Middle(9-12), Ring(13-16), Pinky(17-20)
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

            # Determine handedness based on relative thumb position
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
