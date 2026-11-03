"""
Lightweight 2D Centroid & IoU Object Tracker for SUTRA Vision Engine.
Provides consistent tracking IDs and velocity estimation without heavy Kalman filters.
"""

from typing import Optional
import numpy as np

from modules.detection.detector_base import DetectedObject, BoundingBox


class TrackedEntity:
    """Internal state for an actively tracked object."""

    def __init__(self, track_id: int, detected_obj: DetectedObject):
        self.track_id = track_id
        self.class_name = detected_obj.class_name
        self.bbox = detected_obj.bbox
        self.center = detected_obj.center
        self.confidence = detected_obj.confidence
        self.last_seen_timestamp = detected_obj.timestamp
        self.frames_since_seen = 0
        self.total_frames_tracked = 1
        self.velocity = (0.0, 0.0)  # (dx/dt, dy/dt) in pixels/sec

    def update(self, detected_obj: DetectedObject):
        dt = detected_obj.timestamp - self.last_seen_timestamp
        if dt > 0.001:
            vx = (detected_obj.center[0] - self.center[0]) / dt
            vy = (detected_obj.center[1] - self.center[1]) / dt
            self.velocity = (vx, vy)

        self.bbox = detected_obj.bbox
        self.center = detected_obj.center
        self.confidence = detected_obj.confidence
        self.last_seen_timestamp = detected_obj.timestamp
        self.frames_since_seen = 0
        self.total_frames_tracked += 1


class CentroidTracker:
    """
    Lightweight tracker matching incoming detections to active tracks
    based on Euclidean distance and BoundingBox IoU.
    """

    def __init__(self, max_distance: float = 75.0, max_lost_frames: int = 15):
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames
        self._next_id = 1
        self._tracks: dict[int, TrackedEntity] = {}

    def update(self, detections: list[DetectedObject], timestamp: float) -> list[DetectedObject]:
        """
        Match detections to existing tracks and assign tracking IDs.
        Returns the updated detections list with tracking_id populated.
        """
        if not detections:
            # Increment lost counter for all tracks
            for track in list(self._tracks.values()):
                track.frames_since_seen += 1
                if track.frames_since_seen > self.max_lost_frames:
                    del self._tracks[track.track_id]
            return []

        active_tracks = list(self._tracks.values())
        if not active_tracks:
            # Register all incoming detections as new tracks
            for det in detections:
                tid = self._next_id
                self._next_id += 1
                det.tracking_id = tid
                self._tracks[tid] = TrackedEntity(tid, det)
            return detections

        # Compute cost matrix: Euclidean distance + IoU penalty
        num_dets = len(detections)
        num_tracks = len(active_tracks)
        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)

        for i, track in enumerate(active_tracks):
            for j, det in enumerate(detections):
                # Class mismatch gets heavy penalty
                if track.class_name != det.class_name:
                    cost_matrix[i, j] = 9999.0
                    continue

                dist = np.hypot(track.center[0] - det.center[0], track.center[1] - det.center[1])
                iou = track.bbox.iou(det.bbox)
                cost = dist - (iou * 40.0)
                cost_matrix[i, j] = cost

        # Greedy association
        matched_tracks = set()
        matched_dets = set()

        for _ in range(min(num_tracks, num_dets)):
            min_val = np.min(cost_matrix)
            if min_val > self.max_distance:
                break
            i, j = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            if i in matched_tracks or j in matched_dets:
                cost_matrix[i, j] = 99999.0
                continue

            track = active_tracks[i]
            det = detections[j]
            track.update(det)
            det.tracking_id = track.track_id

            matched_tracks.add(i)
            matched_dets.add(j)
            cost_matrix[i, :] = 99999.0
            cost_matrix[:, j] = 99999.0

        # Unmatched tracks
        for i, track in enumerate(active_tracks):
            if i not in matched_tracks:
                track.frames_since_seen += 1
                if track.frames_since_seen > self.max_lost_frames:
                    del self._tracks[track.track_id]

        # Unmatched detections -> create new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = self._next_id
                self._next_id += 1
                det.tracking_id = tid
                self._tracks[tid] = TrackedEntity(tid, det)

        return detections
