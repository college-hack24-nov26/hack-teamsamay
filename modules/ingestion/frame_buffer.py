"""
Thread-safe, zero-lag bounded frame buffer for real-time video ingestion.
Guarantees that stale frames are discarded when the perception pipeline
is slower than the camera ingest rate.
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class IngestedFrame:
    """Container for ingested video frames with full temporal metadata."""
    frame_id: int
    timestamp_wall: float      # Unix epoch timestamp (seconds)
    timestamp_mono: float      # Monotonic clock timestamp (seconds)
    frame: np.ndarray          # BGR image array
    source_type: str           # 'webcam', 'rtsp', 'video_file', 'synthetic'
    fps_measured: float = 0.0


class LatestFrameBuffer:
    """
    Bounded, thread-safe frame buffer implementing a drop-oldest strategy.
    
    Prevents camera buffer bloat and lag accumulation. When full, incoming frames
    immediately overwrite the oldest unprocessed frame.
    """

    def __init__(self, max_capacity: int = 2):
        if max_capacity < 1:
            raise ValueError("max_capacity must be at least 1")
        self.max_capacity = max_capacity
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        
        self._buffer: list[IngestedFrame] = []
        self._total_ingested: int = 0
        self._total_dropped: int = 0
        self._last_ingest_time: float = 0.0

    def put(self, frame: np.ndarray, timestamp_wall: float, timestamp_mono: float, 
            source_type: str, frame_id: int, fps_measured: float = 0.0) -> bool:
        """
        Store a new frame into the buffer.
        If capacity is reached, drop the oldest frame.
        Returns True if a frame was dropped.
        """
        dropped = False
        with self._lock:
            if len(self._buffer) >= self.max_capacity:
                self._buffer.pop(0)
                self._total_dropped += 1
                dropped = True
            
            ingested = IngestedFrame(
                frame_id=frame_id,
                timestamp_wall=timestamp_wall,
                timestamp_mono=timestamp_mono,
                frame=frame,
                source_type=source_type,
                fps_measured=fps_measured
            )
            self._buffer.append(ingested)
            self._total_ingested += 1
            self._last_ingest_time = timestamp_mono
            self._not_empty.notify_all()
        return dropped

    def get_latest(self, timeout: float = 0.5) -> Optional[IngestedFrame]:
        """
        Retrieve and remove the freshest frame from the buffer.
        Blocks for up to `timeout` seconds if empty.
        Returns None if buffer remains empty after timeout.
        """
        with self._not_empty:
            if not self._buffer:
                if not self._not_empty.wait(timeout=timeout):
                    return None
            if self._buffer:
                # Return the newest frame and clear buffer to guarantee zero lag
                latest = self._buffer.pop(-1)
                self._buffer.clear()
                return latest
            return None

    def peek_latest(self) -> Optional[IngestedFrame]:
        """Inspect the most recent frame without removing it from buffer."""
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
            return None

    def clear(self) -> None:
        """Clear all pending frames and reset buffer."""
        with self._lock:
            self._buffer.clear()

    def get_stats(self) -> dict:
        """Return drop-rate and ingest statistics."""
        with self._lock:
            total = self._total_ingested
            dropped = self._total_dropped
            rate = (dropped / total * 100.0) if total > 0 else 0.0
            return {
                "total_ingested": total,
                "total_dropped": dropped,
                "drop_rate_pct": round(rate, 2),
                "current_depth": len(self._buffer),
                "max_capacity": self.max_capacity
            }

