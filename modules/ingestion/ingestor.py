"""
Dedicated Frame Ingestor supporting Webcams, RTSP/IP streams, MP4/AVI files,
and synthetic frame generators with automated reconnection and fault tolerance.
"""

import logging
import threading
import time
from typing import Callable, Optional, Union
import cv2
import numpy as np

from modules.ingestion.frame_buffer import LatestFrameBuffer, IngestedFrame

logger = logging.getLogger("sutra.ingestion")


class FrameIngestor:
    """
    Asynchronous video frame ingestor.
    Runs a worker thread that captures frames, calculates empirical ingestion FPS,
    detects signal loss, and attempts exponential-backoff reconnection.
    """

    def __init__(
        self,
        source: Union[int, str, Callable[[], Optional[np.ndarray]]],
        buffer: LatestFrameBuffer,
        target_fps: float = 30.0,
        target_resolution: tuple[int, int] = (640, 480),
        loop_video: bool = False,
        reconnect_interval_s: float = 1.0,
        max_reconnect_interval_s: float = 10.0,
        on_disconnect: Optional[Callable[[], None]] = None,
        on_reconnect: Optional[Callable[[], None]] = None,
    ):
        self.source = source
        self.buffer = buffer
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        self.loop_video = loop_video
        self.reconnect_interval_s = reconnect_interval_s
        self.max_reconnect_interval_s = max_reconnect_interval_s
        self.on_disconnect = on_disconnect
        self.on_reconnect = on_reconnect

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_connected = False
        self._frame_count = 0
        self._source_type = self._determine_source_type()
        self._lock = threading.Lock()
        
        # Test hook for simulating camera disconnections
        self._simulate_failure = False

    def _determine_source_type(self) -> str:
        if callable(self.source):
            return "synthetic"
        if isinstance(self.source, int):
            return "webcam"
        if isinstance(self.source, str):
            if self.source.startswith("rtsp://") or self.source.startswith("http://"):
                return "rtsp"
            return "video_file"
        return "unknown"

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._is_connected

    @property
    def source_type(self) -> str:
        return self._source_type

    def start(self) -> None:
        """Start the background ingestion thread."""
        if self._running:
            logger.warning("FrameIngestor already running.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="IngestorWorker", daemon=True)
        self._thread.start()
        logger.info(f"FrameIngestor started for source: {self.source} ({self._source_type})")

    def stop(self) -> None:
        """Gracefully stop ingestion thread and release hardware resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_capture()
        with self._lock:
            self._is_connected = False
        logger.info("FrameIngestor stopped.")

    def switch_source(self, new_source: Union[int, str, Callable[[], Optional[np.ndarray]]]) -> None:
        """Switch video source at runtime without stopping background pipeline."""
        with self._lock:
            self.source = new_source
            self._source_type = self._determine_source_type()
            self._is_connected = False
        self._release_capture()
        logger.info(f"FrameIngestor video source switched to: {new_source} ({self._source_type})")

    def simulate_hardware_disconnect(self, simulate: bool = True) -> None:
        """Test hook to simulate camera drops or signal loss."""
        self._simulate_failure = simulate
        if simulate:
            logger.warning("Simulated hardware disconnect triggered.")

    def _open_capture(self) -> bool:
        if callable(self.source):
            with self._lock:
                self._is_connected = True
            return True

        self._release_capture()
        try:
            if isinstance(self.source, int) and hasattr(cv2, 'CAP_DSHOW'):
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(self.source)

            if not cap.isOpened():
                return False

            if self._source_type == "webcam":
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
                cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            self._cap = cap
            with self._lock:
                self._is_connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to open video source {self.source}: {e}")
            return False

    def _release_capture(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _capture_loop(self) -> None:
        backoff = self.reconnect_interval_s
        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.033
        last_frame_time = time.monotonic()
        fps_tracker = []

        while self._running:
            if not self._is_connected:
                opened = self._open_capture()
                if not opened:
                    logger.warning(f"Video source offline. Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, self.max_reconnect_interval_s)
                    continue
                else:
                    backoff = self.reconnect_interval_s
                    logger.info("Video source acquired/reconnected.")
                    if self.on_reconnect:
                        try:
                            self.on_reconnect()
                        except Exception as e:
                            logger.error(f"Error in on_reconnect callback: {e}")

            # Simulated hardware drop test hook
            if self._simulate_failure:
                self._handle_disconnect()
                time.sleep(1.0)
                continue

            frame = None
            if callable(self.source):
                try:
                    frame = self.source()
                except Exception as e:
                    logger.error(f"Error in synthetic frame generator: {e}")
                    frame = None
            else:
                if self._cap is None or not self._cap.isOpened():
                    self._handle_disconnect()
                    continue
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    if self._source_type == "video_file" and self.loop_video:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.warning("Frame read failed or video ended.")
                        self._handle_disconnect()
                        continue

            now_mono = time.monotonic()
            now_wall = time.time()
            self._frame_count += 1

            # Resize if needed
            h, w = frame.shape[:2]
            if (w, h) != self.target_resolution and self.target_resolution != (0, 0):
                frame = cv2.resize(frame, self.target_resolution, interpolation=cv2.INTER_LINEAR)

            # Measure frame-to-frame FPS
            dt = now_mono - last_frame_time
            last_frame_time = now_mono
            if dt > 0:
                fps_tracker.append(1.0 / dt)
                if len(fps_tracker) > 30:
                    fps_tracker.pop(0)
            avg_fps = sum(fps_tracker) / len(fps_tracker) if fps_tracker else self.target_fps

            # Put into zero-lag latest-frame buffer
            self.buffer.put(
                frame=frame,
                timestamp_wall=now_wall,
                timestamp_mono=now_mono,
                source_type=self._source_type,
                frame_id=self._frame_count,
                fps_measured=round(avg_fps, 2)
            )

            # Throttle ingestion to respect target FPS if reading too quickly (e.g., video files)
            elapsed = time.monotonic() - now_mono
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _handle_disconnect(self) -> None:
        with self._lock:
            was_connected = self._is_connected
            self._is_connected = False
        self._release_capture()
        if was_connected:
            logger.warning("Camera stream disconnected!")
            if self.on_disconnect:
                try:
                    self.on_disconnect()
                except Exception as e:
                    logger.error(f"Error in on_disconnect callback: {e}")

