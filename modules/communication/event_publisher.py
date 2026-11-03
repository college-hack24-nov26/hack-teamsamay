"""
Asynchronous HTTP Event Publisher with Bounded Queue, Exponential Backoff,
and Event Deduplication for SUTRA Vision Engine.
Runs independently; gracefully handles offline or unreachable backend endpoints.
"""

import collections
import logging
import queue
import threading
import time
from typing import Optional
import requests

from modules.telemetry.event_schema import TelemetryEvent

logger = logging.getLogger("sutra.communication.publisher")


class EventPublisher:
    """
    Background worker that flushes queued events to an HTTP endpoint.
    If endpoint is absent or offline, queues events up to max_queue_size
    and retries with exponential backoff without stalling the AI Vision pipeline.
    """

    def __init__(
        self,
        target_url: Optional[str] = None,
        max_queue_size: int = 1000,
        initial_backoff_s: float = 0.5,
        max_backoff_s: float = 10.0,
        request_timeout_s: float = 2.0
    ):
        self.target_url = target_url
        self.max_queue_size = max_queue_size
        self.initial_backoff_s = initial_backoff_s
        self.max_backoff_s = max_backoff_s
        self.request_timeout_s = request_timeout_s

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._sent_event_ids: collections.deque = collections.deque(maxlen=2000)
        self._sent_ids_set: set[str] = set()

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._is_server_reachable = False
        self._total_published = 0
        self._total_failed = 0
        self._total_dropped = 0

    @property
    def is_connected(self) -> bool:
        return self._is_server_reachable

    def start(self) -> None:
        """Start the publisher background worker."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._publish_loop, name="EventPublisher", daemon=True)
        self._worker_thread.start()
        logger.info(f"EventPublisher started. Target: {self.target_url or 'OFFLINE / LOCAL ONLY'}")

    def stop(self) -> None:
        """Gracefully stop publisher worker."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        logger.info("EventPublisher stopped.")

    def publish(self, event: TelemetryEvent) -> bool:
        """
        Enqueues an event for delivery. Drops oldest if queue exceeds capacity.
        Returns True if enqueued, False if dropped due to queue congestion.
        """
        # Deduplication check
        if event.event_id in self._sent_ids_set:
            logger.debug(f"Duplicate event {event.event_id} ignored.")
            return True

        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            # Drop oldest event to prevent memory bloat
            try:
                dropped: TelemetryEvent = self._queue.get_nowait()
                self._total_dropped += 1
                logger.warning(f"EventPublisher queue congested. Dropped oldest event {dropped.event_id}")
            except queue.Empty:
                pass
            self._queue.put_nowait(event)
            return False

    def _publish_loop(self) -> None:
        backoff = self.initial_backoff_s

        while self._running:
            if not self.target_url:
                # Local-only mode; do not attempt network requests
                time.sleep(1.0)
                continue

            try:
                # Wait for event
                event: TelemetryEvent = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Attempt transmission with retry loop
            delivered = False
            while self._running and not delivered:
                try:
                    headers = {"Content-Type": "application/json"}
                    payload = event.model_dump()
                    resp = requests.post(
                        self.target_url,
                        json=payload,
                        headers=headers,
                        timeout=self.request_timeout_s
                    )
                    if resp.status_code in (200, 201, 202):
                        delivered = True
                        self._is_server_reachable = True
                        self._total_published += 1
                        backoff = self.initial_backoff_s

                        # Record for deduplication
                        if len(self._sent_event_ids) >= 2000:
                            old_id = self._sent_event_ids.popleft()
                            self._sent_ids_set.discard(old_id)
                        self._sent_event_ids.append(event.event_id)
                        self._sent_ids_set.add(event.event_id)
                        self._queue.task_done()
                    else:
                        logger.warning(f"Publisher got HTTP {resp.status_code} from {self.target_url}")
                        self._handle_network_failure(backoff)
                        backoff = min(backoff * 1.5, self.max_backoff_s)
                except Exception as e:
                    logger.debug(f"Publisher connection error to {self.target_url}: {e}")
                    self._handle_network_failure(backoff)
                    backoff = min(backoff * 1.5, self.max_backoff_s)

    def _handle_network_failure(self, backoff: float) -> None:
        self._is_server_reachable = False
        self._total_failed += 1
        time.sleep(backoff)

    def get_stats(self) -> dict:
        return {
            "target_url": self.target_url,
            "is_connected": self._is_server_reachable,
            "queue_depth": self._queue.qsize(),
            "total_published": self._total_published,
            "total_failed": self._total_failed,
            "total_dropped": self._total_dropped
        }

