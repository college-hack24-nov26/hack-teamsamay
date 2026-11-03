"""
Local Telemetry Storage and Event Log Manager for SUTRA Vision Engine.
Provides crash-resilient append-only JSONL recording, JSON export, and replay capabilities.
"""

import json
import logging
import os
import threading
from typing import Optional

from modules.telemetry.event_schema import TelemetryEvent

logger = logging.getLogger("sutra.telemetry.storage")


class EventStorage:
    """
    Append-only telemetry storage. Ensures all events are persisted locally to disk
    regardless of network connectivity.
    """

    def __init__(self, log_file_path: str = "data/events/telemetry_events.jsonl"):
        self.log_file_path = log_file_path
        self._lock = threading.Lock()
        self._total_bytes_written = 0
        self._event_count = 0

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(log_file_path)), exist_ok=True)
        
        # Initialize stats from existing file if present
        if os.path.isfile(self.log_file_path):
            try:
                with open(self.log_file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self._event_count += 1
                self._total_bytes_written = os.path.getsize(self.log_file_path)
            except Exception:
                pass

    def record_event(self, event: TelemetryEvent) -> None:
        """Append event to the JSONL log file synchronously with thread safety."""
        line = event.to_json_str() + "\n"
        encoded = line.encode("utf-8")
        
        with self._lock:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
            self._total_bytes_written += len(encoded)
            self._event_count += 1

    def read_events(self, limit: Optional[int] = None) -> list[TelemetryEvent]:
        """Read recorded events from disk in chronological order."""
        events: list[TelemetryEvent] = []
        if not os.path.isfile(self.log_file_path):
            return events

        with self._lock:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            events.append(TelemetryEvent(**data))
                            if limit and len(events) >= limit:
                                break
                        except Exception as e:
                            logger.error(f"Malformed telemetry line: {e}")
        return events

    def export_to_json(self, export_path: str) -> int:
        """Export all recorded events as a single structured JSON array."""
        events = self.read_events()
        raw_list = [e.model_dump() for e in events]
        os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(raw_list, f, indent=2)
        logger.info(f"Exported {len(events)} telemetry events to {export_path}")
        return len(events)

    def get_stats(self) -> dict:
        """Return operational telemetry metrics."""
        with self._lock:
            return {
                "log_file": self.log_file_path,
                "total_events": self._event_count,
                "total_bytes_written": self._total_bytes_written,
                "size_kb": round(self._total_bytes_written / 1024.0, 2)
            }
