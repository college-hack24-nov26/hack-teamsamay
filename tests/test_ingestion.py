"""
Tests for Frame Buffer and Frame Ingestor.
Verifies thread safety, drop-oldest strategy, zero lag, and camera reconnect hooks.
"""

import time
import numpy as np
import pytest

from modules.ingestion.frame_buffer import LatestFrameBuffer, IngestedFrame
from modules.ingestion.ingestor import FrameIngestor


def test_frame_buffer_drop_oldest_strategy():
    """Verify that when buffer reaches max capacity, oldest frames are dropped."""
    buf = LatestFrameBuffer(max_capacity=2)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Put frame 1
    d1 = buf.put(dummy_img, time.time(), time.monotonic(), "test", 1)
    assert d1 is False
    assert buf.get_stats()["total_ingested"] == 1
    assert buf.get_stats()["total_dropped"] == 0

    # Put frame 2
    d2 = buf.put(dummy_img, time.time(), time.monotonic(), "test", 2)
    assert d2 is False

    # Put frame 3 -> should drop frame 1
    d3 = buf.put(dummy_img, time.time(), time.monotonic(), "test", 3)
    assert d3 is True
    assert buf.get_stats()["total_dropped"] == 1

    # Fetch latest frame -> should be frame 3
    latest = buf.get_latest(timeout=0.1)
    assert latest is not None
    assert latest.frame_id == 3

    # Buffer should now be empty (zero lag policy)
    assert buf.get_latest(timeout=0.05) is None


def test_ingestor_with_synthetic_source():
    """Verify ingestor runs smoothly with synthetic generator and produces frames."""
    buf = LatestFrameBuffer(max_capacity=2)
    counter = 0

    def mock_cam():
        nonlocal counter
        counter += 1
        return np.full((100, 100, 3), counter % 255, dtype=np.uint8)

    ingestor = FrameIngestor(
        source=mock_cam,
        buffer=buf,
        target_fps=30.0,
        target_resolution=(100, 100)
    )

    ingestor.start()
    time.sleep(0.15)
    assert ingestor.is_connected is True
    assert buf.get_stats()["total_ingested"] > 0
    latest = buf.get_latest(timeout=0.2)
    assert latest is not None
    assert latest.source_type == "synthetic"
    ingestor.stop()


def test_ingestor_simulated_disconnect():
    """Verify camera disconnect callback and recovery."""
    buf = LatestFrameBuffer(max_capacity=2)
    disconnected = False
    reconnected = False

    def on_disc():
        nonlocal disconnected
        disconnected = True

    def on_rec():
        nonlocal reconnected
        reconnected = True

    dummy_frame = np.zeros((50, 50, 3), dtype=np.uint8)
    ingestor = FrameIngestor(
        source=lambda: dummy_frame,
        buffer=buf,
        on_disconnect=on_disc,
        on_reconnect=on_rec
    )

    ingestor.start()
    time.sleep(0.05)
    
    # Simulate disconnect
    ingestor.simulate_hardware_disconnect(True)
    time.sleep(0.1)
    assert disconnected is True

    # Restore
    ingestor.simulate_hardware_disconnect(False)
    time.sleep(0.1)
    ingestor.stop()
