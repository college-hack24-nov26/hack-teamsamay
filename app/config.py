"""
Configuration Management for SUTRA Vision Engine.
Loads environment variables and sets conservative, CPU-optimized defaults for 8 GB RAM laptops.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

# Absolute directory of sutra-ai-engine project root
BASE_DIR = Path(__file__).resolve().parent.parent


class SutraConfig(BaseSettings):
    """Global configuration settings for the AI Vision Engine."""

    # Hardware & System Identity
    device_id: str = Field(default="ai-device-01", description="Physical unit ID")
    experiment_id: str = Field(default="DEMO-WETLAB-01", description="Experiment protocol ID")
    
    # Ingestion Settings
    video_source: str = Field(default="0", description="0 for webcam, or path to MP4/AVI file, or rtsp://")
    target_fps: float = Field(default=15.0, description="Target ingestion FPS for CPU budget")
    frame_width: int = Field(default=640, description="Video width")
    frame_height: int = Field(default=480, description="Video height")
    loop_video: bool = Field(default=True, description="Loop input video continuously")

    # Detection & Pose Backends
    model_path: str = Field(
        default=str(BASE_DIR / "models" / "yolo_sutra.onnx"),
        description="Path to custom ONNX weights"
    )
    detection_conf_threshold: float = Field(default=0.45, description="Object detection threshold")
    pose_backend: str = Field(default="kinematic", description="'mediapipe' or 'kinematic'")
    
    # Protocol & Calibration
    protocol_path: str = Field(
        default=str(BASE_DIR / "protocols" / "demo_wetlab.yaml"),
        description="YAML protocol definition"
    )
    calibration_path: str = Field(
        default=str(BASE_DIR / "data" / "calibration.json"),
        description="JSON rack calibration file"
    )

    # Storage & Telemetry
    storage_log_path: str = Field(
        default=str(BASE_DIR / "data" / "events" / "telemetry_events.jsonl"),
        description="Local JSONL path"
    )
    storage_export_path: str = Field(
        default=str(BASE_DIR / "data" / "events" / "telemetry_export.json"),
        description="Export JSON path"
    )

    # Network / Publisher
    publisher_url: Optional[str] = Field(default=None, description="Optional backend endpoint URL")
    enable_http_publisher: bool = Field(default=False, description="Enable background HTTP publisher")

    model_config = {
        "env_prefix": "SUTRA_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


def get_config() -> SutraConfig:
    return SutraConfig()
