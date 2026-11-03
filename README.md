# SUTRA AI Vision Engine
### Smart Unified Task & Recognition Assistant for On-Board Space Station Experiments

- **Competition:** Smart India Hackathon 2026 (SIH 2026)
- **Problem Statement ID:** SIH26174
- **Problem Statement Title:** AI Human Activity Recognition for On-board BAS Experiments
- **Theme:** Space Technology
- **Team ID:** 162833
- **Team Name:** Team Samay

---

## 1. Project Overview

The **SUTRA AI Vision Engine** is a fully autonomous, offline computer vision and temporal human activity recognition engine designed to assist astronauts conducting scientific payload experiments aboard the **Bharatiya Antariksh Station (BAS)**.

Executing microgravity science experiments requires strict adherence to multi-step protocols. In space, cognitive overload, floating orientation shifts, and communication blackouts can lead to irreversible procedural errors. SUTRA acts as an intelligent payload guardian by:
1. Monitoring experiment execution in real time via fixed payload cameras.
2. Tracking human hands, tools, containers, and physical interactions.
3. Validating sequential step execution against a formal Directed Acyclic Graph (DAG) state machine.
4. Flagging procedural deviations (skipped steps, out-of-order execution, incorrect tools) immediately.
5. Emitting compact, down-linkable JSON telemetry logs (>98% bandwidth savings compared to raw video).

---

## 2. System Architecture

```
Camera / Video Stream (Webcam / MP4 / RTSP)
        ↓
Frame Ingestion (Asynchronous Thread, Auto-Reconnect)
        ↓
Zero-Lag Frame Buffer (Drop-Oldest Strategy, Zero-Lag)
        ↓
Object Detection (YOLO ONNX Runtime / Heuristic Lab Detector)
        ↓
Hand & Keypoint Pose Estimation (21 Landmark Kinematic / MediaPipe)
        ↓
Rack Coordinate Normalization (4-Point Microgravity Homography [0.0, 1.0])
        ↓
Hand-Object Interaction Engine (Proximity, IoU, Duration, Grasps/Enters)
        ↓
Temporal HAR & Debounce Engine (Sliding Window, Hysteresis, Occlusion Tolerance)
        ↓
Protocol State Graph / FSM (5-Step Wet-Lab Verification, Deviation Alerts)
        ↓
Telemetry Event Contract (Schema v1.0, JSONL Storage, JSON Export)
        ↓
Event Publisher (Bounded Queue, Exponential Backoff, Offline Resilient)
```

### Why Generic Object Detection Alone Fails
In safety-critical space biology experiments, detecting a syringe or vial in a video frame does **not** prove that fluid aspiration occurred. SUTRA enforces a **multi-stage verification hierarchy**:
- **Physical Preconditions:** Prior DAG steps must be verified.
- **Relational Interaction:** Hand must actively grasp the tool, and the tool must enter the container.
- **Temporal Dwell:** Action must persist for the experimentally required physical duration ($t \ge \Delta t_{min}$), smoothed across temporal hysteresis filters.

---

## 3. Hardware Requirements & Resource Budget

This engine is engineered specifically for **resource-constrained 8 GB RAM laptops without a dedicated GPU**:
- **Execution Target:** x86_64 CPU (Windows 10/11 or Linux).
- **RAM Footprint:** Strictly $\le 1.8\text{ GB}$ (measured peak RSS: **~75 - 165 MB**).
- **Inference Latency:** Target $\le 120\text{ ms}$ (measured mean: **~7.5 - 9.5 ms**).
- **Ingestion:** Zero-lag non-blocking ring buffer that discards stale frames if processing slows.

---

## 4. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.14)
- Git

### Windows Setup
```powershell
# Clone the repository
git clone <repository_url>
cd sutra-ai-engine

# Create virtual environment (optional)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install lightweight dependencies
pip install -r requirements.txt
```

### Linux Setup
```bash
# Clone the repository
git clone <repository_url>
cd sutra-ai-engine

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install lightweight dependencies
pip install -r requirements.txt
```

---

## 5. Running the Engine

The engine provides multiple standalone operational modes via `app/main.py`:

### 1. Synthetic Full Protocol Simulation
Simulates the complete 5-step biological assay with animated tools and hands:
```bash
python app/main.py --mode synthetic --frames 120
```

### 2. Procedural Deviation Demonstration
Simulates an experiment where the astronaut skips Step 3 (Aspiration) and attempts Step 4 directly, triggering an instant `DEVIATION_DETECTED` alert and telemetry log:
```bash
python app/main.py --mode test-deviation --frames 120
```

### 3. Video Replay Mode (MP4 / AVI)
First generate the sample test videos, then replay through the pipeline:
```bash
# Generate sample video files
python scripts/generate_synthetic_video.py

# Replay normal experiment video
python app/main.py --mode video --source data/sample_normal_run.mp4

# Replay deviation experiment video
python app/main.py --mode video --source data/sample_deviation_run.mp4
```

### 4. Live USB Webcam Mode
```bash
python app/main.py --mode live --source 0
```

### 5. Automated Benchmark Suite
Measures actual CPU throughput, latency percentiles, and memory consumption:
```bash
python app/main.py --mode benchmark
```

---

## 6. Telemetry Event Contract (Schema v1.0)

Every state transition and alert produces a structured, validated JSON event:

```json
{
  "schema_version": "1.0",
  "event_id": "ec6b2771-3c32-4330-84db-0d45f4499694",
  "source": "sutra-ai-engine",
  "device_id": "ai-device-01",
  "experiment_id": "DEMO-WETLAB-01",
  "timestamp": "2026-11-03T11:16:55.964025+00:00",
  "event_type": "STEP_COMPLETED",
  "step_id": "STEP_01",
  "step_name": "Workspace Inspection & Sanitization",
  "status": "COMPLETED",
  "confidence": 0.88,
  "evidence": {
    "objects": ["hand"],
    "duration_active": 1.6
  },
  "rack_coordinates": {
    "x": 0.45,
    "y": 0.62,
    "calibration_status": "CALIBRATED"
  },
  "requires_review": false
}
```

### Supported Event Types
- `HEARTBEAT`: Periodic health and alive signal.
- `STEP_STARTED`: Protocol step preconditions satisfied and action initiated.
- `STEP_COMPLETED`: Minimum duration and verified observations achieved.
- `DEVIATION_DETECTED`: Sequence violation (skipped step, out-of-order, incorrect tool).
- `NEEDS_REVIEW`: Ambiguous observation or maximum duration exceeded.
- `CAMERA_DISCONNECTED`: Video feed lost; state checkpoint safely preserved.
- `CAMERA_RECONNECTED`: Video feed recovered; state checkpoint restored.
- `SYSTEM_STATUS`: Initialization and component health diagnostic.

---

## 7. Automated Test Suite

Run the full automated test suite covering all modules:
```bash
python -m pytest tests -v
```

### Test Coverage Summary (26 Tests)
- `tests/test_ingestion.py`: Buffer drop-oldest strategy, zero-lag guarantee, auto-reconnect.
- `tests/test_detection.py`: BoundingBox math, IoU, heuristic detection, ONNX fallback.
- `tests/test_pose.py`: 21 landmark structure, pinch distance, kinematic estimation.
- `tests/test_calibration.py`: Homography perspective transformation, microgravity invariance.
- `tests/test_interaction.py`: Spatial proximity, grasp duration, tool-in-vial relations.
- `tests/test_temporal.py`: Hysteresis debounce, consecutive consensus, occlusion tolerance.
- `tests/test_protocol_fsm.py`: Sequential transitions, duration enforcement, skipped steps, state recovery.
- `tests/test_telemetry.py`: Schema validation, append-only JSONL recording, JSON export.
- `tests/test_communication.py`: Queue bounding, deduplication, mock receiver validation.

---

## 8. Empirical Benchmark Performance

Measured on a standard laptop CPU (Windows x86_64, 12 threads, no GPU):

| Performance Metric | Measured Value | Requirement Target | Status |
| :--- | :--- | :--- | :--- |
| **Throughput Processing** | **136.8 FPS** | 8.0 - 15.0 FPS | **PASS** |
| **Mean Inference Latency** | **7.30 ms** | $\le 100\text{ ms}$ | **PASS** |
| **P50 Latency (Median)** | **6.15 ms** | $\le 90\text{ ms}$ | **PASS** |
| **P95 Latency** | **8.92 ms** | $\le 120\text{ ms}$ | **PASS** |
| **P99 Latency** | **11.35 ms** | $\le 160\text{ ms}$ | **PASS** |
| **Peak Memory Footprint (RSS)** | **161.7 MB** | $\le 1,800\text{ MB}$ | **PASS** |
| **Telemetry Size Reduction** | **98.95%** | $\ge 95.0\%$ | **PASS** |
| **Buffer Queue Lag** | **0 frames** | 0 lag accumulation | **PASS** |

---

## 9. Known Limitations & Roadmap

### Current MVP Scope & Boundaries
1. **Model Weights:** Custom fine-tuned weights (`models/yolo_sutra.onnx`) are trained on user-annotated lab footage. If unpopulated, the engine uses `HeuristicLabDetector` without claiming false neural recognition.
2. **Monocular 2D:** Current MVP uses single-camera 2D perspective mapping; 3D stereo depth requires multi-camera calibration.
3. **Demo Protocol:** The 5-step wet-lab workflow is a representative demonstration assay, not an official ISRO flight protocol.

### Jetson AGX Orin Migration Path
- Compile ONNX models to **NVIDIA TensorRT** FP16/INT8 engines via `Jetson_TensorRTBackend`.
- Ingest multi-camera streams via zero-copy hardware GStreamer pipelines (`nvv4l2camerasrc ! nvvidconv ! NVMM`).
- Integrate local quantized SLM (`llama.cpp` Llama-3-8B GGUF) and Piper TTS for natural language astronaut interaction.
