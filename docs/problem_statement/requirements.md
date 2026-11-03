# SUTRA: System Requirements Specification (SRS)
## SIH26174 - AI Human Activity Recognition for On-board BAS Experiments

- **Project:** SUTRA (Smart Unified Task & Recognition Assistant)
- **Team ID:** 162833 | **Team Name:** Team Samay
- **Status:** Architectural Draft - Pre-Implementation Analysis

---

### 1. Scope and System Context

The system monitors human actions during scientific payload experiments aboard the Bharatiya Antariksh Station (BAS). It ensures procedural compliance, detects protocol deviations or missed steps in real time, alerts the astronaut locally, and generates compact down-linkable telemetry records.

The requirements below distinguish strictly between **Confirmed Specifications** (explicitly derived from the SIH proposal), **Working Assumptions** (engineered to build a working prototype), and **Missing Information** that must be clarified with SIH/ISRO organizers.

---

### 2. Confirmed Requirements (from SIH26174 Proposal)

#### 2.1 Functional Requirements
- **REQ-CONF-01: Autonomous Offline Operation**
  The system must execute 100% locally on edge hardware with zero reliance on cloud services, external internet access, or live ground-station links during experiment monitoring.
- **REQ-CONF-02: Microgravity Coordinate Invariance**
  Vision models must evaluate spatial relationships and object positions relative to the experiment payload rack coordinate frame, without assuming terrestrial "up" or "down" gravity vectors.
- **REQ-CONF-03: Multi-Stage Perception Pipeline**
  The core architecture must incorporate human pose estimation, object detection, interaction understanding, and temporal sequence modeling rather than naive single-frame detection.
- **REQ-CONF-04: Protocol State Tracking & Anomaly Detection**
  The system must track multi-step scientific protocols, detect skipped steps, out-of-order execution, and invalid tool usage, and trigger alerts upon deviation.
- **REQ-CONF-05: Low-Bandwidth Telemetry Generation**
  The edge engine must produce structured, timestamped state logs (JSON format) detailing step start/completion timestamps, confidence values, and deviation events, achieving >95% bandwidth savings over raw video feeds.
- **REQ-CONF-06: Local Operator Alerting**
  Visual status updates and localized audio alerts (offline TTS) must be delivered to the operator immediately upon detection of an error or step completion.

#### 2.2 Target Operational Environment
- **REQ-CONF-07: Target Platform Architecture**
  Production deployment is targeted for the **NVIDIA Jetson AGX Orin** platform (15W - 60W power profile) running optimized edge inference.

---

### 3. Assumptions Requiring Technical Validation

Since the official proposal does not provide fine-grained implementation details, the following technical assumptions are established for the initial prototype:

- **ASSUMP-01: Standardized Microgravity Wet-Lab Protocol Mock**
  In the absence of an official ISRO BAS experiment protocol, the MVP will implement a representative 5-step life-sciences / wet-lab biological sample transfer workflow:
  1. *Step 1 - Workspace Inspection & Sanitization:* Hand enters payload enclosure, inspects sterile rack area.
  2. *Step 2 - Sample Container Unsealing:* Hand unscrews/opens the target sample vial.
  3. *Step 3 - Reagent Aspiration:* Hand grasps micropipette/syringe, inserts tip into reagent bottle, and draws liquid.
  4. *Step 4 - Liquid Dispensation:* Hand moves pipette to target vial and dispenses reagent.
  5. *Step 5 - Container Sealing:* Hand reseals the target sample vial and returns pipette to rack cradle.
- **ASSUMP-02: Single-Camera Setup for Laptop MVP**
  The MVP will use a single fixed frontal or overhead RGB camera (standard 720p/1080p UVC webcam or pre-recorded evaluation video feed) covering the payload work zone.
- **ASSUMP-03: Controlled Lighting Conditions**
  The payload enclosure is assumed to have internal LED illumination providing stable exposure without extreme outdoor solar glare or complete darkness.
- **ASSUMP-04: Rigid Payload Frame Anchors**
  Visual fiducials (e.g., ArUco markers or high-contrast structural rack corners) can be detected in initial frames to establish the local normalized coordinate space $(X_{rack}, Y_{rack}, Z_{rack})$.
- **ASSUMP-05: Non-Blocking Telemetry Ingestion**
  Ground communication is intermittent; all telemetry is stored locally in an embedded ACID-compliant store (e.g., SQLite / append-only JSONL) and streamed when a transmission link is simulated.

---

### 4. Missing Information Needed from Official Problem Statement

The following information is not present in the SIH proposal or problem title and represents critical unknowns:

1. **Official BAS Experiment Definition:**
   - What exact scientific domains will be evaluated during SIH judging (e.g., cell biology, fluid physics, metallurgy, crystal growth)?
   - Are there specific flight checklists and formal state transition rules provided by ISRO?
2. **Camera and Sensor Hardware Specifications:**
   - Will the real flight payload use monocular RGB, stereo vision, or active RGB-D (depth) sensors?
   - What are the optical parameters (focal length, field of view, barrel distortion coefficients)?
3. **Hardware Constraints for Live Judging:**
   - Are teams expected to demonstrate exclusively on laptop hardware during hackathon rounds, or will Jetson hardware be physically present/evaluated?
4. **Dataset Availability:**
   - Will ISRO release zero-g parabolic flight footage, neutral buoyancy laboratory video, or synthetic CAD renders for model fine-tuning?
5. **Telemetry and Command Interfaces:**
   - Does ISRO mandate CCSDS (Consultative Committee for Space Data Systems) telemetry packet formatting or custom JSON/Protobuf schemas?

---

### 5. Proposed MVP Features (Laptop Prototype: 8 GB RAM, CPU-Only)

To ensure smooth execution on resource-constrained laptops during hackathon development and testing, the MVP is strictly scoped as follows:

| Component | MVP Implementation | Resource Footprint |
| :--- | :--- | :--- |
| **Video Ingestion** | OpenCV / PyAV threaded ring buffer; supports USB webcam and pre-recorded MP4/AVI experiment footage. | < 50 MB RAM |
| **Object Detection** | Quantized lightweight detector (YOLOv8n / YOLO11n ONNX) targeting 6 core classes: `[hand, vial_sealed, vial_open, pipette, tip_box, waste_bin]`. Runs at 5–10 Hz via ONNX Runtime / OpenVINO CPU provider. | ~150 MB RAM |
| **Hand Pose & Keypoints** | MediaPipe Hands or YOLOv8n-Pose (quantized CPU). Extracts 21 hand keypoints to track index tip, thumb, and palm center. | ~120 MB RAM |
| **Interaction Heuristic (HOI)** | Geometric spatial proximity filter: 2D/3D Euclidean distance between hand keypoints and tool bounding box centers + IoU overlap heuristic. | < 10 MB RAM |
| **Temporal State Engine** | Deterministic Finite State Machine (FSM / DAG) with sliding-window hysteresis debounce (5–10 frame consensus) to confirm action transitions and ignore brief occlusions. | < 20 MB RAM |
| **Local Audio Alerts** | Lightweight offline TTS (pyttsx3 / pre-synthesized Piper WAV alerts cached in RAM). | < 30 MB RAM |
| **Operator GUI / HUD** | Lightweight web dashboard (FastAPI backend with Server-Sent Events / WebSocket to HTML5 Canvas) or native PySide6 UI displaying bounding boxes, state timeline, and deviation flags. | ~100 MB RAM |
| **Telemetry Dispatcher** | Structured JSON event stream writer recording `timestamp`, `event_type`, `step_id`, `confidence`, and `rack_coords`. | < 15 MB RAM |
| **Total System Footprint** | **Runs comfortably within ~1.2 GB - 1.8 GB RAM**, leaving >6 GB free for Windows OS and development tools. |

> [!NOTE]
> Heavyweight modules proposed in the long-term roadmap (such as 3D Video-Swin Transformers, HO-RCNN, and 8-billion parameter LLMs) are **intentionally decoupled** from the MVP prototype to prevent out-of-memory (OOM) crashes and CPU starvation on 8 GB laptops.

---

### 6. Future Advanced Features (Jetson AGX Orin / Flight Target)

When migrating from the laptop prototype to the flight-grade NVIDIA Jetson AGX Orin:

1. **Deep Learning HOI & Temporal Action Segmentation:**
   - Replace geometric heuristics with deep learning models: HO-RCNN and 1D-TCN or Video-Swin Transformer compiled via NVIDIA TensorRT (FP16 / INT8 precision).
2. **Multi-Camera 3D Spatial Fusion:**
   - Ingest multiple synchronized camera streams (e.g., wide payload angle + wrist-mounted micro camera) via zero-copy hardware GStreamer (`nvv4l2camerasrc` $\rightarrow$ `nvvidconv` $\rightarrow$ CUDA memory).
3. **On-Board Conversational Assistant (Offline Multimodal RAG):**
   - Deploy quantized 4-bit Llama-3-8B (or equivalent edge SLM) via `llama.cpp` / TensorRT-LLM alongside FAISS vector embeddings of scientific experiment manuals, enabling astronauts to query procedures hands-free via Whisper.cpp voice recognition.
4. **Autonomous Fiducial-Free Coordinate Mapping:**
   - Visual Odometry / SLAM relative to rack interior geometry, maintaining sub-millimeter precision under severe crew and tool floating motion.
5. **CCSDS Telemetry Packetizer:**
   - Formal space-grade packaging of event logs into CCSDS packets ready for direct integration into satellite downlink telemetry busses.

