# SUTRA: AI Human Activity Recognition for On-board BAS Experiments
## Problem Statement Analysis and Technical Dissection

- **Problem Statement ID:** SIH26174
- **Problem Statement Title:** AI Human Activity Recognition for On-board BAS Experiments
- **Theme:** Space Technology
- **Team ID:** 162833
- **Team Name:** Team Samay
- **Project Name:** SUTRA (Smart Unified Task & Recognition Assistant)

---

### 1. Simple Technical Explanation of the Problem

The **Bharatiya Antariksh Station (BAS)** is India's planned space station. On-board BAS, astronauts (vyomanauts) will conduct sensitive scientific experiments spanning biology, materials science, fluid dynamics, and microgravity crystallization. 

Executing these experiments involves rigorous, multi-step scientific protocols (e.g., sterilizing sample chambers, mounting cuvettes, pipetting reagents, sealing culture vessels, initiating thermal cycles). In space, performing these tasks is difficult due to several factors:
1. **Microgravity disorientation:** Crew members and floating objects have no fixed terrestrial gravitational reference frame ("up" vs. "down").
2. **High cognitive workload:** Astronauts must juggle dozens of protocols across multiple domains, increasing the probability of human error (e.g., skipping a step, reversing execution order, using an incorrect reagent, or missing timing windows).
3. **Communication latency and blackouts:** Spacecraft orbits lead to communication blackouts or high latency with Earth ground stations. Ground control cannot watch live high-definition video to guide every step in real time.
4. **Bandwidth constraints:** Continuous streaming of multi-camera 1080p/4K video back to Earth is impossible over constrained deep-space/orbital satellite links.

**The core problem:** We must create an **autonomous, fully offline, edge-deployable AI assistant** that monitors experiment execution in real time using payload-mounted cameras. The system must verify that each action in a protocol is executed accurately, flag procedural deviations or safety violations immediately to the astronaut via local visual and audio alerts, and record/downlink only lightweight structured telemetry (status logs) rather than raw video.

---

### 2. Dissection of Computer Vision & Action Understanding Technologies

A critical architectural pitfall in Activity Recognition projects is conflating basic visual detection with full temporal task comprehension. To monitor procedural scientific experiments, four distinct computer vision paradigms must be integrated into a cohesive hierarchy:

```
+-----------------------------------------------------------------------------------+
| 4. Temporal Human Activity Recognition & Protocol State Tracking                   |
|    "Astronaut is dispensing 50 uL buffer into test tube A (Step 3 in progress)"   |
+-----------------------------------------^-----------------------------------------+
                                          |
+-----------------------------------------+-----------------------------------------+
| 3. Hand-Object Interaction (HOI) Detection                                        |
|    Relational tuple: <Right Hand, Holding/Depressing, Micropipette>               |
+-----------------------------------------^-----------------------------------------+
                                          |
+------------------------------------+----+-----------------------------------------+
| 1. Object Detection                | 2. Human Pose & Hand Landmark Estimation     |
|    Localizes tools, vials, racks   |    Tracks wrists, fingers, joints in space   |
+------------------------------------+----------------------------------------------+
```

| Technology | Input | Output / Representation | Role in BAS Experiment Monitoring | Critical Limitation if Used Alone |
| :--- | :--- | :--- | :--- | :--- |
| **Object Detection** *(e.g., YOLOv8/v11, RT-DETR)* | Single static frame ($H \times W \times 3$) | 2D/3D Bounding Boxes with class labels and confidence scores (e.g., `micropipette: 0.92`, `reagent_vial: 0.88`). | Identifies presence, locations, and inventory of experimental tools, containers, and apparatus in the payload bay. | **Zero temporal or behavioral awareness.** Detecting a pipette next to a vial does *not* indicate that a transfer took place, nor whether it was done correctly. |
| **Human Pose Estimation** *(e.g., YOLO-Pose, MediaPipe Pose/Hands)* | Single static frame ($H \times W \times 3$) | Anatomical keypoints $(x, y, z, v)$ for joints, wrists, palms, and fingertips. | Tracks hand trajectories, posture, reaching motions, and proximity to payload work zones. | **Semantic blindness to manipulated items.** Knows where wrists and fingers are, but cannot determine whether the hand is grasping a sterile pipette, an empty tube, or nothing. |
| **Hand-Object Interaction (HOI)** *(e.g., HO-RCNN, Interaction Transformers)* | Static frame or short pair of frames + Pose + Objects | Relational `<Subject, Action/Predicate, Object>` triplets (e.g., `<Hand_R, Grasp, Pipette_1>`). | Determines active physical engagement between the astronaut and specific experimental tools or containers. | **Lacks long-term state awareness.** Identifies that a pipette is gripped at time $t$, but cannot determine if fluid was drawn, if it was mixed for 30 seconds, or if Step 2 preceded Step 3. |
| **Temporal Human Activity Recognition (HAR) & Action Segmentation** *(e.g., 1D-TCN, Video-Swin, State Machine)* | Sequence of feature vectors over time window $T$ ($t_0 \dots t_k$) | Temporal action boundary, phase prediction, and protocol state verification (e.g., `Pipetting_Reagent` [Start: 00:14, End: 00:28, Status: COMPLETED]). | Validates execution of complete procedural steps over time against an experiment protocol graph (DAG), filtering occlusions and detecting skipped/reordered steps. | **Computationally intensive.** Full 3D video transformers cannot run naively on low-power edge devices; requires decoupled feature extraction and temporal state modeling. |

> [!IMPORTANT]
> **Core Architectural Rule:** Generic object detection is **never** proof of completed activity. An experiment step like *"Incubate sample for 60 seconds after reagent injection"* requires temporal duration tracking, state history, and interaction confirmation—not merely the visual presence of a syringe.

---

### 3. Application to On-Board BAS Experiment Monitoring

Integrating these four technologies creates an automated, fail-safe guardian for space experiments:

1. **Microgravity Pose & Tool Decoupling:**
   Astronauts may float horizontally, upside-down, or obliquely relative to the cabin. By using pose estimation coupled with fixed payload rack fiducials (coordinate anchors), positions are calculated *relative to the experiment rack frame*, eliminating dependence on terrestrial gravitational vectors ("up/down").
2. **Continuous Step Verification:**
   The temporal state engine monitors the transition from one sub-step to the next (e.g., Container Opened $\rightarrow$ Tool Grasped $\rightarrow$ Reagent Drawn $\rightarrow$ Reagent Dispensed $\rightarrow$ Container Closed).
3. **Real-Time Procedural Deviation Alerts:**
   If an astronaut reaches to close a container without dispensing reagent (Step 3 skipped), the HOI and temporal state logic detect an invalid transition in the Protocol Directed Acyclic Graph (DAG) and fire an immediate visual and auditory alert before irreversible contamination or protocol failure occurs.
4. **Tolerance to Hand and Body Occlusions:**
   In close-quarter payload enclosures, an astronaut's hands or arms frequently block the camera's view of small vials. A temporal model (such as a 1D Temporal Convolutional Network or debounce state filter) bridges multi-frame visual dropouts by maintaining state continuity across short occlusion intervals.
5. **Ultra-Low-Bandwidth Telemetry Downlink:**
   Instead of downlinking gigabytes of high-definition video over congested satellite links, the edge node emits timestamped, cryptographically verifiable, lightweight JSON status packets (<50 bytes per event), reducing required transmission bandwidth by >95%.

---

### 4. Confirmed vs. Unknown Analysis (Based on Provided SIH Proposal)

To build a reliable system, we must clearly separate what the official proposal confirms from what remains unspecified or assumed.

```
+-------------------------------------------------------------------------------+
|                       INFORMATION STATUS MATRIX                               |
+-------------------------------------------------------------------------------+
| CONFIRMED BY PDF PROPOSAL                    | UNKNOWN / MISSING SPECIFICATIONS|
| - PS ID: SIH26174                             | - Exact official BAS protocols |
| - PS Title: AI HAR for On-board BAS Exp.     | - Camera specs, FOV, mounts    |
| - Theme: Space Technology                    | - Official ISRO training data  |
| - Target Edge Hardware: Jetson AGX Orin      | - Evaluation / judging harness |
| - High-level pipeline: YOLO + Swin + TCN +   | - Lighting and rack dimensions |
|   DAG + Offline LLM (Llama-3-8B) + TTS       | - Target scientific domain     |
| - Core goal: Fully offline, microgravity-    | - CCSDS telemetry packet spec  |
|   tolerant procedural verification           | - Audio/noise levels in rack   |
+-------------------------------------------------------------------------------+
```

#### A. Confirmed Information
1. **Competition Identifiers:** Problem Statement SIH26174, Theme: Space Technology, Team ID: 162833, Team: Team Samay, Project Name: SUTRA.
2. **Operational Environment:** Bharatiya Antariksh Station (BAS) payload racks; microgravity conditions where crew and tools float; zero real-time cloud or ground-station connectivity during experiments.
3. **Target Production Hardware Architecture:** NVIDIA Jetson AGX Orin (15W - 60W power profile) running fully offline inference.
4. **High-Level Proposed Tech Stack (Proposal Vision):**
   - Video ingestion via GStreamer / CUDA.
   - Vision & Pose: YOLOv8 / YOLO11 for detection/pose, Video-Swin for action recognition.
   - Temporal & State Logic: 1D-TCN for sequence smoothing, HO-RCNN for interaction validation, Python DAG for protocol state graph.
   - Offline Cognitive Assistant: Llama-3-8B (GGUF quantized), FAISS/ChromaDB vector store, Whisper.cpp for STT, Piper TTS for local audio alerts.
   - Telemetry: SQLite/JSON state logs.
5. **High-Level Performance Claims / Aims:**
   - Real-time execution latency (<100 ms on Jetson).
   - Telemetry bandwidth reduction (>95% vs raw video).
   - 100% offline edge execution.

#### B. Unknown and Unspecified Items (Not in PDF)
1. **Official Scientific Protocol Specifications:** The PDF provides an illustrative 4-step example (*1. Open Container, 2. Take Sample, 3. Transfer Sample, 4. Close Container*), but does **not** specify which official BAS scientific experiment protocols ISRO requires (e.g., protein crystal growth, cell culturing, fluid physics, or plant growth).
2. **Payload Rack Physical & Optical Setup:**
   - Camera count, positions, focal length, field-of-view (FOV), and resolutions are unspecified.
   - Whether RGB-D (depth) sensors are permitted or strictly mono RGB cameras.
   - Work volume lighting stability and background contrast inside the rack.
3. **Official Dataset & Annotations:**
   - No dataset has been supplied by ISRO or SIH.
   - Ground truth annotation conventions (bounding box taxonomy, keypoint standards, temporal action boundaries) are unknown.
4. **Flight Hardware & Software Compliance:**
   - Specific real-time OS (RTOS) or Linux kernel constraints (e.g., Jetpack version, safety-critical Linux).
   - Exact telemetry packet standard (e.g., CCSDS Space Packet Protocol vs raw JSON over UDP/Serial).
5. **Evaluation and Benchmark Harness:**
   - How the SIH jury will evaluate the software: on pre-recorded test video clips, live camera input in an exhibition hall, or synthetic 3D simulations.

---

### 5. Architectural Strategy: Two-Tier Development Model

Because the hackathon development phase takes place on consumer laptops (e.g., 8 GB RAM, standard quad/hexa-core CPU, no discrete NVIDIA GPU), we establish a clear two-tier strategy:

1. **Phase 1: CPU-First Resilient Prototype (MVP)**
   - Tailored specifically to run within 8 GB RAM and standard CPU budgets.
   - Employs lightweight, quantized models (YOLO11n/YOLOv8n ONNX via OpenVINO/CPU runtime, MediaPipe hand/pose estimation, deterministic spatial interaction heuristics, and a temporal state machine).
   - Runs mock experiment protocols simulating space station biological assays.
2. **Phase 2: High-Performance Jetson AGX Orin Target**
   - Modularly swaps CPU backends for TensorRT-accelerated Deep HOI, Video-Swin/1D-TCN models, zero-copy NVMM GStreamer pipelines, and quantized local LLMs for crew Q&A.
   - Clean hardware abstraction layers (HAL) ensure the core protocol state engine and telemetry pipeline remain identical between Phase 1 and Phase 2.

