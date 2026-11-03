# SUTRA AI — Autonomous BAS Experiment Guardian & Vision Engine
### Smart Unified Task & Recognition Assistant for On-Board Bharatiya Antariksh Station (BAS) Experiments

[![SIH 2026](https://img.shields.io/badge/SIH-2026-orange.svg)](https://www.sih.gov.in/)
[![Problem Statement](https://img.shields.io/badge/Problem%20ID-SIH26174-blue.svg)](https://www.sih.gov.in/)
[![Theme](https://img.shields.io/badge/Theme-Space%20Technology-purple.svg)](https://www.sih.gov.in/)
[![Team](https://img.shields.io/badge/Team-Team%20Samay%20(ID:%20162833)-success.svg)](https://github.com/college-hack24-nov26/hack-teamsamay)
[![Tests](https://img.shields.io/badge/Tests-28%2F28%20Passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-CPU%20Only%20(No%20GPU%20Required)-lightgrey.svg)]()

---

## 1. Project Overview

**SUTRA** is a fully autonomous, offline Computer Vision and Temporal Human Activity Recognition (HAR) Engine built for astronauts conducting wetlab biological and chemical experiments on the **Bharatiya Antariksh Station (BAS)**.

Astronauts working in pressurized suits or cleanroom environments cannot type on keyboards or interact with chatbots while handling liquid reagents, biological vials, and micropipettes. SUTRA serves as an **autonomous, hands-free mission guardian**:
1. **The Eyes (Vision Perception):** Real-time neural detection of the **Astronaut**, **Tools** (pipettes, sampling tools), and **Samples** (vials, containers) with deep learning wrist/hand vector pose tracking.
2. **The Nervous System (Temporal State Machine):** Real-time evaluation of Human-Object Interactions (HOI) over time, validated against a formal protocol Directed Acyclic Graph (DAG).
3. **The Brain (Autonomous Flight Copilot):** Automatically checks procedural integrity—verifying the **Expected Sequence** against the **Observed Sequence** and detecting skipped, delayed, or out-of-order steps.
4. **The Voice (Interactive Voice AI Assistant):** Issues spoken voice directives aloud through the astronaut's headset, with two-way microphone conversation support (`🎙️ Talk to SUTRA`).

---

## 2. Official Problem Statement Specification & UI Architecture

SUTRA strictly implements the official visual verification layout:

```
+----------------------------------------------------+------------------------------------------------+
|                   VIDEO HUD                        |          VERIFICATION PANELS                   |
+----------------------------------------------------+------------------------------------------------+
|  [ Astronaut ]  (Green Bounding Box)               |  [✓] Expected Sequence (Green Card)            |
|       │                                            |      1. Open Container                     ✓   |
|       ├─ [ Tool ] (Blue Bounding Box)              |      2. Take Sample                         ✓   |
|       │                                            |      3. Transfer Sample                     ✓   |
|       └─ [ Sample ] (Orange Bounding Box)          |      4. Close Container                     ✓   |
|                                                    +------------------------------------------------+
|                                                    |  [⚠] Observed Sequence (Red Card)              |
|                                                    |      1. Open Container                     ✓   |
|                                                    |      2. Take Sample                         ✓   |
|                                                    |      4. Close Container                     ✗   |
|                                                    +------------------------------------------------+
|                                                    |  (!) Step 3 Skipped (Alert Pill)               |
|                                                    +------------------------------------------------+
|                                                    |  (🔊) Voice Alert Issued (Audio Announcement)  |
+----------------------------------------------------+------------------------------------------------+
```

---

## 3. Key Capabilities & Technical Innovations

- **Neural Detection with Zero False Head Classifications:** Employs Ultralytics YOLOv8n with head/face spatial exclusion zones, ensuring the operator's face is never misclassified as a hand or tool.
- **Universal Tool & Apparatus Recognition:** Automatically detects and maps lab items (`vial_sealed`, `vial_open`, `pipette`, `sample_box`, `apparatus`) as well as everyday testing tools (pens, bottles, chargers, cups, devices).
- **YOLO Deep-Learning Body & Hand Pose:** Localizes wrists, arms, and hand orientation vectors even when the astronaut's hands are occluded behind objects or holding tools.
- **Sub-30ms CPU Latency (30+ FPS):** Optimized 320x320 neural inference running on standard x86 CPU without requiring dedicated NVIDIA GPUs.
- **Multi-Source Video Input:**
  - 💻 Standard Laptop / USB Webcam
  - 📱 Android Phone Camera via IP Webcam (`http://<phone_ip>:8080/video`)
  - 🎬 Preset SIH Deviation Videos (Step 3 Skipped Replay)
  - 📁 Custom MP4 Video File Upload & Playback
  - 🧪 4-Step Synthetic Demonstration Simulation
- **Two-Way Voice AI Dialogue:** Click the mic button or press **Spacebar** to speak directly to SUTRA in natural language (*"What is this object?"*, *"What step is next?"*) and receive spoken audio replies.

---

## 4. Hardware Budget & Empirical Benchmarks

Tested on standard AMD Ryzen 5 CPU (8 GB RAM, no dedicated GPU):

| Metric | Target | Measured Empirical Performance |
| :--- | :---: | :---: |
| **Inference Latency (P95)** | $\le 120\text{ ms}$ | **~22.4 ms** |
| **Video Playback FPS** | $\ge 15\text{ FPS}$ | **30 - 32 FPS** |
| **Process Memory (RSS)** | $\le 1800\text{ MB}$ | **~115 - 135 MB** |
| **Telemetry Size Reduction** | $\ge 95\%$ | **98.4% bandwidth savings** vs raw video |
| **Camera Reconnect Recovery** | $\le 3\text{ s}$ | **< 1.0 s** with state checkpoint restore |

---

## 5. Quick Start Guide

### Step 1: Clone Repository
```powershell
git clone https://github.com/college-hack24-nov26/hack-teamsamay.git
cd hack-teamsamay
```

### Step 2: Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 3: Launch SUTRA Live Web HUD
```powershell
python app/main.py --mode live --web
```

Open your browser at: **`http://127.0.0.1:8000`**

---

## 6. Video Source & Testing Options

Once the web dashboard is open, you can switch between sources with 1 click:

1. **Laptop Webcam:** Click **Laptop Webcam** to test with your laptop's built-in camera.
2. **Play Deviation Video:** Click **Play Deviation Video (Step 3 Skipped)** to watch the exact scenario where the astronaut unseals the container, takes the sample, but skips Step 3 and attempts Step 4! Watch the **Observed Sequence** card turn red, the **Step 3 Skipped** pill appear, and hear the audio announcement: *"Caution! Step 3 Skipped!"*
3. **Play Nominal Video:** Click **Play Nominal Video** to watch all 4 steps verified with green checkmarks.
4. **Upload Custom MP4:** Click **Upload MP4 Video** to upload any video file from your computer and analyze it in real time.
5. **Connect Android Phone (IP Webcam):**
   - Install the **IP Webcam** app from Google Play Store on your Android phone.
   - Connect phone and laptop to the same Wi-Fi / Hotspot.
   - Tap **Start server** in the app.
   - Enter `http://<phone_ip>:8080/video` in the dashboard and click **Connect Mobile**.

---

## 7. Interactive Voice Assistant Guide

SUTRA includes a built-in Two-Way Voice AI:
- **Click the Mic Button** (`🎙️ Talk to SUTRA`) or press the **Spacebar**.
- Speak into your microphone:
  - *"What object am I holding?"*
  - *"What is the active protocol step?"*
  - *"Is my current action safe?"*
- SUTRA will analyze the live camera frames and **speak the flight guidance response out loud** in real time!

---

## 8. Automated Test Suite

Run all unit and integration tests across the vision pipeline, temporal engine, FSM state machine, and copilot:

```powershell
python -m pytest tests -v
```

Output:
```text
======================= 28 passed in 11.58s =======================
```

---

## 9. Team & Submission Details

- **Event:** Smart India Hackathon 2026 (SIH2026)
- **Problem Statement ID:** SIH26174
- **Problem Statement Title:** AI Human Activity Recognition for On-board BAS Experiments
- **Theme:** Space Technology
- **Team ID:** 162833
- **Team Name:** Team Samay
- **Repository:** [https://github.com/college-hack24-nov26/hack-teamsamay.git](https://github.com/college-hack24-nov26/hack-teamsamay.git)
