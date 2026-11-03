# SUTRA: Performance Targets & Benchmarking Specifications
## Quantitative Metrics, Test Harnesses, and Operational Boundaries

- **Project:** SUTRA (Smart Unified Task & Recognition Assistant)
- **Problem Statement ID:** SIH26174 | **Theme:** Space Technology
- **Team ID:** 162833 | **Team Name:** Team Samay

---

### 1. Performance Target Matrix: Laptop Prototype vs. Jetson AGX Orin

> [!WARNING]
> **Strict Operational Boundary:** The laptop prototype is designed to prove architectural validity, temporal logic, and state verification on an 8 GB RAM CPU-only machine. It is **not** expected to match the hardware-accelerated throughput, sub-100ms latency, or neural deep-HOI execution of the production Jetson AGX Orin.

| Metric | Metric Definition | Laptop Prototype Target (8 GB RAM, CPU-Only) | Jetson AGX Orin Target (15W - 60W, TensorRT) | Verification / Measurement Method |
| :--- | :--- | :--- | :--- | :--- |
| **1. Inference Latency** | Time taken from frame arrival in memory to model bounding box & keypoint output. | $\le 120\text{ ms}$ (per frame, INT8/FP32 CPU ONNX) | $\le 25\text{ ms}$ (per frame, FP16/INT8 TensorRT) | High-precision monotonic timer (`time.perf_counter_ns()`) wrapped around `model.run()`. |
| **2. Effective Processing FPS** | Rate at which the perception & state pipeline processes visual frames without dropped events. | **8 – 12 FPS** (subsampled from 30 FPS camera feed) | **30 – 60 FPS** (full native frame rate processing) | Rolling average frame counter over 1,000 processed frames. |
| **3. Detection Confidence & Precision** | Accuracy in localizing tools, containers, and hand landmarks in payload workspace. | $\text{mAP@50} \ge 0.80$, Hand Keypoint PCK $\ge 85\%$ | $\text{mAP@50} \ge 0.92$, Hand Keypoint PCK $\ge 95\%$ | Evaluated against standard annotated benchmark test dataset. |
| **4. Event Delivery Delay** | Elapsed time from physical action completion to local audio alert / UI state update. | $\le 350\text{ ms}$ (includes 5-frame debounce window) | $\le 80\text{ ms}$ (satisfies proposal claim of $<100\text{ ms}$) | Video timestamp of action vs. system timestamp of event dispatch. |
| **5. Stream Reconnection Time** | Time required for the ingestion pipeline to re-acquire feed and resume state tracking after camera disconnect or signal drop. | $\le 1,500\text{ ms}$ | $\le 500\text{ ms}$ | Automated disconnect/reconnect simulation test harness. |
| **6. Telemetry Size Reduction** | Percentage reduction in transmitted data size comparing raw video to structured event JSON logs. | **$\ge 99.0\%$ reduction** (~1.2 KB/s telemetry vs ~12 Mbps 1080p stream) | **$\ge 99.5\%$ reduction** (compressed binary/JSON state stream) | Byte count comparison: `len(telemetry_stream) / len(raw_video_stream) * 100`. |
| **7. System Memory Footprint (RSS)** | Peak Resident Set Size (RAM) during sustained multi-minute pipeline execution. | $\le 1.8\text{ GB}$ (Leaves $>6\text{ GB}$ free on 8 GB laptop) | $\le 6.0\text{ GB}$ (including local LLM weights & KV cache) | Monitored via `psutil.Process().memory_info().rss`. |

---

### 2. Metric Breakdown & Measurement Methodologies

#### Metric 1: Inference Latency
- **Formula:** $T_{infer} = t_{post\_forward} - t_{pre\_preprocess}$
- **Measurement Protocol:** Run 500 warmup cycles followed by 2,000 timed cycles on standardized 640x480 resolution images. Log 50th, 95th, and 99th percentile latencies.
- **Laptop Target:** $P_{95} \le 120\text{ ms}$ on CPU using ONNX Runtime with multi-threading configured to available physical cores.
- **Jetson Target:** $P_{95} \le 25\text{ ms}$ using TensorRT execution engine with CUDA streams.

#### Metric 2: Effective Processing FPS and Frame Dropping
- **Ingestion Decoupling:** Ingestion runs asynchronously at 30 FPS. The perception pipeline queries the latest frame every $100\text{ ms}$ (10 FPS).
- **Zero-Lag Invariant:** Queued lag must remain exactly 0 frames. If perception falls behind, the ring buffer overwrites stale frames rather than buffering backlog.
- **Acceptance Criterion:** Over a 10-minute continuous run, video display remains synchronized with real-world time ($\Delta t_{lag} < 50\text{ ms}$).

#### Metric 3: Detection Confidence & Temporal Step Accuracy
- **Static Detection:** Object detection mAP (mean Average Precision at IoU=0.50) on 6 test classes: `[hand, vial_sealed, vial_open, pipette, tip_box, waste_bin]`.
- **Temporal Action Accuracy:** Percentage of correctly identified step transitions without false positives:
  $$\text{Accuracy}_{step} = \frac{\text{True Step Transitions Validated}}{\text{Total Ground Truth Step Transitions}} \ge 90\%$$
- **Hysteresis Stability:** A step is not triggered on single-frame spikes; $K \ge 5$ consecutive consensus frames are required.

#### Metric 4: Event Delivery Delay
- **Formula:** $T_{event} = T_{infer} \times K_{debounce} + T_{DAG\_eval} + T_{dispatch}$
- **Measurement Protocol:** Simulated action onset is injected into the pipeline; elapsed time until the alert event listener receives the message is recorded.
- **Laptop Budget Breakdown:**
  - Inference (5 frames @ 50 ms): ~250 ms
  - State DAG Transition: ~5 ms
  - IPC / SSE UI Dispatch: ~15 ms
  - Audio Trigger Initiation: ~30 ms
  - **Total Laptop Delay:** $\mathbf{\approx 300\text{ ms}}$, well within acceptable human operator reaction feedback limits.

#### Metric 5: Stream Reconnection Time
- **Fault Tolerance Test:**
  1. Pipeline actively tracks Step 2 of an experiment.
  2. Camera connection is severed (simulated hardware failure or unplugged USB).
  3. Reconnection daemon detects `cv2.VideoCapture` failure within $200\text{ ms}$.
  4. System attempts exponential backoff reconnection.
  5. Upon camera re-acquisition, the state engine preserves the last confirmed protocol state without resetting to `STATE_0_IDLE`.
- **Target:** Recovery and resumption of valid state within $\le 1.5\text{ seconds}$ on the laptop prototype.

#### Metric 6: Telemetry Bandwidth Reduction
- **Baseline Raw Video Stream:**
  $1080\text{p} \ (1920 \times 1080) \ \text{at} \ 30\text{ FPS, H.264 encoded at 6 Mbps to 10 Mbps}$.
  - Over a 10-minute experiment: $\approx 600\text{ MB}$ data volume.
- **SUTRA Compressed Event Telemetry:**
  - State change records: ~120 bytes per JSON record.
  - Telemetry heartbeats: ~80 bytes emitted at 1 Hz.
  - Total over 10-minute experiment: $10 \times 60 \times 80\text{ bytes} + 20\text{ events} \times 120\text{ bytes} \approx 50.4\text{ KB}$.
- **Net Bandwidth Reduction:**
  $$\text{Reduction} = \left( 1 - \frac{50.4\text{ KB}}{600\text{ MB}} \right) \times 100\% \approx \mathbf{99.99\% \text{ bandwidth savings}}$$
  *(Comfortably surpasses the proposal target of $>95\%$).*

---

### 3. Automated Benchmark Harness Plan

The project test suite will include a dedicated benchmark harness (`tests/benchmark_pipeline.py`) capable of running on any development machine to generate a standardized report:

```
================================================================================
                    SUTRA BENCHMARK HARNESS REPORT
================================================================================
Target Platform: Windows x86_64 (CPU Only)
RAM Available: 7.82 GB | Physical Cores: 6
Model: YOLO11n-ONNX (Quantized INT8) + MediaPipe Hands
--------------------------------------------------------------------------------
Metric                              Measured        Target (Laptop)    Status
--------------------------------------------------------------------------------
Inference Latency (Avg)             84.2 ms         <= 120 ms          PASS
Inference Latency (P95)             108.5 ms        <= 150 ms          PASS
Processing Throughput               11.2 FPS        >= 8.0 FPS         PASS
Event Delivery Delay                312 ms          <= 400 ms          PASS
Memory Footprint (RSS Peak)         642 MB          <= 1800 MB         PASS
Telemetry Reduction Ratio           99.98%          >= 95.00%          PASS
State Machine Transition Integrity  100.0%          100.0%             PASS
================================================================================
```

