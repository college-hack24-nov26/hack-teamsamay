"""
Real-Time Browser-Based Live Video HUD for SUTRA Vision Engine.
Provides an MJPEG live video stream and telemetry dashboard viewable in any browser.
Works 100% reliably even on headless environments without native GUI dependencies.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SutraConfig, get_config
from app.engine import SutraVisionEngine

logger = logging.getLogger("sutra.web_viewer")

app = FastAPI(title="SUTRA Real-Time Experiment Guardian HUD")

engine_instance: Optional[SutraVisionEngine] = None
latest_annotated_jpeg: Optional[bytes] = None
latest_telemetry_events: list[dict] = []


HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SUTRA AI - On-Board BAS Experiment Guardian</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d1117;
      color: #c9d1d9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 16px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 12px;
      border-bottom: 1px solid #30363d;
      margin-bottom: 16px;
    }
    h1 { font-size: 20px; color: #58a6ff; }
    .badge {
      background: #238636;
      color: #fff;
      padding: 4px 8px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: bold;
    }
    .container {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 16px;
    }
    .video-card, .sidebar-card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 12px;
    }
    .video-container {
      position: relative;
      width: 100%;
      background: #000;
      border-radius: 6px;
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 400px;
    }
    .video-container img {
      width: 100%;
      height: auto;
      max-height: 520px;
      object-fit: contain;
    }
    .hud-controls {
      display: flex;
      gap: 10px;
      margin-top: 12px;
    }
    button {
      background: #21262d;
      color: #c9d1d9;
      border: 1px solid #30363d;
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-weight: bold;
    }
    button:hover { background: #30363d; }
    button.alert-btn { background: #da3633; color: white; border: none; }
    .event-log {
      max-height: 480px;
      overflow-y: auto;
      font-family: monospace;
      font-size: 12px;
      display: flex;
      flex-direction: column-reverse;
      gap: 8px;
    }
    .event-item {
      background: #0d1117;
      border-left: 3px solid #58a6ff;
      padding: 8px;
      border-radius: 4px;
    }
    .event-item.DEVIATION_DETECTED {
      border-left-color: #f85149;
      background: #220808;
    }
    .event-item.STEP_COMPLETED {
      border-left-color: #2ea043;
    }
    .metric-row {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 13px;
    }
    .metric-val { color: #58a6ff; font-weight: bold; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>SUTRA - AI Human Activity Recognition (SIH26174)</h1>
      <small style="color: #8b949e">Autonomous Edge Guardian for Bharatiya Antariksh Station (BAS)</small>
    </div>
    <div>
      <span class="badge" id="status-badge">LIVE EDGE STREAM</span>
    </div>
  </header>

  <div class="container">
    <div class="video-card">
      <div class="video-container">
        <img id="stream-img" src="/video_feed" alt="Live AI Video Stream">
      </div>
      <div class="hud-controls">
        <button onclick="ackDeviation()" class="alert-btn">Acknowledge Deviation</button>
        <button onclick="fetchEvents()">Refresh Logs</button>
        <span id="fps-label" style="margin-left: auto; align-self: center; font-size: 13px; color: #8b949e">FPS: --</span>
      </div>
    </div>

    <div class="sidebar-card">
      <h3 style="margin-bottom: 12px; font-size: 15px; border-bottom: 1px solid #30363d; padding-bottom: 6px;">Live Telemetry Events</h3>
      <div class="metric-row">
        <span>Latency P95:</span>
        <span class="metric-val" id="p95-val">-- ms</span>
      </div>
      <div class="metric-row">
        <span>Memory RSS:</span>
        <span class="metric-val" id="ram-val">-- MB</span>
      </div>
      <div class="metric-row">
        <span>Active Step:</span>
        <span class="metric-val" id="step-val">STEP_01</span>
      </div>

      <div class="event-log" id="event-log">
        <div class="event-item">Awaiting live events...</div>
      </div>
    </div>
  </div>

  <script>
    async function ackDeviation() {
      await fetch('/api/acknowledge', { method: 'POST' });
    }

    async function fetchEvents() {
      try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();
        const container = document.getElementById('event-log');
        container.innerHTML = '';
        data.events.slice(-15).reverse().forEach(evt => {
          const div = document.createElement('div');
          div.className = 'event-item ' + evt.event_type;
          div.innerHTML = `<strong>[${evt.event_type}]</strong> ${evt.step_name || evt.step_id || ''}<br><span style="color:#8b949e">${evt.timestamp.split('T')[1].split('.')[0]}</span> - Conf: ${evt.confidence}`;
          container.appendChild(div);
        });
        document.getElementById('p95-val').innerText = data.diagnostics.latency_p95_ms + ' ms';
        document.getElementById('ram-val').innerText = data.diagnostics.memory_rss_mb + ' MB';
        document.getElementById('fps-label').innerText = 'Processed: ' + data.diagnostics.processed_frames + ' frames';
      } catch (e) {}
    }
    setInterval(fetchEvents, 1000);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML_DASHBOARD)


def generate_mjpeg_stream():
    """Generator yielding JPEG frames for browser streaming."""
    global latest_annotated_jpeg
    while True:
        if latest_annotated_jpeg is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + latest_annotated_jpeg + b"\r\n")
        time.sleep(0.04)  # ~25 FPS stream rate


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/telemetry")
def get_telemetry():
    global engine_instance
    diag = engine_instance.get_diagnostics() if engine_instance else {}
    events = engine_instance.storage.read_events(limit=25) if engine_instance else []
    return {
        "diagnostics": diag,
        "events": [e.model_dump() for e in events]
    }


@app.post("/api/acknowledge")
def acknowledge():
    global engine_instance
    if engine_instance:
        engine_instance.protocol_engine.acknowledge_deviation()
    return {"status": "ACKNOWLEDGED"}


def start_web_server(engine: SutraVisionEngine, host: str = "127.0.0.1", port: int = 8000):
    """
    Launch background thread running the engine loop, and start uvicorn web server.
    """
    global engine_instance, latest_annotated_jpeg
    engine_instance = engine
    engine.start()

    import threading

    def pipeline_worker():
        global latest_annotated_jpeg
        while True:
            try:
                ingested = engine.frame_buffer.get_latest(timeout=0.5)
                if ingested is None:
                    if not engine.ingestor.is_connected:
                        time.sleep(0.2)
                        continue
                    else:
                        break

                res = engine.process_single_frame(ingested, annotate=True)
                if res.annotated_frame is not None:
                    ret, buf = cv2.imencode(".jpg", res.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        latest_annotated_jpeg = buf.tobytes()
            except Exception as e:
                logger.error(f"Error in pipeline worker: {e}")
                time.sleep(0.05)

    t = threading.Thread(target=pipeline_worker, name="WebHUDWorker", daemon=True)
    t.start()

    print("\n" + "=" * 65)
    print(" [✓] SUTRA LIVE WEB HUD IS RUNNING!")
    print(f" [✓] Open your browser at: http://{host}:{port}")
    print("=" * 65 + "\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    conf = get_config()
    engine = SutraVisionEngine(conf)
    start_web_server(engine, host="127.0.0.1", port=8000)
