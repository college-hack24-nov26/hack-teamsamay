"""
Real-Time Browser-Based Live Video HUD for SUTRA Vision Engine.
Provides high-performance, smooth frame streaming via HTML5 Canvas with modern White/Light theme,
theme toggle (Light/Dark), snapshot capture, and instant operator controls.
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
import numpy as np
from fastapi import FastAPI, Response
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


HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SUTRA AI - On-Board BAS Experiment Guardian</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --text-muted: #64748b;
      --primary: #2563eb;
      --success: #10b981;
      --danger: #ef4444;
      --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }
    body.dark {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --success: #34d399;
      --danger: #f87171;
      --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 18px 24px;
      transition: background 0.3s ease, color 0.3s ease;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 20px;
    }
    .brand-title {
      font-size: 22px;
      font-weight: 800;
      color: var(--primary);
      letter-spacing: -0.5px;
    }
    .brand-subtitle {
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 2px;
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .badge {
      background: var(--success);
      color: #ffffff;
      padding: 5px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 6px;
      box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      background: #ffffff;
      border-radius: 50%;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.8); }
      100% { opacity: 1; transform: scale(1); }
    }
    .container {
      display: grid;
      grid-template-columns: 2.2fr 1fr;
      gap: 20px;
    }
    @media (max-width: 960px) {
      .container { grid-template-columns: 1fr; }
    }
    .video-card, .sidebar-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      box-shadow: var(--shadow);
      transition: background 0.3s ease, border-color 0.3s ease;
    }
    .video-container {
      position: relative;
      width: 100%;
      background: #000;
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 440px;
      border: 1px solid var(--border);
    }
    #video-canvas {
      width: 100%;
      height: auto;
      max-height: 540px;
      object-fit: contain;
      display: block;
      border-radius: 6px;
    }
    .hud-controls {
      display: flex;
      gap: 10px;
      margin-top: 14px;
      align-items: center;
      flex-wrap: wrap;
    }
    button {
      background: var(--card-bg);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 9px 16px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      font-size: 13px;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    button:hover {
      background: var(--border);
      transform: translateY(-1px);
    }
    button.primary-btn {
      background: var(--primary);
      color: #ffffff;
      border: none;
    }
    button.primary-btn:hover {
      filter: brightness(1.1);
    }
    button.alert-btn {
      background: var(--danger);
      color: #ffffff;
      border: none;
    }
    button.alert-btn:hover {
      filter: brightness(1.15);
    }
    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 0;
      border-bottom: 1px solid var(--border);
      font-size: 13px;
    }
    .metric-row:last-child {
      border-bottom: none;
    }
    .metric-val {
      color: var(--primary);
      font-weight: 700;
      font-family: monospace;
      font-size: 14px;
    }
    .event-log {
      max-height: 420px;
      overflow-y: auto;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
      font-size: 12px;
      display: flex;
      flex-direction: column-reverse;
      gap: 8px;
      margin-top: 12px;
      padding-right: 4px;
    }
    .event-item {
      background: var(--bg);
      border-left: 4px solid var(--primary);
      padding: 10px 12px;
      border-radius: 6px;
      border: 1px solid var(--border);
      border-left-width: 4px;
      transition: all 0.2s ease;
    }
    .event-item.DEVIATION_DETECTED {
      border-left-color: var(--danger);
      background: rgba(239, 68, 68, 0.08);
    }
    .event-item.STEP_COMPLETED {
      border-left-color: var(--success);
      background: rgba(16, 185, 129, 0.08);
    }
    .card-heading {
      font-size: 15px;
      font-weight: 700;
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="brand-title">SUTRA AI - On-Board Experiment Guardian</div>
      <div class="brand-subtitle">Smart India Hackathon 2026 (SIH26174) &bull; Bharatiya Antariksh Station</div>
    </div>
    <div class="header-actions">
      <button onclick="toggleTheme()" id="theme-btn">🌙 Dark Mode</button>
      <div class="badge" id="status-badge">
        <span class="pulse-dot"></span>
        <span>EDGE ACTIVE</span>
      </div>
    </div>
  </header>

  <div class="container">
    <!-- Left: Video Stream & Main Controls -->
    <div class="video-card">
      <div class="video-container">
        <canvas id="video-canvas" width="640" height="480"></canvas>
      </div>
      <div class="hud-controls">
        <button onclick="ackDeviation()" class="alert-btn">&#x2714; Acknowledge Deviation</button>
        <button onclick="downloadSnapshot()">&#128247; Save Snapshot</button>
        <button onclick="fetchEvents()">&#8635; Refresh</button>
        <span id="fps-label" style="margin-left: auto; font-size: 13px; font-weight: 600; color: var(--text-muted)">Live Feed: 30 FPS</span>
      </div>
    </div>

    <!-- Right: Telemetry & Diagnostic Metrics -->
    <div class="sidebar-card">
      <div class="card-heading">
        <span>Real-Time Diagnostics</span>
        <small style="color: var(--text-muted); font-size: 11px;">CPU-First</small>
      </div>
      
      <div class="metric-row">
        <span>Inference Latency (P95):</span>
        <span class="metric-val" id="p95-val">-- ms</span>
      </div>
      <div class="metric-row">
        <span>Memory Usage (RSS):</span>
        <span class="metric-val" id="ram-val">-- MB</span>
      </div>
      <div class="metric-row">
        <span>Active Protocol Step:</span>
        <span class="metric-val" id="step-val" style="color: var(--primary);">STEP_01</span>
      </div>

      <div class="card-heading" style="margin-top: 14px;">
        <span>Telemetry State Events</span>
        <small id="event-count-badge" style="color: var(--primary); font-weight: bold;">0 logged</small>
      </div>

      <div class="event-log" id="event-log">
        <div class="event-item">Awaiting real-time events...</div>
      </div>
    </div>
  </div>

  <script>
    const canvas = document.getElementById('video-canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    let isFetching = false;
    let frameCount = 0;
    let lastFpsTime = performance.now();

    img.onload = function() {
      if (img.width && img.height) {
        if (canvas.width !== img.width || canvas.height !== img.height) {
          canvas.width = img.width;
          canvas.height = img.height;
        }
        ctx.drawImage(img, 0, 0);
      }
      isFetching = false;
      frameCount++;
      const now = performance.now();
      if (now - lastFpsTime >= 1000) {
        document.getElementById('fps-label').innerText = 'Live Feed: ' + frameCount + ' FPS';
        frameCount = 0;
        lastFpsTime = now;
      }
    };

    img.onerror = function() {
      isFetching = false;
    };

    function scheduleNextFrame() {
      if (!isFetching) {
        isFetching = true;
        img.src = '/api/frame?t=' + Date.now();
      }
      setTimeout(scheduleNextFrame, 33);
    }
    scheduleNextFrame();

    async function ackDeviation() {
      await fetch('/api/acknowledge', { method: 'POST' });
    }

    function downloadSnapshot() {
      const link = document.createElement('a');
      link.download = 'sutra_snapshot_' + Date.now() + '.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
    }

    function toggleTheme() {
      const isDark = document.body.classList.toggle('dark');
      document.getElementById('theme-btn').innerText = isDark ? '☀️ Light Mode' : '🌙 Dark Mode';
    }

    async function fetchEvents() {
      try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();
        const container = document.getElementById('event-log');
        if (data.events && data.events.length > 0) {
          container.innerHTML = '';
          document.getElementById('event-count-badge').innerText = data.events.length + ' logged';
          data.events.slice(-15).reverse().forEach(evt => {
            const div = document.createElement('div');
            div.className = 'event-item ' + evt.event_type;
            const timeStr = evt.timestamp.includes('T') ? evt.timestamp.split('T')[1].split('.')[0] : evt.timestamp;
            div.innerHTML = `<strong>[${evt.event_type}]</strong> ${evt.step_name || evt.step_id || ''}<br><span style="color:var(--text-muted)">${timeStr}</span> &bull; Conf: ${evt.confidence}`;
            container.appendChild(div);
          });
        }
        if (data.diagnostics) {
          document.getElementById('p95-val').innerText = (data.diagnostics.latency_p95_ms || 0) + ' ms';
          document.getElementById('ram-val').innerText = (data.diagnostics.memory_rss_mb || 0) + ' MB';
          if (data.events && data.events.length > 0) {
            const latest = data.events[data.events.length - 1];
            document.getElementById('step-val').innerText = latest.step_id || 'STEP_01';
          }
        }
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


@app.get("/api/frame")
def get_current_frame():
    """Returns the latest annotated JPEG frame directly with no-cache headers."""
    global latest_annotated_jpeg
    if latest_annotated_jpeg is not None:
        return Response(
            content=latest_annotated_jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    blank = np.full((480, 640, 3), 240, dtype=np.uint8)
    cv2.putText(blank, "INITIALIZING CAMERA FEED...", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    ret, buf = cv2.imencode(".jpg", blank)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


def generate_mjpeg_stream():
    """Generator yielding JPEG frames for browser streaming."""
    global latest_annotated_jpeg
    while True:
        if latest_annotated_jpeg is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(latest_annotated_jpeg)).encode() + b"\r\n\r\n"
                + latest_annotated_jpeg + b"\r\n"
            )
        time.sleep(0.04)


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
                    ret, buf = cv2.imencode(".jpg", res.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        latest_annotated_jpeg = buf.tobytes()
            except Exception as e:
                logger.error(f"Error in pipeline worker: {e}")
                time.sleep(0.05)

    t = threading.Thread(target=pipeline_worker, name="WebHUDWorker", daemon=True)
    t.start()

    print("\n" + "=" * 65)
    print(" [OK] SUTRA LIVE WEB HUD IS RUNNING!")
    print(f" [OK] Open your browser at: http://{host}:{port}")
    print("=" * 65 + "\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    conf = get_config()
    engine = SutraVisionEngine(conf)
    start_web_server(engine, host="127.0.0.1", port=8000)
