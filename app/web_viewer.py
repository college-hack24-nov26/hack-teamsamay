"""
Real-Time Browser-Based Live Video HUD for SUTRA Vision Engine.
Includes:
- HTML5 Canvas high-performance real-time video stream (30 FPS)
- Mobile IP Webcam & USB Webcam runtime switching
- Groq LLM Flight Assistant ("Kya ho raha hai? / Kya missing hai?")
- Voice Alerts (Speech Synthesis) for step updates & deviation warnings
- Light / Dark Theme toggle & Snapshot capture
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
from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SutraConfig, get_config
from app.engine import SutraVisionEngine
from modules.assistant.groq_assistant import SutraExperimentAssistant

logger = logging.getLogger("sutra.web_viewer")

app = FastAPI(title="SUTRA Real-Time Experiment Guardian HUD")

engine_instance: Optional[SutraVisionEngine] = None
latest_annotated_jpeg: Optional[bytes] = None
assistant_instance = SutraExperimentAssistant()


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
      --warning: #f59e0b;
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
      --warning: #fbbf24;
      --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 16px 24px;
      transition: background 0.3s ease, color 0.3s ease;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .brand-title {
      font-size: 20px;
      font-weight: 800;
      color: var(--primary);
      letter-spacing: -0.5px;
    }
    .brand-subtitle {
      font-size: 12px;
      color: var(--text-muted);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
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
    
    /* Camera Source Switcher Bar */
    .source-bar {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 14px;
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      box-shadow: var(--shadow);
    }
    .source-label {
      font-size: 13px;
      font-weight: 700;
      color: var(--text-muted);
    }
    .source-btn {
      padding: 6px 12px;
      font-size: 12px;
      border-radius: 6px;
    }
    .source-btn.active {
      background: var(--primary);
      color: #fff;
      border-color: var(--primary);
    }
    .ip-input {
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 12px;
      width: 240px;
    }

    .container {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 16px;
    }
    @media (max-width: 1000px) {
      .container { grid-template-columns: 1fr; }
    }
    .video-card, .sidebar-card, .assistant-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      box-shadow: var(--shadow);
      transition: background 0.3s ease;
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
      max-height: 520px;
      object-fit: contain;
      display: block;
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
      padding: 8px 14px;
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
    button.alert-btn {
      background: var(--danger);
      color: #ffffff;
      border: none;
    }
    button.voice-btn.active {
      background: #8b5cf6;
      color: white;
      border-color: #8b5cf6;
    }
    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 0;
      border-bottom: 1px solid var(--border);
      font-size: 13px;
    }
    .metric-row:last-child { border-bottom: none; }
    .metric-val {
      color: var(--primary);
      font-weight: 700;
      font-family: monospace;
      font-size: 13px;
    }
    .event-log {
      max-height: 220px;
      overflow-y: auto;
      font-size: 12px;
      display: flex;
      flex-direction: column-reverse;
      gap: 6px;
      margin-top: 10px;
    }
    .event-item {
      background: var(--bg);
      border-left: 4px solid var(--primary);
      padding: 8px 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      border-left-width: 4px;
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
      font-size: 14px;
      font-weight: 700;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    /* Assistant Chat Styling */
    .assistant-card {
      margin-top: 16px;
    }
    .chat-output {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      min-height: 80px;
      max-height: 160px;
      overflow-y: auto;
      font-size: 13px;
      line-height: 1.5;
      margin-bottom: 10px;
    }
    .chat-input-row {
      display: flex;
      gap: 8px;
    }
    .chat-input {
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 13px;
    }
    .quick-chips {
      display: flex;
      gap: 6px;
      margin-top: 8px;
      flex-wrap: wrap;
    }
    .chip {
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 4px 10px;
      border-radius: 14px;
      font-size: 11px;
      cursor: pointer;
    }
    .chip:hover {
      border-color: var(--primary);
      color: var(--primary);
    }
    
    /* Help Modal */
    .modal-overlay {
      display: none;
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0,0,0,0.6);
      z-index: 999;
      justify-content: center;
      align-items: center;
    }
    .modal-content {
      background: var(--card-bg);
      color: var(--text);
      padding: 24px;
      border-radius: 12px;
      max-width: 500px;
      width: 90%;
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
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
      <button onclick="toggleVoice()" id="voice-btn" class="voice-btn active">&#128266; Voice: ON</button>
      <button onclick="toggleTheme()" id="theme-btn">&#9790; Dark Mode</button>
      <button onclick="openHelpModal()">&#10067; Phone Guide</button>
      <div class="badge" id="status-badge">
        <span class="pulse-dot"></span>
        <span id="badge-text">EDGE LIVE</span>
      </div>
    </div>
  </header>

  <!-- Video Source Switcher -->
  <div class="source-bar">
    <span class="source-label">&#128249; Video Source:</span>
    <button class="source-btn active" onclick="switchSource('0', this)">&#128187; Laptop Webcam</button>
    <button class="source-btn" onclick="switchSource('synthetic', this)">&#129514; 5-Step Simulation</button>
    <button class="source-btn" onclick="switchSource('data/sample_deviation_run.mp4', this)">&#9888; Deviation Replay</button>
    <input type="text" id="ipwebcam-url" class="ip-input" placeholder="http://192.168.X.X:8080/video">
    <button class="source-btn" onclick="connectIpWebcam()">&#128241; Connect Mobile</button>
  </div>

  <div class="container">
    <!-- Left: Video Stream -->
    <div class="video-card">
      <div class="video-container">
        <canvas id="video-canvas" width="640" height="480"></canvas>
      </div>
      <div class="hud-controls">
        <button onclick="ackDeviation()" class="alert-btn">&#10004; Acknowledge Deviation</button>
        <button onclick="downloadSnapshot()">&#128247; Snapshot</button>
        <button onclick="fetchEvents()">&#8635; Refresh</button>
        <span id="fps-label" style="margin-left: auto; font-size: 13px; font-weight: 600; color: var(--text-muted)">Live Feed: 30 FPS</span>
      </div>

      <!-- Groq / Flight Assistant Section -->
      <div class="assistant-card" style="margin-top: 14px; padding: 14px; background: var(--bg);">
        <div class="card-heading">
          <span>&#129302; SUTRA Flight Guidance Assistant (Groq LLM)</span>
          <button onclick="speakLastReply()" style="padding: 4px 8px; font-size: 11px;">&#128266; Read Aloud</button>
        </div>
        <div class="chat-output" id="chat-output">
          &#128075; <strong>Namaste Astronaut!</strong> Main SUTRA assistant hoon. Camera ke saamne jo bhi ho raha hai, main analyze kar sakta hoon. Poochiye: "Kya ho raha hai?" ya "Kya missing hai?"
        </div>
        <div class="chat-input-row">
          <input type="text" id="chat-input" class="chat-input" placeholder="Poochiye: Kya missing hai? Ya next kya karna chahiye?..." onkeydown="if(event.key==='Enter') sendChat()">
          <button onclick="sendChat()" style="background: var(--primary); color: white; border: none;">Poochein</button>
        </div>
        <div class="quick-chips">
          <span class="chip" onclick="askChip('Kya ho raha hai abhi?')">&#128269; Kya ho raha hai?</span>
          <span class="chip" onclick="askChip('Koi mistake ya missing step hai kya?')">&#9888; Kya missing hai?</span>
          <span class="chip" onclick="askChip('Next step kya karna chahiye?')">&#10145; Next kya karna hai?</span>
          <span class="chip" onclick="askChip('Is experiment step ke rules kya hain?')">&#128220; Rules kya hain?</span>
        </div>
      </div>
    </div>

    <!-- Right: Telemetry & State -->
    <div class="sidebar-card">
      <div class="card-heading">
        <span>Real-Time Edge Telemetry</span>
        <small style="color: var(--text-muted); font-size: 11px;">CPU-First</small>
      </div>
      
      <div class="metric-row">
        <span>Inference Latency:</span>
        <span class="metric-val" id="p95-val">-- ms</span>
      </div>
      <div class="metric-row">
        <span>Process RAM:</span>
        <span class="metric-val" id="ram-val">-- MB</span>
      </div>
      <div class="metric-row">
        <span>Current Protocol Step:</span>
        <span class="metric-val" id="step-val" style="color: var(--primary);">STEP_01</span>
      </div>
      <div class="metric-row">
        <span>Step Description:</span>
        <span id="step-name-val" style="font-size: 12px; font-weight: 600;">Workspace Inspection</span>
      </div>

      <div class="card-heading" style="margin-top: 14px;">
        <span>Telemetry State Events</span>
        <small id="event-count-badge" style="color: var(--primary); font-weight: bold;">0 logged</small>
      </div>

      <div class="event-log" id="event-log">
        <div class="event-item">Connecting to video feed...</div>
      </div>
    </div>
  </div>

  <!-- Mobile IP Webcam Help Modal -->
  <div class="modal-overlay" id="help-modal" onclick="if(event.target===this) closeHelpModal()">
    <div class="modal-content">
      <h3 style="margin-bottom: 12px; color: var(--primary);">&#128241; Mobile Phone ko Camera Kaise Banayein</h3>
      <ol style="padding-left: 20px; line-height: 1.6; font-size: 13px;">
        <li>Apne Android Phone mein Play Store se <strong>"IP Webcam"</strong> app install karein.</li>
        <li>Phone aur Laptop dono ko <strong>same Wi-Fi ya Mobile Hotspot</strong> se connect karein.</li>
        <li>Phone mein IP Webcam app open karein, neeche scroll karein aur <strong>"Start server"</strong> par tap karein.</li>
        <li>Phone ki screen par IP dikhegi (jaise: <code>http://192.168.1.15:8080</code>).</li>
        <li>Upar dashboard ke input box mein <code>/video</code> lagakar likhein: <br><strong style="color: var(--primary);">http://192.168.1.15:8080/video</strong></li>
        <li><strong>"Connect Mobile"</strong> button dabayein! Live phone camera stream shuru ho jayegi!</li>
      </ol>
      <button onclick="closeHelpModal()" style="margin-top: 16px; width: 100%; justify-content: center; background: var(--primary); color: white; border: none;">Samajh Gaya (Close)</button>
    </div>
  </div>

  <script>
    const canvas = document.getElementById('video-canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    let isFetching = false;
    let frameCount = 0;
    let lastFpsTime = performance.now();
    let voiceEnabled = true;
    let lastSpokenEventId = null;
    let lastReplyText = '';

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

    // Continuous smooth frame fetch loop
    function scheduleNextFrame() {
      if (!isFetching) {
        isFetching = true;
        img.src = '/api/frame?t=' + Date.now();
      }
      setTimeout(scheduleNextFrame, 33);
    }
    scheduleNextFrame();

    // Voice Text-to-Speech Engine
    function speakText(text) {
      if (!voiceEnabled || !window.speechSynthesis) return;
      window.speechSynthesis.cancel();
      const clean = text.replace(/[*_#`]/g, '');
      const utterance = new SpeechSynthesisUtterance(clean);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }

    function toggleVoice() {
      voiceEnabled = !voiceEnabled;
      const btn = document.getElementById('voice-btn');
      btn.className = 'voice-btn ' + (voiceEnabled ? 'active' : '');
      btn.innerHTML = voiceEnabled ? '&#128266; Voice: ON' : '&#128263; Voice: OFF';
      if (!voiceEnabled && window.speechSynthesis) window.speechSynthesis.cancel();
    }

    function speakLastReply() {
      if (lastReplyText) speakText(lastReplyText);
    }

    async function ackDeviation() {
      await fetch('/api/acknowledge', { method: 'POST' });
      speakText("Deviation acknowledged by operator.");
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

    function openHelpModal() { document.getElementById('help-modal').style.display = 'flex'; }
    function closeHelpModal() { document.getElementById('help-modal').style.display = 'none'; }

    // Switch Video Source API
    async function switchSource(src, btnEl) {
      document.querySelectorAll('.source-btn').forEach(b => b.classList.remove('active'));
      if (btnEl) btnEl.classList.add('active');
      document.getElementById('badge-text').innerText = 'SWITCHING...';
      try {
        await fetch('/api/switch_source', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ source: src })
        });
        document.getElementById('badge-text').innerText = 'EDGE LIVE';
        speakText("Video source switched.");
      } catch (e) {
        alert("Failed to switch video source: " + e);
      }
    }

    function connectIpWebcam() {
      let url = document.getElementById('ipwebcam-url').value.trim();
      if (!url) {
        alert("Pehle phone ka IP URL dalein (jaise: http://192.168.1.15:8080/video)");
        return;
      }
      if (!url.endsWith('/video') && !url.includes('?')) {
        url = url.replace(/\\/$/, '') + '/video';
      }
      switchSource(url, null);
    }

    // Assistant Chat
    async function sendChat() {
      const input = document.getElementById('chat-input');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      const output = document.getElementById('chat-output');
      output.innerHTML = '<em>Thinking and analyzing active experiment...</em>';

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: text })
        });
        const data = await res.json();
        lastReplyText = data.reply;
        output.innerHTML = data.reply.replace(/\\n/g, '<br>');
        speakText(data.reply);
      } catch (e) {
        output.innerHTML = 'Error calling assistant: ' + e;
      }
    }

    function askChip(prompt) {
      document.getElementById('chat-input').value = prompt;
      sendChat();
    }

    // Telemetry Polling & Voice Announcements
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

          // Check for new state changes to speak aloud
          const latest = data.events[data.events.length - 1];
          if (latest && latest.event_id !== lastSpokenEventId) {
            lastSpokenEventId = latest.event_id;
            if (latest.event_type === 'DEVIATION_DETECTED') {
              speakText("Warning! Procedural deviation detected. " + (latest.evidence?.active_activity || "Please check sequence."));
            } else if (latest.event_type === 'STEP_COMPLETED') {
              speakText("Step completed: " + (latest.step_name || latest.step_id));
            } else if (latest.event_type === 'STEP_STARTED') {
              speakText("Step started: " + (latest.step_name || latest.step_id));
            }
          }
        }
        if (data.diagnostics) {
          document.getElementById('p95-val').innerText = (data.diagnostics.latency_p95_ms || 0) + ' ms';
          document.getElementById('ram-val').innerText = (data.diagnostics.memory_rss_mb || 0) + ' MB';
          if (data.events && data.events.length > 0) {
            const latest = data.events[data.events.length - 1];
            document.getElementById('step-val').innerText = latest.step_id || 'STEP_01';
            document.getElementById('step-name-val').innerText = latest.step_name || 'In Progress';
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
    blank = np.full((480, 640, 3), 245, dtype=np.uint8)
    cv2.putText(blank, "INITIALIZING CAMERA FEED...", (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
    ret, buf = cv2.imencode(".jpg", blank)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


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


@app.post("/api/switch_source")
async def switch_source(request: Request):
    """Switch camera source at runtime (Webcam, IP Webcam phone, or Synthetic)."""
    global engine_instance
    body = await request.json()
    src = body.get("source", "0")
    if str(src).isdigit():
        src_val = int(src)
    elif src == "synthetic":
        from scripts.generate_synthetic_video import create_synthetic_frame_generator
        src_val = create_synthetic_frame_generator(simulate_deviation=False)
    else:
        src_val = str(src)

    if engine_instance and engine_instance.ingestor:
        engine_instance.ingestor.switch_source(src_val)
        logger.info(f"Video source switched to: {src_val}")

    return {"status": "SWITCHED", "source": str(src)}


@app.post("/api/chat")
async def chat_assistant(request: Request):
    """Interact with SUTRA Flight Guidance Assistant (Groq LLM)."""
    global engine_instance, assistant_instance
    body = await request.json()
    prompt = body.get("prompt", "")

    # Gather live telemetry context
    ctx = {}
    if engine_instance:
        pe = engine_instance.protocol_engine
        cur_step = pe.protocol.steps[pe.current_step_idx] if pe.current_step_idx < len(pe.protocol.steps) else None
        completed = [sid for sid, st in pe.step_statuses.items() if st.value == "COMPLETED"]
        ctx = {
            "protocol_name": pe.protocol.protocol_name,
            "current_step_id": cur_step.id if cur_step else "COMPLETED",
            "current_step_name": cur_step.name if cur_step else "Experiment Finished",
            "step_status": pe.step_statuses.get(cur_step.id, "COMPLETED").value if cur_step else "COMPLETED",
            "completed_steps": completed,
            "active_deviation": pe.active_deviation,
        }

    reply = assistant_instance.ask_assistant(prompt, ctx)
    return {"reply": reply}


@app.post("/api/set_groq_key")
async def set_groq_key(request: Request):
    global assistant_instance
    body = await request.json()
    key = body.get("api_key", "")
    success = assistant_instance.set_api_key(key)
    return {"success": success}


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
        # Continuous loop that NEVER breaks on empty frames
        while engine._running:
            try:
                ingested = engine.frame_buffer.get_latest(timeout=0.2)
                if ingested is None:
                    time.sleep(0.04)
                    continue

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
