"""
Real-Time Browser-Based Live Video HUD for SUTRA Vision Engine.
Includes:
- HTML5 Canvas high-performance real-time video stream (30 FPS)
- Autonomous Flight Copilot Engine (Zero typing - automatically observes, guides & speaks)
- Mobile IP Webcam & USB Webcam runtime switching
- Hands-Free Voice Directives (Speech Synthesis) for step updates & deviation warnings
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
from fastapi.responses import HTMLResponse
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SutraConfig, get_config
from app.engine import SutraVisionEngine
from modules.assistant.copilot_engine import AutonomousCopilotEngine, CopilotDirective

logger = logging.getLogger("sutra.web_viewer")

app = FastAPI(title="SUTRA Autonomous BAS Experiment Guardian HUD")

engine_instance: Optional[SutraVisionEngine] = None
latest_annotated_jpeg: Optional[bytes] = None
copilot_engine = AutonomousCopilotEngine()
latest_directive: Optional[CopilotDirective] = None


HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SUTRA AI - Autonomous BAS Experiment Guardian</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --text-muted: #64748b;
      --primary: #2563eb;
      --primary-light: #eff6ff;
      --success: #10b981;
      --success-light: #ecfdf5;
      --danger: #ef4444;
      --danger-light: #fef2f2;
      --warning: #f59e0b;
      --warning-light: #fffbeb;
      --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }
    body.dark {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-light: #082f49;
      --success: #34d399;
      --success-light: #064e3b;
      --danger: #f87171;
      --danger-light: #450a0a;
      --warning: #fbbf24;
      --warning-light: #451a03;
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
      border: 1px solid var(--border);
      background: var(--card-bg);
      color: var(--text);
      cursor: pointer;
      font-weight: 600;
      transition: all 0.2s ease;
    }
    .source-btn:hover {
      background: var(--border);
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
      width: 250px;
    }

    .container {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 16px;
    }
    @media (max-width: 1000px) {
      .container { grid-template-columns: 1fr; }
    }
    .video-card, .sidebar-card, .copilot-card {
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

    /* Autonomous Copilot Section */
    .copilot-card {
      margin-top: 16px;
      border-left: 5px solid var(--primary);
    }
    .copilot-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }
    .copilot-status-pill {
      font-size: 11px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 12px;
      background: var(--primary-light);
      color: var(--primary);
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .directive-box {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 10px;
    }
    .directive-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      margin-bottom: 4px;
    }
    .directive-text {
      font-size: 14px;
      font-weight: 600;
      color: var(--text);
      line-height: 1.4;
    }
    .directive-alert {
      background: var(--danger-light);
      border-color: var(--danger);
      color: var(--danger);
    }
    .directive-alert .directive-title {
      color: var(--danger);
    }
    .directive-alert .directive-text {
      color: var(--danger);
    }

    /* Diagnostics & Events */
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
      max-height: 240px;
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
      background: var(--danger-light);
    }
    .event-item.STEP_COMPLETED {
      border-left-color: var(--success);
      background: var(--success-light);
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
      max-width: 520px;
      width: 90%;
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="brand-title">SUTRA AI - Autonomous BAS Experiment Guardian</div>
      <div class="brand-subtitle">Smart India Hackathon 2026 (SIH26174) &bull; Bharatiya Antariksh Station (Space Technology)</div>
    </div>
    <div class="header-actions">
      <button onclick="toggleVoice()" id="voice-btn" class="voice-btn active">&#128266; Hands-Free Voice: ON</button>
      <button onclick="toggleTheme()" id="theme-btn">&#9790; Dark Mode</button>
      <button onclick="openHelpModal()">&#10067; Phone Guide</button>
      <div class="badge" id="status-badge">
        <span class="pulse-dot"></span>
        <span id="badge-text">AUTONOMOUS ACTIVE</span>
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
    <!-- Left: Video Stream & Autonomous Copilot Guidance -->
    <div>
      <div class="video-card">
        <div class="video-container">
          <canvas id="video-canvas" width="640" height="480"></canvas>
        </div>
        <div class="hud-controls">
          <button onclick="ackDeviation()" class="alert-btn">&#10004; Acknowledge Deviation</button>
          <button onclick="downloadSnapshot()">&#128247; Snapshot</button>
          <button onclick="fetchTelemetry()">&#8635; Refresh Status</button>
          <span id="fps-label" style="margin-left: auto; font-size: 13px; font-weight: 600; color: var(--text-muted)">Live Feed: 30 FPS</span>
        </div>
      </div>

      <!-- Autonomous Cognitive Copilot Section (No Chatbot - Pure Autonomous Oversight) -->
      <div class="copilot-card" id="copilot-card">
        <div class="copilot-header">
          <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 16px;">&#129302;</span>
            <strong style="font-size: 14px;">Autonomous Flight Copilot (Groq / Cognitive Engine)</strong>
          </div>
          <div class="copilot-status-pill" id="copilot-badge">
            <span class="pulse-dot" style="background: var(--primary);"></span>
            AUTONOMOUS GUARDIAN
          </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">
          <div class="directive-box">
            <div class="directive-title">Current Perception Observation</div>
            <div class="directive-text" id="copilot-status">Observing workspace setup and hand posture...</div>
          </div>
          <div class="directive-box" id="deviation-box">
            <div class="directive-title" id="deviation-title">Procedural & Safety Audit</div>
            <div class="directive-text" id="copilot-missing">Nominal &bull; Sequence matches flight protocol</div>
          </div>
        </div>

        <div class="directive-box" style="background: var(--primary-light); border-color: var(--primary);">
          <div class="directive-title" style="color: var(--primary);">Real-Time Mission Directive</div>
          <div class="directive-text" id="copilot-directive" style="color: var(--primary); font-size: 15px;">
            Proceed with active assay steps. Video stream continuously analyzed.
          </div>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px; font-size: 12px; color: var(--text-muted);">
          <span>&#128266; Spoken Voice: <strong id="copilot-spoken-text" style="color: var(--text);">"Standby for automated directive."</strong></span>
          <button onclick="repeatLastDirective()" style="padding: 4px 10px; font-size: 11px;">&#128260; Repeat Voice</button>
        </div>
      </div>
    </div>

    <!-- Right: Telemetry & State -->
    <div class="sidebar-card">
      <div class="card-heading">
        <span>Real-Time Edge Telemetry</span>
        <small style="color: var(--text-muted); font-size: 11px;">CPU-First &bull; &le;120MB</small>
      </div>
      
      <div class="metric-row">
        <span>Inference Latency (P95):</span>
        <span class="metric-val" id="p95-val">-- ms</span>
      </div>
      <div class="metric-row">
        <span>Process RAM:</span>
        <span class="metric-val" id="ram-val">-- MB</span>
      </div>
      <div class="metric-row">
        <span>Active Protocol Step:</span>
        <span class="metric-val" id="step-val" style="color: var(--primary);">STEP_01</span>
      </div>
      <div class="metric-row">
        <span>Step Description:</span>
        <span id="step-name-val" style="font-size: 12px; font-weight: 600;">Workspace Inspection</span>
      </div>
      <div class="metric-row">
        <span>Observed Action:</span>
        <span id="active-action-val" style="font-size: 12px; font-weight: 600; color: #10b981;">INSPECTING</span>
      </div>

      <div class="card-heading" style="margin-top: 14px;">
        <span>Telemetry State Events</span>
        <small id="event-count-badge" style="color: var(--primary); font-weight: bold;">0 logged</small>
      </div>

      <div class="event-log" id="event-log">
        <div class="event-item">Initializing SUTRA autonomous pipeline...</div>
      </div>
    </div>
  </div>

  <!-- Mobile IP Webcam Help Modal -->
  <div class="modal-overlay" id="help-modal" onclick="if(event.target===this) closeHelpModal()">
    <div class="modal-content">
      <h3 style="margin-bottom: 12px; color: var(--primary);">&#128241; Mobile Phone ko Live Camera Banayein</h3>
      <ol style="padding-left: 20px; line-height: 1.6; font-size: 13px;">
        <li>Apne Android Phone mein Play Store se <strong>"IP Webcam"</strong> install karein.</li>
        <li>Phone aur Laptop dono ko <strong>same Wi-Fi ya Mobile Hotspot</strong> se connect karein.</li>
        <li>Phone mein IP Webcam app open karein, scroll karein aur <strong>"Start server"</strong> tap karein.</li>
        <li>Phone ki screen par IP address dikhega (jaise: <code>http://192.168.1.15:8080</code>).</li>
        <li>Upar dashboard ke input box mein likhein: <br><strong style="color: var(--primary);">http://192.168.1.15:8080/video</strong></li>
        <li><strong>"Connect Mobile"</strong> button dabayein! Live phone video stream shuru ho jayegi!</li>
      </ol>
      <button onclick="closeHelpModal()" style="margin-top: 16px; width: 100%; justify-content: center; background: var(--primary); color: white; border: none;">Close Guide</button>
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
    let lastSpokenText = '';
    let lastDirectiveAnnouncement = '';

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

    // Smooth frame fetch loop (Canvas rendering)
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
      btn.innerHTML = voiceEnabled ? '&#128266; Hands-Free Voice: ON' : '&#128263; Voice: OFF';
      if (!voiceEnabled && window.speechSynthesis) window.speechSynthesis.cancel();
    }

    function repeatLastDirective() {
      if (lastDirectiveAnnouncement) speakText(lastDirectiveAnnouncement);
    }

    async function ackDeviation() {
      await fetch('/api/acknowledge', { method: 'POST' });
      speakText("Deviation acknowledged by operator. Resuming nominal protocol sequence.");
    }

    function downloadSnapshot() {
      const link = document.createElement('a');
      link.download = 'sutra_experiment_snapshot_' + Date.now() + '.png';
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
        document.getElementById('badge-text').innerText = 'AUTONOMOUS ACTIVE';
        speakText("Video source switched.");
      } catch (e) {
        alert("Failed to switch video source: " + e);
      }
    }

    function connectIpWebcam() {
      let url = document.getElementById('ipwebcam-url').value.trim();
      if (!url) {
        alert("Please enter your phone IP URL (e.g. http://192.168.1.15:8080/video)");
        return;
      }
      if (!url.startsWith('http://') && !url.startsWith('https://') && !url.startsWith('rtsp://')) {
        url = 'http://' + url;
      }
      url = url.replace('.8080', ':8080');
      if (!url.endsWith('/video') && !url.includes('?')) {
        if (url.endsWith('/')) url = url.slice(0, -1);
        url = url + '/video';
      }
      document.getElementById('ipwebcam-url').value = url;
      switchSource(url, null);
    }

    // Telemetry & Autonomous Copilot Polling Loop
    async function fetchTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();
        
        // 1. Diagnostics & Active Step
        if (data.diagnostics) {
          document.getElementById('p95-val').innerText = (data.diagnostics.latency_p95_ms || 0) + ' ms';
          document.getElementById('ram-val').innerText = (data.diagnostics.memory_rss_mb || 0) + ' MB';
        }
        if (data.pipeline) {
          document.getElementById('step-val').innerText = data.pipeline.current_step_id || 'STEP_01';
          document.getElementById('step-name-val').innerText = data.pipeline.current_step_name || 'Nominal';
          document.getElementById('active-action-val').innerText = data.pipeline.active_action || 'IDLE';
        }

        // 2. Autonomous Copilot Directives
        if (data.copilot) {
          const dir = data.copilot;
          document.getElementById('copilot-status').innerText = dir.current_status || 'Observing...';
          document.getElementById('copilot-directive').innerText = dir.next_action_directive || 'Continue protocol.';
          document.getElementById('copilot-spoken-text').innerText = '"' + (dir.spoken_announcement || '') + '"';

          const devBox = document.getElementById('deviation-box');
          const devTitle = document.getElementById('deviation-title');
          const devText = document.getElementById('copilot-missing');
          const copilotCard = document.getElementById('copilot-card');

          if (dir.missing_or_deviation) {
            devBox.className = 'directive-box directive-alert';
            devTitle.innerText = '⚠️ Deviation / Missing Step Warning';
            devText.innerText = dir.missing_or_deviation;
            copilotCard.style.borderLeftColor = 'var(--danger)';
          } else {
            devBox.className = 'directive-box';
            devTitle.innerText = 'Procedural & Safety Audit';
            devText.innerText = 'Nominal • Sequence matches flight protocol';
            copilotCard.style.borderLeftColor = 'var(--primary)';
          }

          // Voice announcement trigger if changed
          if (dir.spoken_announcement && dir.spoken_announcement !== lastDirectiveAnnouncement) {
            lastDirectiveAnnouncement = dir.spoken_announcement;
            speakText(dir.spoken_announcement);
          }
        }

        // 3. Event Logs
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
      } catch (e) {}
    }
    setInterval(fetchTelemetry, 1000);
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
    global engine_instance, latest_directive
    diag = engine_instance.get_diagnostics() if engine_instance else {}
    events = engine_instance.storage.read_events(limit=25) if engine_instance else []
    
    pipeline_state = {}
    if engine_instance:
        pe = engine_instance.protocol_engine
        cur_step = pe.protocol.steps[pe.current_step_idx] if pe.current_step_idx < len(pe.protocol.steps) else None
        pipeline_state = {
            "current_step_id": cur_step.id if cur_step else "COMPLETED",
            "current_step_name": cur_step.name if cur_step else "Protocol Finished",
            "step_status": pe.step_statuses.get(cur_step.id, "COMPLETED").value if cur_step else "COMPLETED",
            "active_deviation": pe.active_deviation
        }

    copilot_data = None
    if latest_directive:
        copilot_data = {
            "timestamp": latest_directive.timestamp,
            "current_status": latest_directive.current_status,
            "missing_or_deviation": latest_directive.missing_or_deviation,
            "next_action_directive": latest_directive.next_action_directive,
            "spoken_announcement": latest_directive.spoken_announcement,
            "alert_level": latest_directive.alert_level
        }

    return {
        "diagnostics": diag,
        "pipeline": pipeline_state,
        "copilot": copilot_data,
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
        src_val = str(src).strip()
        import re
        src_val = re.sub(r'\.8080', ':8080', src_val)
        if (src_val.startswith("http://") or src_val.startswith("https://")) and not src_val.endswith("/video") and "?" not in src_val:
            src_val = src_val.rstrip("/") + "/video"

    if engine_instance and engine_instance.ingestor:
        engine_instance.ingestor.switch_source(src_val)
        logger.info(f"Video source switched to: {src_val}")

    return {"status": "SWITCHED", "source": str(src)}


def start_web_server(engine: SutraVisionEngine, host: str = "127.0.0.1", port: int = 8000):
    """
    Launch background thread running the engine loop, and start uvicorn web server.
    """
    global engine_instance, latest_annotated_jpeg, latest_directive, copilot_engine
    engine_instance = engine
    engine.start()

    import threading

    def pipeline_worker():
        global latest_annotated_jpeg, latest_directive

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

                # Autonomous Copilot Evaluation (Periodic or on step change/deviation)
                pe = engine.protocol_engine
                cur_step = pe.protocol.steps[pe.current_step_idx] if pe.current_step_idx < len(pe.protocol.steps) else None
                
                ctx = {
                    "protocol_name": pe.protocol.protocol_name,
                    "current_step_id": cur_step.id if cur_step else "COMPLETED",
                    "current_step_name": cur_step.name if cur_step else "Experiment Finished",
                    "step_status": pe.step_statuses.get(cur_step.id, "COMPLETED").value if cur_step else "COMPLETED",
                    "active_deviation": pe.active_deviation,
                    "detected_objects": [o.class_name for o in (res.objects or [])],
                    "active_action": res.temporal_state.confirmed_activity if res.temporal_state else None,
                    "action_duration": res.temporal_state.duration_active if res.temporal_state else 0.0,
                }

                directive = copilot_engine.evaluate_state_autonomously(ctx)
                if directive:
                    latest_directive = directive

            except Exception as e:
                logger.error(f"Error in pipeline worker: {e}")
                time.sleep(0.05)

    t = threading.Thread(target=pipeline_worker, name="WebHUDWorker", daemon=True)
    t.start()

    print("\n" + "=" * 65)
    print(" [OK] SUTRA AUTONOMOUS WEB HUD IS RUNNING!")
    print(f" [OK] Open your browser at: http://{host}:{port}")
    print("=" * 65 + "\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    conf = get_config()
    engine = SutraVisionEngine(conf)
    start_web_server(engine, host="127.0.0.1", port=8000)
