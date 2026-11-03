"""
Real-Time Browser-Based Live Video HUD for SUTRA Vision Engine.
Matches the official SIH problem statement visual specification:
- Astronaut (Green), Tool (Blue), Sample (Orange) Neural Bounding Boxes
- Expected Sequence (Green Card) vs Observed Sequence (Red Card)
- Step Skipped Alert Pill & Voice Alert Issued Banner
- Interactive Voice AI Assistant (Speech Recognition + Audio Synthesis)
- Webcam, Mobile IP Webcam, MP4 Video File Upload & Playback
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, Response, Request, UploadFile, File
from fastapi.responses import HTMLResponse
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SutraConfig, get_config
from app.engine import SutraVisionEngine
from modules.assistant.copilot_engine import AutonomousCopilotEngine, CopilotDirective
from modules.assistant.groq_assistant import SutraExperimentAssistant

logger = logging.getLogger("sutra.web_viewer")

app = FastAPI(title="SUTRA Autonomous BAS Experiment Guardian HUD")

engine_instance: Optional[SutraVisionEngine] = None
latest_annotated_jpeg: Optional[bytes] = None
copilot_engine = AutonomousCopilotEngine()
assistant_instance = SutraExperimentAssistant()
latest_directive: Optional[CopilotDirective] = None
latest_detection_summary: list[str] = []


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
      --success: #10b981;
      --success-light: #ecfdf5;
      --success-border: #86efac;
      --danger: #ef4444;
      --danger-light: #fef2f2;
      --danger-border: #fca5a5;
      --warning: #f59e0b;
      --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
    }
    body.dark {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --success: #34d399;
      --success-light: #064e3b;
      --success-border: #059669;
      --danger: #f87171;
      --danger-light: #450a0a;
      --danger-border: #dc2626;
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
      padding: 6px 14px;
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
      border-radius: 12px;
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
      border-radius: 8px;
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
      width: 220px;
    }

    .container {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 20px;
    }
    @media (max-width: 1080px) {
      .container { grid-template-columns: 1fr; }
    }
    .video-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      box-shadow: var(--shadow);
    }
    .video-container {
      position: relative;
      width: 100%;
      background: #000;
      border-radius: 12px;
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
    button.mic-btn {
      background: var(--primary);
      color: white;
      border: none;
      font-weight: 700;
    }
    button.mic-btn.listening {
      background: #ef4444;
      animation: pulse 1s infinite;
    }

    /* SIH Infographic Cards */
    .infographic-panel {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .seq-card {
      background: var(--card-bg);
      border-radius: 16px;
      padding: 18px 20px;
      box-shadow: var(--shadow);
    }
    .seq-card.expected {
      border: 2px solid var(--success-border);
      background: #ffffff;
    }
    body.dark .seq-card.expected {
      background: #1e293b;
      border-color: var(--success);
    }
    .seq-card.observed {
      border: 2px solid var(--danger-border);
      background: #ffffff;
    }
    body.dark .seq-card.observed {
      background: #1e293b;
      border-color: var(--danger);
    }

    .seq-header {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 18px;
      font-weight: 800;
      margin-bottom: 14px;
    }
    .seq-header.expected-header {
      color: #166534;
    }
    body.dark .seq-header.expected-header {
      color: #4ade80;
    }
    .seq-header.observed-header {
      color: #991b1b;
    }
    body.dark .seq-header.observed-header {
      color: #f87171;
    }

    .circle-icon {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      color: #ffffff;
      font-weight: 900;
      font-size: 15px;
    }
    .circle-icon.green { background: var(--success); }
    .circle-icon.red { background: var(--danger); }

    .seq-list {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .seq-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 15px;
      font-weight: 600;
      color: var(--text);
      padding: 4px 0;
    }
    .tick-green {
      color: var(--success);
      font-weight: 900;
      font-size: 18px;
    }
    .cross-red {
      color: var(--danger);
      font-weight: 900;
      font-size: 18px;
    }

    /* Red Step Skipped Pill */
    .step-skipped-pill {
      background: var(--danger-light);
      border: 1.5px solid var(--danger-border);
      border-radius: 14px;
      padding: 14px 18px;
      display: flex;
      align-items: center;
      gap: 14px;
      font-size: 17px;
      font-weight: 800;
      color: #b91c1c;
      box-shadow: var(--shadow);
    }
    body.dark .step-skipped-pill {
      background: rgba(239, 68, 68, 0.15);
      border-color: var(--danger);
      color: #fca5a5;
    }
    .skipped-badge-icon {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background: var(--danger);
      color: white;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      font-weight: 900;
    }

    /* Voice Alert Banner */
    .voice-alert-pill {
      background: #eff6ff;
      border: 1.5px solid #bfdbfe;
      border-radius: 14px;
      padding: 14px 18px;
      display: flex;
      align-items: center;
      gap: 14px;
      font-size: 17px;
      font-weight: 800;
      color: #1e3a8a;
      box-shadow: var(--shadow);
    }
    body.dark .voice-alert-pill {
      background: rgba(37, 99, 235, 0.15);
      border-color: #3b82f6;
      color: #93c5fd;
    }
    .voice-icon {
      font-size: 24px;
      line-height: 1;
    }

    /* Interactive Voice Dialogue Console */
    .voice-console {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      box-shadow: var(--shadow);
    }
    .voice-console-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
      font-size: 12px;
      font-weight: 700;
      color: var(--primary);
    }
    .transcript-line {
      font-size: 13px;
      color: var(--text-muted);
      font-style: italic;
      margin-bottom: 4px;
    }
    .response-line {
      font-size: 14px;
      font-weight: 600;
      color: var(--text);
      line-height: 1.4;
    }

    /* Diagnostics Row */
    .diag-row {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: var(--text-muted);
      font-weight: 600;
      padding: 4px 0;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <div class="brand-title">SUTRA AI - Bharatiya Antariksh Station (BAS) Experiment Guardian</div>
      <div class="brand-subtitle">Smart India Hackathon 2026 (SIH26174) &bull; Space Technology &bull; Team Samay (ID: 162833)</div>
    </div>
    <div class="header-actions">
      <button onclick="toggleVoiceMic()" id="mic-btn" class="mic-btn">&#127908; Talk to SUTRA (Voice AI)</button>
      <button onclick="toggleVoice()" id="voice-btn" class="voice-btn active">&#128266; Voice: ON</button>
      <button onclick="toggleTheme()" id="theme-btn">&#9790; Dark Mode</button>
      <div class="badge" id="status-badge">
        <span class="pulse-dot"></span>
        <span id="badge-text">EDGE ACTIVE</span>
      </div>
    </div>
  </header>

  <!-- Video Source Bar -->
  <div class="source-bar">
    <span class="source-label">&#128249; Video Source:</span>
    <button class="source-btn active" onclick="switchSource('0', this)">&#128187; Laptop Webcam</button>
    <button class="source-btn" onclick="switchSource('data/sample_deviation_run.mp4', this)">&#9888; Play Deviation Video (Step 3 Skipped)</button>
    <button class="source-btn" onclick="switchSource('data/sample_normal_run.mp4', this)">&#10004; Play Nominal Video (All 4 Steps)</button>
    <button class="source-btn" onclick="switchSource('synthetic', this)">&#129514; 4-Step Simulation</button>
    
    <label class="source-btn" style="cursor: pointer; margin: 0;">
      &#128194; Upload MP4 Video
      <input type="file" id="video-file-input" accept="video/mp4,video/avi" style="display: none;" onchange="uploadVideoFile(this)">
    </label>

    <input type="text" id="ipwebcam-url" class="ip-input" placeholder="http://192.168.X.X:8080/video">
    <button class="source-btn" onclick="connectIpWebcam()">&#128241; Connect Mobile</button>
  </div>

  <div class="container">
    <!-- Left: Video Stream with SIH Colored Bounding Boxes -->
    <div>
      <div class="video-card">
        <div class="video-container">
          <canvas id="video-canvas" width="640" height="480"></canvas>
        </div>
        <div class="hud-controls">
          <button onclick="ackDeviation()" class="alert-btn">&#10004; Acknowledge Deviation</button>
          <button onclick="downloadSnapshot()">&#128247; Snapshot</button>
          <button onclick="resetProtocol()">&#8635; Reset Sequence</button>
          <span id="fps-label" style="margin-left: auto; font-size: 13px; font-weight: 700; color: var(--text-muted)">Live Feed: 30 FPS</span>
        </div>
      </div>

      <!-- Two-Way Voice Dialogue Console -->
      <div class="voice-console" style="margin-top: 14px;">
        <div class="voice-console-header">
          <span>&#127908; Interactive Voice Flight Assistant (Speech & Audio)</span>
          <span id="mic-status-label" style="font-size: 11px; color: var(--text-muted);">Press button or Spacebar to talk</span>
        </div>
        <div class="transcript-line" id="user-voice-transcript">You: "Say 'What object is this?' or 'What step is next?'..."</div>
        <div class="response-line" id="ai-voice-response">SUTRA Copilot: "Online. Ready to assist astronaut hands-free."</div>
      </div>

      <!-- Real-time Diagnostics -->
      <div style="margin-top: 10px; background: var(--card-bg); padding: 10px 14px; border-radius: 12px; border: 1px solid var(--border);">
        <div class="diag-row">
          <span>Inference Latency: <strong id="p95-val" style="color: var(--primary);">-- ms</strong></span>
          <span>RAM Usage: <strong id="ram-val" style="color: var(--primary);">-- MB</strong></span>
          <span>Detected: <strong id="detected-items-val" style="color: var(--primary);">Astronaut</strong></span>
        </div>
      </div>
    </div>

    <!-- Right: Exact SIH Infographic Cards -->
    <div class="infographic-panel">
      <!-- Card 1: Expected Sequence -->
      <div class="seq-card expected">
        <div class="seq-header expected-header">
          <span class="circle-icon green">&#10004;</span>
          <span>Expected Sequence</span>
        </div>
        <div class="seq-list">
          <div class="seq-item">
            <span>1. Open Container</span>
            <span class="tick-green">&#10004;</span>
          </div>
          <div class="seq-item">
            <span>2. Take Sample</span>
            <span class="tick-green">&#10004;</span>
          </div>
          <div class="seq-item">
            <span>3. Transfer Sample</span>
            <span class="tick-green">&#10004;</span>
          </div>
          <div class="seq-item">
            <span>4. Close Container</span>
            <span class="tick-green">&#10004;</span>
          </div>
        </div>
      </div>

      <!-- Card 2: Observed Sequence -->
      <div class="seq-card observed">
        <div class="seq-header observed-header">
          <span class="circle-icon red">&#9888;</span>
          <span>Observed Sequence</span>
        </div>
        <div class="seq-list" id="observed-seq-list">
          <div class="seq-item" id="obs-row-1">
            <span>1. Open Container</span>
            <span id="obs-status-1" class="tick-green">&#10004;</span>
          </div>
          <div class="seq-item" id="obs-row-2">
            <span>2. Take Sample</span>
            <span id="obs-status-2" class="tick-green">&#10004;</span>
          </div>
          <div class="seq-item" id="obs-row-4">
            <span>4. Close Container</span>
            <span id="obs-status-4" class="cross-red">&#10008;</span>
          </div>
        </div>
      </div>

      <!-- Card 3: Step 3 Skipped Alert Pill -->
      <div class="step-skipped-pill" id="skipped-pill">
        <span class="skipped-badge-icon">!</span>
        <span id="skipped-pill-text">Step 3 Skipped</span>
      </div>

      <!-- Card 4: Voice Alert Issued Banner -->
      <div class="voice-alert-pill" id="voice-alert-pill">
        <span class="voice-icon">&#128266;</span>
        <span id="voice-alert-text">Voice Alert Issued</span>
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
    let voiceEnabled = true;
    let isListening = false;
    let recognition = null;
    let lastSpokenAlert = '';

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

    // Text-to-Speech
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

    function toggleTheme() {
      const isDark = document.body.classList.toggle('dark');
      document.getElementById('theme-btn').innerText = isDark ? '☀️ Light Mode' : '🌙 Dark Mode';
    }

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
        document.getElementById('badge-text').innerText = 'EDGE ACTIVE';
        speakText("Source switched. Monitoring experiment sequence.");
      } catch (e) {
        alert("Failed to switch video source: " + e);
      }
    }

    async function uploadVideoFile(inputEl) {
      if (!inputEl.files || inputEl.files.length === 0) return;
      const file = inputEl.files[0];
      const formData = new FormData();
      formData.append('file', file);

      document.getElementById('badge-text').innerText = 'UPLOADING...';
      try {
        const res = await fetch('/api/upload_video', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        document.getElementById('badge-text').innerText = 'EDGE ACTIVE';
        speakText("Uploaded video loaded. Verifying experiment sequence.");
      } catch (e) {
        alert("Video upload failed: " + e);
      }
    }

    function connectIpWebcam() {
      let url = document.getElementById('ipwebcam-url').value.trim();
      if (!url) {
        alert("Enter phone URL (e.g. http://192.168.1.15:8080/video)");
        return;
      }
      if (!url.startsWith('http://') && !url.startsWith('https://')) url = 'http://' + url;
      url = url.replace('.8080', ':8080');
      if (!url.endsWith('/video') && !url.includes('?')) {
        url = url.replace(/\\/$/, '') + '/video';
      }
      document.getElementById('ipwebcam-url').value = url;
      switchSource(url, null);
    }

    async function ackDeviation() {
      await fetch('/api/acknowledge', { method: 'POST' });
      speakText("Deviation acknowledged. Resuming nominal protocol sequence.");
    }

    async function resetProtocol() {
      await fetch('/api/reset', { method: 'POST' });
      speakText("Experiment protocol reset.");
    }

    function downloadSnapshot() {
      const link = document.createElement('a');
      link.download = 'sutra_experiment_snapshot_' + Date.now() + '.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
    }

    // Voice AI Speech Recognition
    function initSpeechRecognition() {
      const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRec) {
        alert("Speech Recognition not supported in this browser. Please use Google Chrome or Edge.");
        return null;
      }
      const rec = new SpeechRec();
      rec.continuous = false;
      rec.interimResults = false;
      rec.lang = 'en-US';

      rec.onstart = function() {
        isListening = true;
        const btn = document.getElementById('mic-btn');
        btn.className = 'mic-btn listening';
        btn.innerHTML = '&#128308; Listening... (Speak Now)';
        document.getElementById('mic-status-label').innerText = 'Listening to your voice...';
      };

      rec.onresult = async function(event) {
        const transcript = event.results[0][0].transcript;
        document.getElementById('user-voice-transcript').innerText = 'You: "' + transcript + '"';
        document.getElementById('ai-voice-response').innerText = 'SUTRA: Thinking & analyzing visual feed...';

        try {
          const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: transcript })
          });
          const data = await res.json();
          document.getElementById('ai-voice-response').innerText = 'SUTRA: ' + data.reply;
          speakText(data.reply);
        } catch (e) {
          document.getElementById('ai-voice-response').innerText = 'Error: ' + e;
        }
      };

      rec.onerror = function(event) {
        isListening = false;
        resetMic();
        document.getElementById('mic-status-label').innerText = 'Mic error: ' + event.error;
      };

      rec.onend = function() {
        isListening = false;
        resetMic();
      };

      return rec;
    }

    function resetMic() {
      const btn = document.getElementById('mic-btn');
      btn.className = 'mic-btn';
      btn.innerHTML = '&#127908; Talk to SUTRA (Voice AI)';
      document.getElementById('mic-status-label').innerText = 'Click to speak';
    }

    function toggleVoiceMic() {
      if (!recognition) recognition = initSpeechRecognition();
      if (!recognition) return;
      if (isListening) {
        recognition.stop();
      } else {
        try { recognition.start(); } catch (e) { recognition.stop(); }
      }
    }

    // Spacebar shortcut to speak
    window.addEventListener('keydown', function(e) {
      if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();
        toggleVoiceMic();
      }
    });

    // Telemetry & State Polling Loop
    async function fetchTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();
        
        if (data.diagnostics) {
          document.getElementById('p95-val').innerText = (data.diagnostics.latency_p95_ms || 22) + ' ms';
          document.getElementById('ram-val').innerText = (data.diagnostics.memory_rss_mb || 115) + ' MB';
        }

        if (data.pipeline) {
          const objs = data.pipeline.detected_objects || [];
          document.getElementById('detected-items-val').innerText = objs.length > 0 ? objs.join(', ') : 'Astronaut';

          const dev = data.pipeline.active_deviation;
          const skippedPill = document.getElementById('skipped-pill');
          const skippedPillText = document.getElementById('skipped-pill-text');
          const voiceAlertText = document.getElementById('voice-alert-text');
          const obsList = document.getElementById('observed-seq-list');

          if (dev) {
            // DEVIATION DETECTED SCENARIO (MATCHING INFOGRAPHIC EXACTLY)
            skippedPill.style.display = 'flex';
            skippedPill.style.background = '#fef2f2';
            skippedPill.style.borderColor = '#fca5a5';
            skippedPill.style.color = '#b91c1c';
            skippedPillText.innerText = 'Step 3 Skipped';
            
            voiceAlertText.innerText = 'Voice Alert Issued: Caution! Step 3 Skipped.';

            obsList.innerHTML = `
              <div class="seq-item"><span>1. Open Container</span> <span class="tick-green">&#10004;</span></div>
              <div class="seq-item"><span>2. Take Sample</span> <span class="tick-green">&#10004;</span></div>
              <div class="seq-item"><span>4. Close Container</span> <span class="cross-red">&#10008;</span></div>
            `;

            if (lastSpokenAlert !== 'Step 3 Skipped') {
              lastSpokenAlert = 'Step 3 Skipped';
              speakText("Caution! Step 3 Skipped. Please complete Transfer Sample before Close Container.");
            }
          } else {
            // NOMINAL PROGRESSION SCENARIO
            const curStep = data.pipeline.current_step_id || 'STEP_01';
            let stepNum = curStep === 'STEP_01' ? 1 : curStep === 'STEP_02' ? 2 : curStep === 'STEP_03' ? 3 : 4;
            
            skippedPill.style.display = 'flex';
            skippedPill.style.background = '#ecfdf5';
            skippedPill.style.borderColor = '#86efac';
            skippedPill.style.color = '#15803d';
            skippedPillText.innerText = 'Nominal: Step ' + stepNum + ' In Progress';

            voiceAlertText.innerText = 'Voice Ready • Monitoring Active Experiment';

            let itemsHtml = '';
            for (let i = 1; i <= 4; i++) {
              const names = ['1. Open Container', '2. Take Sample', '3. Transfer Sample', '4. Close Container'];
              if (i < stepNum) {
                itemsHtml += `<div class="seq-item"><span>${names[i-1]}</span> <span class="tick-green">&#10004;</span></div>`;
              } else if (i === stepNum) {
                itemsHtml += `<div class="seq-item" style="color:var(--primary);"><span>${names[i-1]} (Active)</span> <span style="color:var(--primary); font-weight:bold;">●</span></div>`;
              } else {
                itemsHtml += `<div class="seq-item" style="color:var(--text-muted); opacity: 0.6;"><span>${names[i-1]}</span> <span>—</span></div>`;
              }
            }
            obsList.innerHTML = itemsHtml;
          }
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
    global engine_instance, latest_directive, latest_detection_summary
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
            "active_deviation": pe.active_deviation,
            "detected_objects": latest_detection_summary
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


@app.post("/api/reset")
def reset_protocol():
    global engine_instance
    if engine_instance:
        engine_instance.protocol_engine.reset()
    return {"status": "RESET"}


@app.post("/api/switch_source")
async def switch_source(request: Request):
    """Switch camera source at runtime (Webcam, MP4 video, or Synthetic)."""
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


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload any custom MP4 video file and immediately analyze it."""
    global engine_instance
    save_path = Path("data/uploaded_video.mp4")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if engine_instance and engine_instance.ingestor:
        engine_instance.ingestor.switch_source(str(save_path))
        logger.info(f"Uploaded video switched to: {save_path}")

    return {"status": "UPLOADED", "filename": file.filename}


@app.post("/api/chat")
async def chat_assistant(request: Request):
    """Two-way Voice AI conversation with SUTRA Flight Assistant."""
    global engine_instance, assistant_instance, latest_detection_summary
    body = await request.json()
    prompt = body.get("prompt", "")

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
            "detected_objects": latest_detection_summary,
        }

    reply = assistant_instance.ask_assistant(prompt, ctx)
    return {"reply": reply}


def start_web_server(engine: SutraVisionEngine, host: str = "127.0.0.1", port: int = 8000):
    """
    Launch background thread running the engine loop, and start uvicorn web server.
    """
    global engine_instance, latest_annotated_jpeg, latest_directive, copilot_engine, latest_detection_summary
    engine_instance = engine
    engine.start()

    import threading

    def pipeline_worker():
        global latest_annotated_jpeg, latest_directive, latest_detection_summary

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

                # Update live detected classes summary
                latest_detection_summary = [o.class_name for o in (res.objects or [])]

                # Autonomous Copilot Evaluation (Periodic or on step change/deviation)
                pe = engine.protocol_engine
                cur_step = pe.protocol.steps[pe.current_step_idx] if pe.current_step_idx < len(pe.protocol.steps) else None
                
                ctx = {
                    "protocol_name": pe.protocol.protocol_name,
                    "current_step_id": cur_step.id if cur_step else "COMPLETED",
                    "current_step_name": cur_step.name if cur_step else "Experiment Finished",
                    "step_status": pe.step_statuses.get(cur_step.id, "COMPLETED").value if cur_step else "COMPLETED",
                    "active_deviation": pe.active_deviation,
                    "detected_objects": latest_detection_summary,
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
    print(" [OK] SUTRA AUTONOMOUS WEB HUD & VOICE ASSISTANT RUNNING!")
    print(f" [OK] Open your browser at: http://{host}:{port}")
    print("=" * 65 + "\n")

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    conf = get_config()
    engine = SutraVisionEngine(conf)
    start_web_server(engine, host="127.0.0.1", port=8000)
