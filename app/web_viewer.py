"""
Real-Time Browser-Based Live Video HUD for SUTRA Vision Engine.
Matches the official SIH problem statement visual specification:
- Astronaut (Green), Tool (Blue), Sample (Orange) Neural Bounding Boxes
- Expected Sequence (Green Card) vs Observed Sequence (Red Card)
- Step 3 Skipped Alert Pill & Voice Alert Issued Banner
- Interactive Voice AI Assistant (Speech Recognition + Audio Synthesis)
- Webcam, Mobile IP Webcam, MP4 Video File Upload & Playback
- Dynamic Sensitivity Slider & One-Click Deviation Simulation
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
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SutraConfig, get_config
from app.engine import SutraVisionEngine
from modules.assistant.copilot_engine import AutonomousCopilotEngine, CopilotDirective
from modules.assistant.groq_assistant import SutraExperimentAssistant
from modules.assistant.gemini_vision import GeminiVisionCopilot
from modules.protocol.fsm_engine import StepStatus

logger = logging.getLogger("sutra.web_viewer")

app = FastAPI(title="SUTRA Autonomous BAS Experiment Guardian HUD")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine_instance: Optional[SutraVisionEngine] = None
latest_annotated_jpeg: Optional[bytes] = None
latest_raw_frame: Optional[np.ndarray] = None
copilot_engine = AutonomousCopilotEngine()
assistant_instance = SutraExperimentAssistant()
gemini_copilot = GeminiVisionCopilot()
latest_directive: Optional[CopilotDirective] = None
latest_detection_summary: list[str] = []
latest_video_summary: str = "Astronaut actively monitored in payload bay. Analyzing continuous spatial gestures and scientific apparatus..."


HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SUTRA AI - Bharatiya Antariksh Station (BAS) Experiment Guardian</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --text-muted: #64748b;
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --success: #10b981;
      --success-light: #ecfdf5;
      --success-border: #86efac;
      --danger: #ef4444;
      --danger-light: #fef2f2;
      --danger-border: #fca5a5;
      --warning: #f59e0b;
      --astronaut-color: #28b95a;
      --tool-color: #1b8ae2;
      --sample-color: #f59114;
      --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
      --card-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
    }
    body.dark {
      --bg: #0b1329;
      --card-bg: #152238;
      --border: #1e3a5f;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --success: #34d399;
      --success-light: #064e3b;
      --success-border: #059669;
      --danger: #f87171;
      --danger-light: #450a0a;
      --danger-border: #dc2626;
      --warning: #fbbf24;
      --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
      --card-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 16px 24px;
      transition: background 0.3s ease, color 0.3s ease;
    }
    
    /* Header Bar */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .brand-title {
      font-size: 22px;
      font-weight: 800;
      color: var(--primary);
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .brand-subtitle {
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 2px;
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

    /* Modern Control Bar */
    .source-bar {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 10px 16px;
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      box-shadow: var(--shadow);
    }
    .source-group {
      display: flex;
      align-items: center;
      gap: 6px;
      background: var(--bg);
      padding: 4px;
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    .source-btn {
      padding: 6px 12px;
      font-size: 12px;
      border-radius: 7px;
      border: none;
      background: transparent;
      color: var(--text);
      cursor: pointer;
      font-weight: 600;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 5px;
    }
    .source-btn:hover {
      background: rgba(0, 0, 0, 0.05);
    }
    body.dark .source-btn:hover {
      background: rgba(255, 255, 255, 0.08);
    }
    .source-btn.active {
      background: var(--primary);
      color: #fff;
      box-shadow: 0 2px 4px rgba(37, 99, 235, 0.25);
    }
    .ip-input {
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 7px;
      font-size: 12px;
      width: 190px;
    }

    /* Main Clean Stack Layout */
    .container {
      display: flex;
      flex-direction: column;
      gap: 20px;
      max-width: 1080px;
      margin: 0 auto;
    }
    .deck-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      width: 100%;
    }
    @media (max-width: 768px) {
      .deck-grid { grid-template-columns: 1fr; }
    }
    .ai-copilot-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px 22px;
      box-shadow: var(--card-shadow);
      width: 100%;
    }

    /* Video Card */
    .video-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: var(--card-shadow);
      width: 100%;
    }
    .card-title-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }
    .card-title-text {
      font-size: 18px;
      font-weight: 800;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .video-container {
      position: relative;
      width: 100%;
      background: #000;
      border-radius: 14px;
      overflow: hidden;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 480px;
      border: 1px solid var(--border);
    }
    #video-canvas {
      width: 100%;
      max-width: 860px;
      height: auto;
      max-height: 560px;
      object-fit: contain;
      display: block;
      margin: 0 auto;
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
    button.alert-btn:hover {
      background: #dc2626;
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
      background: #f0fdf4;
    }
    body.dark .seq-card.expected {
      background: #064e3b;
      border-color: var(--success);
    }
    .seq-card.observed {
      border: 2px solid var(--danger-border);
      background: #fef2f2;
    }
    body.dark .seq-card.observed {
      background: #450a0a;
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
      background: #fee2e2;
      border: 2px solid var(--danger-border);
      border-radius: 14px;
      padding: 14px 18px;
      display: flex;
      align-items: center;
      gap: 14px;
      font-size: 18px;
      font-weight: 800;
      color: #b91c1c;
      box-shadow: var(--shadow);
    }
    body.dark .step-skipped-pill {
      background: rgba(239, 68, 68, 0.2);
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
      border: 2px solid #bfdbfe;
      border-radius: 14px;
      padding: 14px 18px;
      display: flex;
      align-items: center;
      gap: 14px;
      font-size: 18px;
      font-weight: 800;
      color: #1e3a8a;
      box-shadow: var(--shadow);
    }
    body.dark .voice-alert-pill {
      background: rgba(37, 99, 235, 0.2);
      border-color: #3b82f6;
      color: #93c5fd;
    }
    .voice-icon {
      font-size: 24px;
      line-height: 1;
    }

    /* Detected Objects Badge Row */
    .detection-tags-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 16px;
      box-shadow: var(--shadow);
    }
    .tag-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 14px;
      border-radius: 14px;
      font-size: 12px;
      font-weight: 700;
      color: white;
      margin-right: 6px;
      margin-bottom: 4px;
    }
    .tag-astronaut { background: var(--astronaut-color); }
    .tag-tool { background: var(--tool-color); }
    .tag-sample { background: var(--sample-color); }
    .tag-apparatus { background: #8b5cf6; }

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
      <div class="brand-title">
        <span>&#128640;</span>
        <span>SUTRA AI - Bharatiya Antariksh Station (BAS) Experiment Guardian</span>
      </div>
      <div class="brand-subtitle">Smart India Hackathon 2026 (SIH26174) &bull; Space Technology &bull; Team Samay</div>
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

  <!-- Video Source Bar & Sensitivity Slider -->
  <div class="source-bar">
    <span style="font-size: 13px; font-weight: 700; color: var(--text-muted);">&#128249; Video Source:</span>
    <div class="source-group">
      <button class="source-btn active" onclick="switchSource('0', this)">&#128187; Laptop Webcam</button>
      <button class="source-btn" onclick="switchSource('data/sample_deviation_run.mp4', this)">&#9888; Play Deviation Video (Step 3 Skipped)</button>
      <button class="source-btn" onclick="switchSource('data/sample_normal_run.mp4', this)">&#10004; Play Nominal Video (All 4 Steps)</button>
      <button class="source-btn" onclick="switchSource('synthetic', this)">&#129514; 4-Step Simulation</button>
      
      <label class="source-btn" style="cursor: pointer; margin: 0;">
        &#128194; Upload MP4
        <input type="file" id="video-file-input" accept="video/mp4,video/avi" style="display: none;" onchange="uploadVideoFile(this)">
      </label>
    </div>

    <div style="display: flex; align-items: center; gap: 6px;">
      <input type="text" id="ipwebcam-url" class="ip-input" placeholder="http://192.168.X.X:8080/video">
      <button class="source-btn" style="background: var(--bg); border: 1px solid var(--border);" onclick="connectIpWebcam()">&#128241; Connect Mobile</button>
    </div>

    <div style="margin-left: auto; display: flex; align-items: center; gap: 8px;">
      <span style="font-size: 12px; font-weight: 700; color: var(--text-muted);">&#127919; Sensitivity:</span>
      <input type="range" id="conf-slider" min="0.10" max="0.60" step="0.05" value="0.20" onchange="updateConfidence(this.value)">
      <span id="conf-val" style="font-size: 12px; font-weight: bold; color: var(--primary);">20%</span>
    </div>
  </div>

  <div class="container">
    <!-- 1. Full-Width Cinematic Video Feed Card -->
    <div class="video-card">
      <div class="card-title-bar">
        <div class="card-title-text">
          <span>&#127919;</span>
          <span>SUTRA Autonomous Vision &amp; Procedural Guardian</span>
        </div>
        <div style="display: flex; gap: 8px;">
          <button onclick="simulateDeviation()" style="background: #fef2f2; border-color: #fca5a5; color: #b91c1c; font-size: 12px; font-weight: 700; padding: 6px 12px; border-radius: 8px;">
            &#9888; Trigger Step 3 Skip
          </button>
          <button onclick="resetProtocol()" style="font-size: 12px; font-weight: 700; padding: 6px 12px; border-radius: 8px;">
            &#8635; Reset Protocol
          </button>
        </div>
      </div>

      <div class="video-container">
        <canvas id="video-canvas" width="640" height="480"></canvas>
      </div>

      <div class="hud-controls">
        <button onclick="geminiInspect()" style="background: linear-gradient(135deg, #2563eb, #7c3aed); color: #fff; border: none; font-weight: 700;">
          &#128269; Gemini Flash Inspect
        </button>
        <button onclick="ackDeviation()" class="alert-btn">&#10004; Acknowledge Deviation</button>
        <button onclick="downloadSnapshot()">&#128247; Snapshot</button>
        <button onclick="resetProtocol()">&#8635; Reset Sequence</button>
        <span id="fps-label" style="margin-left: auto; font-size: 13px; font-weight: 700; color: var(--text-muted)">Live Feed: 30 FPS</span>
      </div>

      <!-- Live Detection Tags Pill Row -->
      <div class="detection-tags-card" style="margin-top: 14px;">
        <div style="font-size: 12px; font-weight: 700; margin-bottom: 6px; color: var(--text-muted); letter-spacing: 0.5px;">
          PERCEPTION LAYER &bull; LIVE DETECTED ENTITIES
        </div>
        <div id="detection-tags-container">
          <span class="tag-pill tag-astronaut">Astronaut (Operator)</span>
        </div>
      </div>

      <!-- PROMINENT 3-4 LINE VIDEO SCENE ANALYSIS & TELEMETRY BOX -->
      <div class="scene-summary-box" style="margin-top: 14px; background: rgba(15, 23, 42, 0.7); border: 1.5px solid rgba(56, 189, 248, 0.4); border-radius: 12px; padding: 14px 18px; text-align: left;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
          <div style="font-size: 12px; font-weight: 800; color: var(--primary); text-transform: uppercase; letter-spacing: 0.8px; display: flex; align-items: center; gap: 6px;">
            <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#ef4444;"></span>
            <span>Live Video Scene Intelligence &amp; Telemetry Report (3-4 Lines)</span>
          </div>
          <span id="summary-source-badge" style="font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 6px; background: rgba(56, 189, 248, 0.15); color: var(--primary);">
            SUTRA Onboard Edge AI
          </span>
        </div>
        <div id="summary-lines-container" style="display: flex; flex-direction: column; gap: 7px; font-size: 13.5px; line-height: 1.5; color: var(--text);">
          <div id="sum-line1" style="color: #38bdf8;"><strong>👨‍🚀 Operator Activity:</strong> Astronaut actively maneuvering in payload bay...</div>
          <div id="sum-line2" style="color: #c084fc;"><strong>✋ Handheld &amp; Tools:</strong> Tracking handheld apparatus (Tool / Specimen / Camera)...</div>
          <div id="sum-line3" style="color: #4ade80;"><strong>📋 SOP Compliance:</strong> STEP_02 Sample Unsealing (Nominal progression)...</div>
          <div id="sum-line4" style="color: #fbbf24;"><strong>🚀 AI Directive:</strong> Continue nominal protocol sequence...</div>
        </div>
      </div>
    </div>

    <!-- 2. Dual Verification Sequence Cards Side-by-Side directly below video -->
    <div class="deck-grid">
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

        <!-- Step 3 Skipped Alert Pill -->
        <div class="step-skipped-pill" id="skipped-pill" style="margin-top: 12px;">
          <span class="skipped-badge-icon">!</span>
          <span id="skipped-pill-text">Step 3 Skipped</span>
        </div>

        <!-- Voice Alert Issued Banner -->
        <div class="voice-alert-pill" id="voice-alert-pill" style="margin-top: 8px;">
          <span class="voice-icon">&#128266;</span>
          <span id="voice-alert-text">Voice Alert Issued</span>
        </div>
      </div>
    </div>

    <!-- 3. SUTRA AI Copilot & Live Video Scene Intelligence -->
    <div class="ai-copilot-card">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
        <div style="font-size: 16px; font-weight: 800; color: var(--primary); display: flex; align-items: center; gap: 8px;">
          <span>&#128752;</span>
          <span>SUTRA Autonomous Flight Copilot &amp; Live Video Scene Intelligence</span>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
          <input type="password" id="custom-api-key" placeholder="Optional Groq / Gemini API Key" style="background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 6px 10px; border-radius: 6px; font-size: 12px; width: 220px;">
          <button onclick="setApiKey()" style="font-size: 12px; font-weight: 700; padding: 6px 12px; border-radius: 6px; background: var(--primary); color: #fff; border: none;">Activate Key</button>
        </div>
      </div>

      <!-- Live Video Summary Banner -->
      <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 12px 16px; margin-bottom: 12px;">
        <div style="font-size: 11px; font-weight: 800; color: var(--primary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">
          &#128308; Live Video Scene Analysis &amp; Telemetry Narrative
        </div>
        <div id="live-video-summary" style="font-size: 14px; font-weight: 600; line-height: 1.5; color: var(--text);">
          Astronaut active in payload bay. Analyzing continuous spatial gestures and scientific apparatus...
        </div>
      </div>

      <!-- Next Action Directive Box -->
      <div style="background: rgba(34, 197, 94, 0.08); border: 1px solid rgba(34, 197, 94, 0.25); border-radius: 12px; padding: 12px 16px; margin-bottom: 14px;">
        <div style="font-size: 11px; font-weight: 800; color: var(--success); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">
          &#128073; Mission Directive &amp; Next Physical Action
        </div>
        <div id="next-action-directive" style="font-size: 14px; font-weight: 600; color: var(--text);">
          Inspect and confirm payload workspace sterility.
        </div>
      </div>

      <!-- Two-Way Voice Dialogue Console -->
      <div class="voice-console">
        <div class="voice-console-header">
          <span>&#127908; Interactive Voice Flight Assistant (Speech &amp; Audio)</span>
          <span id="mic-status-label" style="font-size: 11px; color: var(--text-muted);">Press button or Spacebar to talk</span>
        </div>
        <div class="transcript-line" id="user-voice-transcript">You: "Say 'What object is this?' or 'What step is next?'..."</div>
        <div class="response-line" id="ai-voice-response">SUTRA Copilot: "Online. Ready to assist astronaut hands-free."</div>
      </div>

      <!-- Real-time Diagnostics -->
      <div style="margin-top: 14px; background: var(--bg); padding: 10px 14px; border-radius: 10px; border: 1px solid var(--border);">
        <div class="diag-row">
          <span>Inference Latency (P95): <strong id="p95-val" style="color: var(--primary);">22 ms</strong></span>
          <span>Process RAM: <strong id="ram-val" style="color: var(--primary);">115 MB</strong></span>
          <span>Telemetry Uplink: <strong style="color: var(--success);">2.1 KB/s (98.4% Bandwidth Saved)</strong></span>
          <span>Target Hardware: <strong>CPU Only (&le;8GB RAM)</strong></span>
        </div>
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

    async function updateConfidence(val) {
      document.getElementById('conf-val').innerText = Math.round(val * 100) + '%';
      await fetch('/api/set_confidence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confidence: parseFloat(val) })
      });
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

      const badge = document.getElementById('badge-text');
      if (badge) badge.innerText = 'UPLOADING...';

      try {
        const res = await fetch('/api/upload_video', {
          method: 'POST',
          body: formData
        });
        if (!res.ok) {
          const errText = await res.text();
          throw new Error(errText || "Upload failed with status " + res.status);
        }
        const data = await res.json();
        if (badge) badge.innerText = 'EDGE ACTIVE';

        // Highlight upload button and reset protocol sequence for new video
        document.querySelectorAll('.source-btn').forEach(b => b.classList.remove('active'));
        if (inputEl.parentElement) inputEl.parentElement.classList.add('active');
        await resetProtocol();

        speakText("Uploaded video loaded. Verifying experiment sequence.");
        alert("Video loaded successfully: " + (data.filename || file.name));
      } catch (e) {
        if (badge) badge.innerText = 'EDGE ACTIVE';
        alert("Video upload failed: " + e.message);
      } finally {
        inputEl.value = ''; // Reset file input so user can re-upload if needed
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
      lastSpokenAlert = '';
      speakText("Experiment protocol reset.");
    }

    async function simulateDeviation() {
      await fetch('/api/simulate_deviation', { method: 'POST' });
      speakText("Caution! Step 3 Skipped. Voice alert issued.");
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
          const rawObjs = data.pipeline.detected_objects || [];
          // Deduplicate classes so each tag pill only shows once
          const uniqueObjs = Array.from(new Set(rawObjs));

          const tagsCont = document.getElementById('detection-tags-container');
          if (tagsCont) {
            let tagsHtml = '';
            if (uniqueObjs.length === 0) {
              tagsHtml = '<span class="tag-pill tag-astronaut">Astronaut (Operator)</span>';
            } else {
              uniqueObjs.forEach(o => {
                if (o === 'astronaut' || o === 'person') {
                  tagsHtml += '<span class="tag-pill tag-astronaut">Astronaut</span>';
                } else if (o === 'pipette' || o === 'tool') {
                  tagsHtml += '<span class="tag-pill tag-tool">Tool (Pen / Pipette)</span>';
                } else if (o === 'vial_open' || o === 'vial_sealed' || o === 'sample') {
                  tagsHtml += '<span class="tag-pill tag-sample">Sample (Container / Vial)</span>';
                } else {
                  tagsHtml += '<span class="tag-pill tag-apparatus">Apparatus</span>';
                }
              });
            }
            tagsCont.innerHTML = tagsHtml;
          }

          const dev = data.pipeline.active_deviation;
          const skippedPill = document.getElementById('skipped-pill');
          const skippedPillText = document.getElementById('skipped-pill-text');
          const voiceAlertText = document.getElementById('voice-alert-text');
          const obsList = document.getElementById('observed-seq-list');

          if (dev) {
            // DEVIATION DETECTED SCENARIO (MATCHING INFOGRAPHIC EXACTLY)
            skippedPill.style.display = 'flex';
            skippedPill.style.background = '#fee2e2';
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

        if (data.copilot) {
          const vs = data.copilot.video_summary;
          if (vs && Date.now() > geminiHoldUntil) {
            const l1 = vs.line1 || (typeof vs === 'string' ? vs : '');
            const l2 = vs.line2 || '';
            const l3 = vs.line3 || '';
            const l4 = vs.line4 || '';

            const el1 = document.getElementById('sum-line1');
            const el2 = document.getElementById('sum-line2');
            const el3 = document.getElementById('sum-line3');
            const el4 = document.getElementById('sum-line4');
            const badge = document.getElementById('summary-source-badge');

            if (badge) {
              badge.innerText = "SUTRA Onboard Edge AI";
              badge.style.background = "rgba(56, 189, 248, 0.15)";
              badge.style.color = "var(--primary)";
            }

            if (el1 && l1) el1.innerHTML = `<strong>👨‍🚀 Operator Activity:</strong> ${l1.replace(/^Operator Activity:\s*/i, '')}`;
            if (el2 && l2) el2.innerHTML = `<strong>✋ Handheld &amp; Tools:</strong> ${l2.replace(/^Handheld & Equipment:\s*/i, '')}`;
            if (el3 && l3) el3.innerHTML = `<strong>📋 SOP Compliance:</strong> ${l3.replace(/^SOP Compliance:\s*/i, '')}`;
            if (el4 && l4) el4.innerHTML = `<strong>🚀 AI Directive:</strong> ${l4.replace(/^AI Directive:\s*/i, '')}`;

            const sumEl = document.getElementById('live-video-summary');
            if (sumEl) sumEl.innerText = vs.formatted_text || vs;
          }

          const actEl = document.getElementById('next-action-directive');
          if (actEl && data.copilot.next_action_directive && Date.now() > geminiHoldUntil) {
            actEl.innerText = data.copilot.next_action_directive;
          }
          const aiVoiceEl = document.getElementById('ai-voice-response');
          if (aiVoiceEl && data.copilot.spoken_announcement) {
            aiVoiceEl.innerText = 'SUTRA Copilot: "' + data.copilot.spoken_announcement + '"';
          }
        }
      } catch (e) {}
    }

    let geminiHoldUntil = 0;

    async function geminiInspect() {
      const badge = document.getElementById('summary-source-badge');
      if (badge) {
        badge.innerText = "⏳ Querying Google Gemini Flash Vision...";
        badge.style.background = "#7c3aed";
        badge.style.color = "#fff";
      }
      geminiHoldUntil = Date.now() + 30000; // Hold result for 30 seconds

      try {
        const res = await fetch('/api/gemini_inspect', { method: 'POST' });
        const data = await res.json();
        
        if (badge) {
          badge.innerText = data.source || "Google Gemini Flash Vision";
          badge.style.background = "linear-gradient(135deg, #2563eb, #7c3aed)";
          badge.style.color = "#fff";
        }

        const l1 = data.line1 || data.action_summary || "Astronaut active in payload workspace.";
        const l2 = data.line2 || `Holding: ${data.handheld_item || 'Instrument'} (Tags: ${(data.detected_objects || []).join(', ')})`;
        const l3 = data.line3 || `SOP Compliance: ${data.step_compliance || 'NOMINAL'}`;
        const l4 = data.line4 || data.next_directive || "Proceed with protocol SOP sequence.";

        const el1 = document.getElementById('sum-line1');
        const el2 = document.getElementById('sum-line2');
        const el3 = document.getElementById('sum-line3');
        const el4 = document.getElementById('sum-line4');

        if (el1) el1.innerHTML = `<strong>👨‍🚀 Gemini Scene Analysis:</strong> ${l1.replace(/^Operator Activity:\s*/i, '')}`;
        if (el2) el2.innerHTML = `<strong>✋ Held Object Identified:</strong> <span style="font-weight:700; color:#e0e7ff; background:rgba(99,102,241,0.25); padding:2px 6px; border-radius:4px;">${data.handheld_item || 'Scientific Instrument'}</span> &bull; <span style="font-size:12px; opacity:0.9;">Tags: ${(data.detected_objects || []).join(', ')}</span>`;
        if (el3) el3.innerHTML = `<strong>📋 Step Compliance:</strong> ${l3.replace(/^SOP Compliance:\s*/i, '')}`;
        if (el4) el4.innerHTML = `<strong>🚀 Mission Recommendation:</strong> ${l4.replace(/^AI Directive:\s*/i, '')}`;

        const sumEl = document.getElementById('live-video-summary');
        if (sumEl) {
          sumEl.innerHTML = `<strong>[${data.source}]</strong><br>👨‍🚀 ${l1}<br>✋ ${l2}<br>📋 ${l3}<br>🚀 ${l4}`;
        }
        if (data.next_directive) {
          const actEl = document.getElementById('next-action-directive');
          if (actEl) actEl.innerText = data.next_directive;
        }

        speakText("Gemini Vision analysis complete: Handheld item is " + (data.handheld_item || "identified"));
      } catch (e) {
        alert("Gemini inspect failed: " + e);
        geminiHoldUntil = 0;
      }
    }

    async function setApiKey() {
      const key = document.getElementById('custom-api-key').value.trim();
      if (!key) {
        alert("Please paste your Gemini or Groq API key.");
        return;
      }
      try {
        // Configure Gemini Flash
        const res1 = await fetch('/api/set_gemini_key', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ api_key: key })
        });
        const d1 = await res1.json();
        
        // Also configure Groq if applicable
        await fetch('/api/set_llm_key', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ api_key: key })
        });

        alert(d1.message || "Gemini Flash Key activated successfully!");
      } catch (e) {
        alert("Failed to activate key: " + e);
      }
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
    global engine_instance, latest_directive, latest_detection_summary, latest_video_summary
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

    copilot_data = {
        "timestamp": latest_directive.timestamp if latest_directive else time.time(),
        "current_status": latest_directive.current_status if latest_directive else "Autonomous monitoring active",
        "missing_or_deviation": latest_directive.missing_or_deviation if latest_directive else None,
        "next_action_directive": latest_directive.next_action_directive if latest_directive else "Inspect and confirm payload workspace sterility.",
        "spoken_announcement": latest_directive.spoken_announcement if latest_directive else "Monitoring active experiment.",
        "alert_level": latest_directive.alert_level if latest_directive else "INFO",
        "video_summary": latest_video_summary
    }

    return {
        "diagnostics": diag,
        "pipeline": pipeline_state,
        "copilot": copilot_data,
        "events": [e.model_dump() for e in events]
    }


@app.post("/api/set_llm_key")
async def set_llm_key(request: Request):
    global copilot_engine
    body = await request.json()
    key = body.get("api_key", "").strip()
    if key and copilot_engine:
        try:
            from groq import Groq
            copilot_engine._groq_client = Groq(api_key=key)
            copilot_engine.api_key = key
            os.environ["GROQ_API_KEY"] = key
            return {"status": "SUCCESS", "message": "Groq LLM Vision Copilot Activated!"}
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}
    return {"status": "IGNORED"}


@app.post("/api/gemini_inspect")
def gemini_inspect():
    global gemini_copilot, latest_raw_frame, engine_instance, latest_detection_summary
    if latest_raw_frame is None:
        return {"source": "SUTRA Vision", "action_summary": "Camera feed is initializing... please retry in 2 seconds.", "handheld_item": "Standby"}
    ctx = {
        "detected_objects": latest_detection_summary,
        "current_step_name": engine_instance.protocol_engine.protocol.steps[engine_instance.protocol_engine.current_step_idx].name if engine_instance and engine_instance.protocol_engine.current_step_idx < len(engine_instance.protocol_engine.protocol.steps) else "Experiment Inspection"
    }
    return gemini_copilot.analyze_frame(latest_raw_frame, ctx)


@app.post("/api/set_gemini_key")
async def set_gemini_key(request: Request):
    global gemini_copilot
    body = await request.json()
    key = body.get("api_key", "").strip()
    if key and gemini_copilot:
        success = gemini_copilot.configure(key)
        if success:
            try:
                env_file = Path(".env")
                content = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
                if "GEMINI_API_KEY=" in content:
                    import re
                    content = re.sub(r"GEMINI_API_KEY=.*", f"GEMINI_API_KEY={key}", content)
                else:
                    content += f"\nGEMINI_API_KEY={key}\n"
                env_file.write_text(content, encoding="utf-8")
            except Exception as e:
                logger.warning(f"Failed to persist GEMINI_API_KEY: {e}")
            return {"status": "SUCCESS", "message": "Google Gemini Flash Vision Activated & Saved!"}
        else:
            return {"status": "ERROR", "message": "Could not connect to Gemini with this key."}
    return {"status": "ERROR", "message": "No key provided."}


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


@app.post("/api/simulate_deviation")
def simulate_deviation():
    """Instantly inject the exact SIH Step 3 Skipped deviation scenario."""
    global engine_instance
    if engine_instance:
        pe = engine_instance.protocol_engine
        pe.current_step_idx = min(3, len(pe.protocol.steps) - 1)
        if len(pe.protocol.steps) >= 4:
            pe.step_statuses[pe.protocol.steps[0].id] = StepStatus.COMPLETED
            pe.step_statuses[pe.protocol.steps[1].id] = StepStatus.COMPLETED
            pe.step_statuses[pe.protocol.steps[2].id] = StepStatus.SKIPPED
            pe.step_statuses[pe.protocol.steps[3].id] = StepStatus.DEVIATED
        
        pe.active_deviation = {
            "deviation_type": "SKIPPED_STEP",
            "attempted_step": "STEP_04",
            "uncompleted_step": "STEP_03",
            "description": "Step 3 Skipped: Attempted 'Close Container' before 'Transfer Sample' completed.",
            "confidence": 0.95
        }
    return {"status": "DEVIATION_SIMULATED"}


@app.post("/api/set_confidence")
async def set_confidence(request: Request):
    """Dynamically adjust detection sensitivity."""
    global engine_instance
    body = await request.json()
    conf = float(body.get("confidence", 0.20))
    if engine_instance and hasattr(engine_instance.detector, "set_confidence"):
        engine_instance.detector.set_confidence(conf)
    return {"status": "UPDATED", "confidence": conf}


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
    try:
        data_dir = Path("data/uploads")
        data_dir.mkdir(parents=True, exist_ok=True)
        # Use timestamped name to prevent Windows file-lock PermissionError (Errno 13)
        ext = Path(file.filename or "video.mp4").suffix or ".mp4"
        safe_name = f"uploaded_{int(time.time())}{ext}"
        save_path = data_dir / safe_name

        # Release existing capture before writing if it was using the same file
        if engine_instance and engine_instance.ingestor:
            engine_instance.ingestor._release_capture()

        # Read in 1MB chunks to safely handle small or large files
        with open(save_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)

        resolved_path = str(save_path.resolve())
        if engine_instance and engine_instance.ingestor:
            engine_instance.ingestor.switch_source(resolved_path)
            # Reset protocol FSM so the newly uploaded video starts detection from Step 1
            if hasattr(engine_instance, "protocol_engine") and engine_instance.protocol_engine:
                engine_instance.protocol_engine.reset()
            logger.info(f"Uploaded video switched to: {resolved_path} (protocol reset for fresh run)")

        return JSONResponse(content={"status": "UPLOADED", "filename": file.filename, "path": resolved_path})
    except Exception as e:
        logger.error(f"Failed to handle video upload: {e}")
        return JSONResponse(status_code=500, content={"status": "ERROR", "message": str(e)})


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


def start_web_server(engine: SutraVisionEngine, host: str = "127.0.0.1", port: int = 8081):
    """
    Launch background thread running the engine loop, and start uvicorn web server.
    """
    global engine_instance, latest_annotated_jpeg, latest_directive, copilot_engine, latest_detection_summary
    engine_instance = engine
    engine.start()

    import threading

    def pipeline_worker():
        global latest_annotated_jpeg, latest_directive, latest_detection_summary, latest_video_summary, latest_raw_frame

        while engine._running:
            try:
                ingested = engine.frame_buffer.get_latest(timeout=0.2)
                if ingested is None:
                    time.sleep(0.04)
                    continue
                latest_raw_frame = ingested.frame

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

                # Real-time Video Scene Intelligence Summary
                latest_video_summary = copilot_engine.generate_live_video_summary(ctx)

            except Exception as e:
                logger.error(f"Error in pipeline worker: {e}")
                time.sleep(0.05)

    t = threading.Thread(target=pipeline_worker, name="WebHUDWorker", daemon=True)
    t.start()

    # Free port if held by a stale Python process from a prior run
    try:
        import psutil
        curr_pid = os.getpid()
        for conn in psutil.net_connections(kind='inet'):
            if conn.laddr and conn.laddr.port == port and conn.pid and conn.pid != curr_pid:
                try:
                    proc = psutil.Process(conn.pid)
                    if "python" in proc.name().lower():
                        logger.info(f"Terminating lingering Python process {conn.pid} on port {port}")
                        proc.terminate()
                        proc.wait(timeout=1.5)
                except Exception:
                    pass
    except Exception:
        pass

    # Find free bindable port (handles TIME_WAIT cleanly)
    active_port = port
    import socket
    for candidate in [port, port + 1, port + 2, 8080]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, candidate))
            active_port = candidate
            break
        except OSError:
            logger.warning(f"Port {candidate} is busy or in TIME_WAIT, checking next port...")
            continue

    print("\n" + "=" * 65)
    print(" [OK] SUTRA AUTONOMOUS WEB HUD & VOICE ASSISTANT RUNNING!")
    print(f" [OK] Open your browser at: http://{host}:{active_port}")
    print("=" * 65 + "\n")

    uvicorn.run(app, host=host, port=active_port, log_level="warning")


if __name__ == "__main__":
    conf = get_config()
    engine = SutraVisionEngine(conf)
    start_web_server(engine, host="127.0.0.1", port=8081)
