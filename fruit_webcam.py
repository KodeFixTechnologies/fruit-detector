#!/usr/bin/env python3
"""
Fruit and vegetable webcam viewer.

Features:
  - Uses the laptop webcam through OpenCV.
  - Runs periodic vision detection on the latest frame.
  - Draws bounding boxes for visible fruits and vegetables.
  - Shows scene context and detection summaries in a simple web UI.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import re
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request
from google import genai
from google.genai import types


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-robotics-er-1.5-preview"
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "960"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "540"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "75"))
PORT = int(os.getenv("PORT", "5050"))
DETECT_INTERVAL = float(os.getenv("DETECT_INTERVAL", "2.0"))

PROMPT = """
You are analyzing a live farm or kitchen-style camera image.
Detect only visible fruits or vegetables in the image.
Detect them whether they are:
- real physical fruits or vegetables
- shown on a mobile phone, tablet, laptop, TV, or monitor screen
- shown in a printed image, poster, package, or label

Return ONLY valid JSON with no markdown and no explanation:
{
  "scene_description": "one short sentence describing the scene",
  "detections": [
    {
      "label": "tomato",
      "box_2d": [ymin, xmin, ymax, xmax],
      "state": "ripe/unripe/unknown",
      "appearance_source": "real_object/screen_image/printed_image/unknown",
      "notes": "short phrase"
    }
  ]
}

Rules:
- Include only fruits or vegetables.
- Do not return phones, screens, bowls, hands, or other non-produce objects.
- If the fruit or vegetable is visible on a mobile screen, monitor, or printed image,
  still detect it as the fruit or vegetable itself.
- box_2d values must be integers normalized from 0 to 1000.
- If nothing relevant is visible, return an empty detections array.
- Keep scene_description short.
- Keep notes short.
""".strip()


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    return re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def parse_detection_payload(text: str) -> Dict[str, Any]:
    cleaned = strip_json_fences(text)
    candidates: List[str] = [cleaned]
    object_match = re.search(r"\{[\s\S]*\}", cleaned)
    if object_match:
        candidates.append(object_match.group(0))

    payload: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break

    if payload is None:
        raise ValueError("Vision response did not contain a JSON object")

    scene_description = payload.get("scene_description", "")
    if not isinstance(scene_description, str):
        scene_description = ""

    detections: List[Dict[str, Any]] = []
    raw_detections = payload.get("detections", [])
    if isinstance(raw_detections, list):
        for item in raw_detections:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            box = item.get("box_2d")
            state = item.get("state", "unknown")
            appearance_source = item.get("appearance_source", "unknown")
            notes = item.get("notes", "")
            if not isinstance(label, str):
                continue
            if not isinstance(box, list) or len(box) != 4:
                continue
            try:
                ymin, xmin, ymax, xmax = [int(v) for v in box]
            except (TypeError, ValueError):
                continue
            detections.append(
                {
                    "label": label.strip() or "unknown",
                    "box_2d": [
                        clamp(ymin, 0, 1000),
                        clamp(xmin, 0, 1000),
                        clamp(ymax, 0, 1000),
                        clamp(xmax, 0, 1000),
                    ],
                    "state": state if isinstance(state, str) else "unknown",
                    "appearance_source": (
                        appearance_source if isinstance(appearance_source, str) else "unknown"
                    ),
                    "notes": notes if isinstance(notes, str) else "",
                }
            )

    return {"scene_description": scene_description, "detections": detections}


def box_to_pixels(box: List[int], width: int, height: int) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = box
    x1 = clamp(int(xmin / 1000.0 * width), 0, width - 1)
    y1 = clamp(int(ymin / 1000.0 * height), 0, height - 1)
    x2 = clamp(int(xmax / 1000.0 * width), 0, width - 1)
    y2 = clamp(int(ymax / 1000.0 * height), 0, height - 1)
    return x1, y1, x2, y2


def detection_area(box: List[int]) -> int:
    ymin, xmin, ymax, xmax = box
    return max(0, ymax - ymin) * max(0, xmax - xmin)


def best_detection(detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not detections:
        return None

    def score(item: Dict[str, Any]) -> tuple[int, int]:
        state = str(item.get("state", "unknown")).lower()
        state_score = 2 if state == "ripe" else 1 if state == "unknown" else 0
        box = item.get("box_2d")
        area = detection_area(box) if isinstance(box, list) and len(box) == 4 else 0
        return state_score, area

    return max(detections, key=score)


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_frame: Optional[Any] = None
        self.last_jpeg: Optional[bytes] = None
        self.detections: List[Dict[str, Any]] = []
        self.scene_description = "Waiting for first scan"
        self.logs: List[str] = []
        self.status = "Starting..."
        self.running = True
        self.gemini_ok = False
        self.camera_ok = False
        self.last_scan_at = 0.0
        self.last_completed_scan_at = 0.0
        self.last_latency_ms: Optional[int] = None
        self.request_in_flight = False
        self.recent_snapshots: List[Dict[str, str]] = []


S = AppState()


def log(msg: str, level: str = "INFO") -> None:
    entry = f"[{time.strftime('%H:%M:%S')}][{level}] {msg}"
    print(entry, flush=True)
    with S.lock:
        S.logs.append(entry)
        if len(S.logs) > 100:
            S.logs.pop(0)


class GeminiDetector:
    def __init__(self) -> None:
        self.client: Optional[genai.Client] = None

    def init(self) -> bool:
        if not GOOGLE_API_KEY:
            log("No API key found", "ERROR")
            return False
        try:
            self.client = genai.Client(api_key=GOOGLE_API_KEY)
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents="ping",
            )
            with S.lock:
                S.gemini_ok = True
            log(f"Vision model ready: {(response.text or '')[:40]}", "SUCCESS")
            return True
        except Exception as exc:
            with S.lock:
                S.gemini_ok = False
            log(f"Vision init failed: {exc}", "ERROR")
            return False

    def detect(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        if self.client is None:
            raise RuntimeError("Vision client not initialized")
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return parse_detection_payload(response.text or "")


detector = GeminiDetector()


class LaptopCamera:
    def __init__(self) -> None:
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            log("Failed to open laptop camera", "ERROR")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap = cap
        with S.lock:
            S.camera_ok = True
        log(f"Laptop camera ready on index {CAMERA_INDEX}", "SUCCESS")
        return True

    def read(self) -> Optional[Any]:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()


camera = LaptopCamera()


def encode_jpeg(frame: Any) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode frame")
    return buffer.tobytes()


def draw_overlay(frame: Any, detections: List[Dict[str, Any]]) -> Any:
    out = frame.copy()
    height, width = out.shape[:2]
    primary = best_detection(detections)
    for det in detections:
        box = det.get("box_2d")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = box_to_pixels(box, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        state = str(det.get("state", "unknown")).lower()
        source = str(det.get("appearance_source", "unknown")).lower()
        color = (0, 255, 0) if state == "ripe" else (0, 165, 255) if state == "unripe" else (255, 220, 0)
        thickness = 4 if det is primary else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = det.get("label", "unknown")
        notes = det.get("notes", "")
        caption = f"{label} | {state}"
        if source == "screen_image":
            caption = f"{caption} | screen"
        elif source == "printed_image":
            caption = f"{caption} | print"
        if isinstance(notes, str) and notes.strip():
            caption = f"{caption} | {notes.strip()[:20]}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        top = max(0, y1 - th - baseline - 8)
        cv2.rectangle(out, (x1, top), (min(x1 + tw + 10, width - 1), y1), color, cv2.FILLED)
        cv2.putText(
            out,
            caption,
            (x1 + 5, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        if det is primary:
            cv2.putText(
                out,
                "PRIMARY",
                (x1, min(height - 10, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
    return out


def make_snapshot(frame: Any, detections: List[Dict[str, Any]]) -> Optional[str]:
    preview = draw_overlay(frame, detections)
    height, width = preview.shape[:2]
    target_width = 300
    if width > target_width:
        target_height = max(1, int(height * (target_width / width)))
        preview = cv2.resize(preview, (target_width, target_height))
    ok, buffer = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 60])
    if not ok:
        return None
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def store_snapshot(frame: Any, detections: List[Dict[str, Any]], scene_description: str) -> None:
    image = make_snapshot(frame, detections)
    if image is None:
        return
    primary = best_detection(detections)
    title = "No detection"
    if primary is not None:
        title = f"{primary.get('label', 'unknown')} | {primary.get('state', 'unknown')}"
    snapshot = {
        "image": image,
        "title": title,
        "subtitle": scene_description[:60] if scene_description else "No scene summary",
        "time": time.strftime("%H:%M:%S"),
    }
    with S.lock:
        S.recent_snapshots.insert(0, snapshot)
        S.recent_snapshots = S.recent_snapshots[:4]


def run_detection(frame: Any) -> None:
    started_at = time.time()
    with S.lock:
        if S.request_in_flight or not S.gemini_ok:
            return
        S.request_in_flight = True
        S.status = "Scanning scene..."
        S.last_scan_at = time.time()

    try:
        jpeg = encode_jpeg(frame)
        payload = detector.detect(jpeg)
        detections = payload["detections"]
        scene_description = payload["scene_description"] or "No extra context"
        latency_ms = int((time.time() - started_at) * 1000)
        with S.lock:
            S.detections = detections
            S.scene_description = scene_description
            S.status = f"Detected {len(detections)} fruits/vegetables"
            S.last_latency_ms = latency_ms
            S.last_completed_scan_at = time.time()
        if detections:
            store_snapshot(frame, detections, scene_description)
        log(f"Detected {len(detections)} item(s)", "VISION")
    except Exception as exc:
        with S.lock:
            S.status = "Detection failed"
            S.last_latency_ms = int((time.time() - started_at) * 1000)
            S.last_completed_scan_at = time.time()
        log(f"Detection failed: {exc}", "ERROR")
    finally:
        with S.lock:
            S.request_in_flight = False


def capture_loop() -> None:
    log("Capture loop started", "INFO")
    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.05)
            continue

        with S.lock:
            S.last_frame = frame.copy()
            detections = list(S.detections)
            running = S.running
            gemini_ok = S.gemini_ok
            last_scan_at = S.last_scan_at
            request_in_flight = S.request_in_flight

        if running and gemini_ok and not request_in_flight and (time.time() - last_scan_at) >= DETECT_INTERVAL:
            threading.Thread(target=run_detection, args=(frame.copy(),), daemon=True).start()

        display = draw_overlay(frame, detections)
        try:
            jpeg = encode_jpeg(display)
            with S.lock:
                S.last_jpeg = jpeg
        except Exception as exc:
            log(f"Preview encoding failed: {exc}", "WARN")

        time.sleep(0.03)


app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AgroPick Vision Model</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#140d09;--card:#20130f;--border:#3b251c;--accent:#ff6842;
  --accent-2:#7ad36b;--text:#f8eee9;--muted:#c1a89d;--surface:#271915;
  --warn:#ffbe55;--bad:#f25f5c;--panel:#1a110d;
}
body{
  font-family:'Space Grotesk',sans-serif;
  background:
    radial-gradient(circle at top left, rgba(255,104,66,.18), transparent 32%),
    radial-gradient(circle at top right, rgba(122,211,107,.14), transparent 28%),
    linear-gradient(180deg, #1b100c 0%, #140d09 60%, #100906 100%);
  color:var(--text);min-height:100vh
}
.header{
  display:flex;justify-content:space-between;align-items:center;padding:14px 22px;
  border-bottom:1px solid var(--border);background:rgba(20,13,9,.88);backdrop-filter:blur(10px);
}
.brand{display:flex;flex-direction:column;gap:4px}
.logo{font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:700;color:var(--accent);letter-spacing:.02em}
.brand-sub{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1.3px}
.status{display:flex;gap:16px;font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--muted)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:var(--bad)}
.dot.on{background:var(--accent-2)}
.layout{max-width:1380px;margin:0 auto;padding:18px;display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:18px}
@media(max-width:960px){.layout{grid-template-columns:1fr}}
.card{background:rgba(32,19,15,.94);border:1px solid var(--border);border-radius:18px;padding:16px;box-shadow:0 18px 40px rgba(0,0,0,.22)}
.feed{width:100%;border-radius:12px;display:block;background:#000}
.controls{display:flex;gap:10px;margin-top:14px}
.btn{
  border:none;border-radius:10px;padding:11px 14px;cursor:pointer;font-weight:700;
  font-size:13px;font-family:'JetBrains Mono',monospace;
}
.btn-primary{background:linear-gradient(135deg,var(--accent),#ff8b67);color:#230f08}
.btn-secondary{background:var(--surface);color:var(--text)}
.section-title{
  font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:1px;
  color:var(--muted);text-transform:uppercase;margin-bottom:10px
}
.status-line{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--text);margin-bottom:10px}
.context{
  background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px;
  min-height:88px;font-size:14px;line-height:1.5
}
.hero{
  background:
    radial-gradient(circle at top right, rgba(122,211,107,.16), transparent 36%),
    linear-gradient(135deg,rgba(255,104,66,.18),rgba(255,104,66,.05));
  border:1px solid rgba(255,104,66,.28);border-radius:16px;padding:16px;margin-bottom:14px
}
.hero-kicker{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1px}
.hero-title{font-size:28px;font-weight:700;margin-top:6px;line-height:1.05}
.hero-sub{margin-top:6px;color:var(--muted);font-size:13px}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}
.metric{
  background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:12px
}
.metric-label{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);text-transform:uppercase}
.metric-value{font-size:20px;font-weight:700;margin-top:5px}
.detect-list{display:grid;gap:10px;margin-top:12px}
.detect-item{
  background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:12px
}
.detect-top{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:6px}
.badge{
  border-radius:999px;padding:3px 8px;font-size:11px;font-family:'JetBrains Mono',monospace;background:#233247;color:var(--text)
}
.badge.ripe{background:rgba(122,211,107,.18);color:var(--accent-2)}
.badge.unripe{background:rgba(255,190,85,.18);color:var(--warn)}
.badge.source{background:rgba(255,104,66,.12);color:#ffb39e}
.mono{font-family:'JetBrains Mono',monospace}
.log-box{
  margin-top:14px;background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:10px;height:180px;overflow:auto;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.45
}
.log-line{padding:2px 0;border-bottom:1px solid rgba(127,144,170,.1)}
.gallery{display:grid;gap:10px;margin-top:12px}
.shot{
  background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:10px
}
.shot img{width:100%;border-radius:8px;display:block;background:#000}
.shot-title{font-weight:700;margin-top:8px}
.shot-sub{font-size:12px;color:var(--muted);margin-top:4px}
</style>
</head>
<body>
<div class="header">
  <div class="brand">
    <div class="logo">AgroPick</div>
    <div class="brand-sub">Vision Model by AgroPick</div>
  </div>
  <div class="status">
    <span><span class="dot" id="camDot"></span>Camera</span>
    <span><span class="dot" id="gemDot"></span>Vision</span>
    <span id="runState">RUNNING</span>
  </div>
</div>
<div class="layout">
  <div class="card">
    <img src="/video_feed" class="feed" alt="Webcam feed">
    <div class="controls">
      <button class="btn btn-primary" id="toggleBtn" onclick="toggleRunning()">Pause Scan</button>
      <button class="btn btn-secondary" onclick="scanNow()">Scan Now</button>
    </div>
  </div>
  <div class="card">
    <div class="hero">
      <div class="hero-kicker">Top Harvest Candidate</div>
      <div class="hero-title" id="heroTitle">Waiting...</div>
      <div class="hero-sub" id="heroSub">AgroPick highlights the strongest fruit or vegetable in view.</div>
    </div>
    <div class="metrics">
      <div class="metric">
        <div class="metric-label">Last Scan</div>
        <div class="metric-value" id="lastScanValue">-</div>
      </div>
      <div class="metric">
        <div class="metric-label">Latency</div>
        <div class="metric-value" id="latencyValue">-</div>
      </div>
    </div>
    <div class="section-title">Status</div>
    <div class="status-line" id="statusLine">Starting...</div>
    <div class="section-title">Scene Context</div>
    <div class="context" id="sceneBox">Waiting for first scan...</div>
    <div class="section-title" style="margin-top:14px">Detections</div>
    <div class="detect-list" id="detectList"></div>
    <div class="section-title" style="margin-top:14px">Recent Snapshots</div>
    <div class="gallery" id="gallery"></div>
    <div class="section-title" style="margin-top:14px">Logs</div>
    <div class="log-box" id="logBox"></div>
  </div>
</div>
<script>
let running = true;
function api(path, body){
  const opts = body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {};
  return fetch('/api/' + path, opts).then(r => r.json());
}
function toggleRunning(){
  running = !running;
  api('running', {running}).then(refreshStatus);
}
function scanNow(){
  api('scan', {});
}
function formatLastScan(ts){
  if(!ts){ return '-'; }
  const seconds = Math.max(0, Math.round(Date.now() / 1000 - ts));
  return seconds === 0 ? 'just now' : `${seconds}s ago`;
}
function formatSource(value){
  const source = (value || 'unknown').toLowerCase();
  if(source === 'screen_image') return 'screen';
  if(source === 'printed_image') return 'printed';
  if(source === 'real_object') return 'real';
  return 'unknown';
}
function renderHero(best){
  const title = document.getElementById('heroTitle');
  const sub = document.getElementById('heroSub');
  if(!best){
    title.textContent = 'No target';
    sub.textContent = 'No fruit or vegetable is currently highlighted.';
    return;
  }
  const state = (best.state || 'unknown').toLowerCase();
  const source = formatSource(best.appearance_source);
  title.textContent = `${best.label} | ${state}`;
  sub.textContent = best.notes || `Source: ${source}. This is the strongest current detection.`;
}
function renderDetections(items){
  const root = document.getElementById('detectList');
  if(!items.length){
    root.innerHTML = '<div class="detect-item">No fruits or vegetables detected.</div>';
    return;
  }
  root.innerHTML = items.map((item, index) => {
    const state = (item.state || 'unknown').toLowerCase();
    const badgeClass = state === 'ripe' ? 'badge ripe' : state === 'unripe' ? 'badge unripe' : 'badge';
    const source = formatSource(item.appearance_source);
    return `
      <div class="detect-item">
        <div class="detect-top">
          <strong>${index + 1}. ${item.label}</strong>
          <div style="display:flex;gap:6px;align-items:center">
            <span class="badge source">${source}</span>
            <span class="${badgeClass}">${state}</span>
          </div>
        </div>
        <div>${item.notes || 'No extra notes'}</div>
        <div class="mono" style="margin-top:6px">box: ${JSON.stringify(item.box_2d || [])}</div>
      </div>
    `;
  }).join('');
}
function renderGallery(items){
  const root = document.getElementById('gallery');
  if(!items.length){
    root.innerHTML = '<div class="shot">No saved detections yet.</div>';
    return;
  }
  root.innerHTML = items.map(item => `
    <div class="shot">
      <img src="${item.image}" alt="${item.title}">
      <div class="shot-title">${item.title}</div>
      <div class="shot-sub">${item.time} | ${item.subtitle}</div>
    </div>
  `).join('');
}
function renderLogs(lines){
  const root = document.getElementById('logBox');
  root.innerHTML = lines.map(line => `<div class="log-line">${line}</div>`).join('');
  root.scrollTop = root.scrollHeight;
}
function refreshStatus(){
  api('status').then(data => {
    running = data.running;
    document.getElementById('camDot').className = 'dot' + (data.camera_ok ? ' on' : '');
    document.getElementById('gemDot').className = 'dot' + (data.gemini_ok ? ' on' : '');
    document.getElementById('runState').textContent = data.running ? 'RUNNING' : 'PAUSED';
    document.getElementById('statusLine').textContent = data.status;
    document.getElementById('sceneBox').textContent = data.scene_description || 'No scene description';
    document.getElementById('toggleBtn').textContent = data.running ? 'Pause Scan' : 'Resume Scan';
    document.getElementById('lastScanValue').textContent = formatLastScan(data.last_completed_scan_at);
    document.getElementById('latencyValue').textContent = data.last_latency_ms ? `${data.last_latency_ms} ms` : '-';
    renderHero(data.best_detection || null);
    renderDetections(data.detections || []);
    renderGallery(data.recent_snapshots || []);
  }).catch(() => {});
}
function refreshLogs(){
  api('logs').then(data => renderLogs(data.logs || [])).catch(() => {});
}
setInterval(refreshStatus, 800);
setInterval(refreshLogs, 1200);
refreshStatus();
refreshLogs();
</script>
</body>
</html>
"""


@app.route("/")
def index() -> str:
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed() -> Response:
    def generate() -> Any:
        while True:
            with S.lock:
                frame = S.last_jpeg
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.033)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status() -> Response:
    with S.lock:
        return jsonify(
            {
                "running": S.running,
                "camera_ok": S.camera_ok,
                "gemini_ok": S.gemini_ok,
                "status": S.status,
                "scene_description": S.scene_description,
                "detections": list(S.detections),
                "best_detection": best_detection(list(S.detections)),
                "last_latency_ms": S.last_latency_ms,
                "last_completed_scan_at": S.last_completed_scan_at,
                "recent_snapshots": list(S.recent_snapshots),
            }
        )


@app.route("/api/logs")
def api_logs() -> Response:
    with S.lock:
        return jsonify({"logs": S.logs[-60:]})


@app.route("/api/running", methods=["POST"])
def api_running() -> Response:
    data = request.get_json(force=True) or {}
    running = bool(data.get("running", True))
    with S.lock:
        S.running = running
        S.status = "Scanning enabled" if running else "Scanning paused"
    log("Scanning resumed" if running else "Scanning paused", "INFO")
    return jsonify({"ok": True})


@app.route("/api/scan", methods=["POST"])
def api_scan() -> Response:
    with S.lock:
        frame = None if S.last_frame is None else S.last_frame.copy()
        gemini_ok = S.gemini_ok
    if frame is None or not gemini_ok:
        return jsonify({"ok": False})
    threading.Thread(target=run_detection, args=(frame,), daemon=True).start()
    return jsonify({"ok": True})


def main() -> None:
    log("Harvest vision starting", "INFO")
    if not camera.start():
        return
    detector.init()
    threading.Thread(target=capture_loop, daemon=True).start()
    log(f"Open http://0.0.0.0:{PORT}", "SUCCESS")
    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)


if __name__ == "__main__":
    try:
        main()
    finally:
        camera.stop()
