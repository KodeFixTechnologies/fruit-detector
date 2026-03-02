from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "gemini-robotics-er-1.5-preview"
REQUEST_INTERVAL_SECONDS = 2.0
WINDOW_NAME = "Gemini Object Detector"
BOX_THICKNESS = 2
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2

# Distance thresholds based on bounding box area (% of frame)
# Tune these after testing with your actual setup
DISTANCE_ZONES = [
    (10.0, "VERY CLOSE",  (0, 0, 255)),   # >10% of frame = red
    (4.0,  "CLOSE",       (0, 165, 255)), # >4%  = orange
    (1.5,  "MEDIUM",      (0, 255, 255)), # >1.5% = yellow
    (0.0,  "FAR",         (0, 255, 0)),   # rest  = green
]

PROMPT_DETECT = """
Return bounding boxes as a JSON array with labels.
Never return masks or code fencing.
Limit to 25 objects.
Include every object you can identify in the image.
If an object is present multiple times, distinguish labels by position,
size, color, or another visible characteristic.

The format must be:
[{"box_2d": [ymin, xmin, ymax, xmax], "label": "object label"}]

Rules:
- box_2d values must be integers.
- Coordinates are normalized to 0-1000.
- If nothing visible, return [].
""".strip()

PROMPT_NAVIGATE = """
You are a rover navigation system. Analyze this image and return a JSON object.

Identify:
1. All obstacles in the path
2. Safe trajectory points to move forward (avoid obstacles)
3. Any fruits or vegetables visible and their approximate position

Return ONLY this JSON format, no explanation:
{
  "obstacles": [{"label": "obstacle name", "box_2d": [ymin, xmin, ymax, xmax], "urgency": "HIGH/MEDIUM/LOW"}],
  "trajectory": [{"point": [y, x], "label": "step number"}],
  "targets": [{"label": "fruit/veg name", "box_2d": [ymin, xmin, ymax, xmax], "pick_point": [y, x]}],
  "rover_command": "FORWARD/STOP/TURN_LEFT/TURN_RIGHT/PICK"
}

Coordinates normalized to 0-1000. box_2d integers only.
""".strip()


# ── Distance Estimation ───────────────────────────────────────────────────────
def estimate_distance(box: list[int], frame_width: int, frame_height: int) -> tuple[str, tuple]:
    """Estimate distance based on bounding box area as % of frame."""
    ymin, xmin, ymax, xmax = box
    box_w = (xmax - xmin) / 1000.0 * frame_width
    box_h = (ymax - ymin) / 1000.0 * frame_height
    box_area = box_w * box_h
    frame_area = frame_width * frame_height
    area_pct = (box_area / frame_area) * 100.0

    for threshold, label, color in DISTANCE_ZONES:
        if area_pct >= threshold:
            return label, color, area_pct

    return "FAR", (0, 255, 0), area_pct


# ── Detection State (thread-safe) ─────────────────────────────────────────────
class DetectionState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.detections: list[dict[str, Any]] = []
        self.nav_data: dict[str, Any] = {}
        self.status = "Waiting for first response..."
        self.last_requested_at = 0.0
        self.request_in_flight = False
        self.mode = "DETECT"  # DETECT or NAVIGATE

    def snapshot(self):
        with self.lock:
            return list(self.detections), dict(self.nav_data), self.status, self.mode

    def should_request(self, now: float) -> bool:
        with self.lock:
            if self.request_in_flight:
                return False
            return now - self.last_requested_at >= REQUEST_INTERVAL_SECONDS

    def mark_requested(self, now: float) -> None:
        with self.lock:
            self.last_requested_at = now
            self.request_in_flight = True
            self.status = "Analyzing frame..."

    def set_detections(self, detections, status):
        with self.lock:
            self.detections = detections
            self.status = status
            self.request_in_flight = False

    def set_nav(self, nav_data, status):
        with self.lock:
            self.nav_data = nav_data
            self.status = status
            self.request_in_flight = False

    def toggle_mode(self):
        with self.lock:
            self.mode = "NAVIGATE" if self.mode == "DETECT" else "DETECT"
            return self.mode


# ── Helpers ───────────────────────────────────────────────────────────────────
def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
    return cleaned


def parse_detections(text: str) -> list[dict[str, Any]]:
    cleaned = strip_json_fences(text)
    match = re.search(r"\[[\s\S]*\]", cleaned)
    if not match:
        raise ValueError("No JSON array found.")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("Not a JSON array.")

    detections = []
    for item in data:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        box = item.get("box_2d")
        if not isinstance(label, str):
            continue
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            ymin, xmin, ymax, xmax = [int(v) for v in box]
        except (TypeError, ValueError):
            continue
        detections.append({
            "label": label.strip() or "unknown",
            "box_2d": [
                clamp(ymin, 0, 1000), clamp(xmin, 0, 1000),
                clamp(ymax, 0, 1000), clamp(xmax, 0, 1000),
            ],
        })
    return detections


def parse_nav(text: str) -> dict:
    cleaned = strip_json_fences(text)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ValueError("No JSON object found.")
    return json.loads(match.group(0))


def normalized_to_pixels(box, width, height):
    ymin, xmin, ymax, xmax = box
    x1 = clamp(int(xmin / 1000.0 * width), 0, width - 1)
    y1 = clamp(int(ymin / 1000.0 * height), 0, height - 1)
    x2 = clamp(int(xmax / 1000.0 * width), 0, width - 1)
    y2 = clamp(int(ymax / 1000.0 * height), 0, height - 1)
    return x1, y1, x2, y2


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_label_box(frame, label, x1, y1, color):
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS
    )
    top = max(y1 - th - baseline - 8, 0)
    bottom = min(top + th + baseline + 8, frame.shape[0] - 1)
    right = min(x1 + tw + 10, frame.shape[1] - 1)
    cv2.rectangle(frame, (x1, top), (right, bottom), color, cv2.FILLED)
    cv2.putText(
        frame, label, (x1 + 5, bottom - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 0, 0),
        TEXT_THICKNESS, cv2.LINE_AA,
    )


def draw_detections(frame, detections):
    """Draw boxes with distance estimation."""
    height, width = frame.shape[:2]

    # Draw center crosshair
    cx, cy = width // 2, height // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

    for det in detections:
        box = det.get("box_2d")
        label = det.get("label", "unknown")
        if not isinstance(box, list) or len(box) != 4:
            continue

        x1, y1, x2, y2 = normalized_to_pixels(box, width, height)
        if x2 <= x1 or y2 <= y1:
            continue

        dist_label, color, area_pct = estimate_distance(box, width, height)

        # Draw bounding box with distance color
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Draw center point of object
        obj_cx = (x1 + x2) // 2
        obj_cy = (y1 + y2) // 2
        cv2.circle(frame, (obj_cx, obj_cy), 4, color, -1)

        # Draw label + distance
        display_label = f"{label} [{dist_label}]"
        draw_label_box(frame, display_label, x1, y1, color)

        # Draw area % (useful for calibration)
        cv2.putText(
            frame, f"{area_pct:.1f}%", (x1, y2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )


def draw_nav(frame, nav_data):
    """Draw navigation overlay."""
    height, width = frame.shape[:2]

    # Draw obstacles in red
    for obs in nav_data.get("obstacles", []):
        box = obs.get("box_2d")
        label = obs.get("label", "obstacle")
        urgency = obs.get("urgency", "LOW")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = normalized_to_pixels(box, width, height)
        color = (0, 0, 255) if urgency == "HIGH" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label_box(frame, f"⚠ {label}", x1, y1, color)

    # Draw trajectory points in cyan
    prev_pt = None
    for pt_data in nav_data.get("trajectory", []):
        pt = pt_data.get("point")
        if not pt or len(pt) != 2:
            continue
        px = clamp(int(pt[1] / 1000.0 * width), 0, width - 1)
        py = clamp(int(pt[0] / 1000.0 * height), 0, height - 1)
        cv2.circle(frame, (px, py), 6, (255, 255, 0), -1)
        if prev_pt:
            cv2.line(frame, prev_pt, (px, py), (255, 255, 0), 2)
        prev_pt = (px, py)

    # Draw pick targets in green
    for target in nav_data.get("targets", []):
        box = target.get("box_2d")
        label = target.get("label", "target")
        pick = target.get("pick_point")
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = normalized_to_pixels(box, width, height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_label_box(frame, f"🎯 {label}", x1, y1, (0, 255, 0))
        if pick and len(pick) == 2:
            px = clamp(int(pick[1] / 1000.0 * width), 0, width - 1)
            py = clamp(int(pick[0] / 1000.0 * height), 0, height - 1)
            cv2.drawMarker(frame, (px, py), (0, 255, 0),
                           cv2.MARKER_CROSS, 20, 2)

    # Draw rover command banner
    command = nav_data.get("rover_command", "")
    if command:
        cmd_colors = {
            "FORWARD": (0, 255, 0),
            "STOP": (0, 0, 255),
            "TURN_LEFT": (255, 165, 0),
            "TURN_RIGHT": (255, 165, 0),
            "PICK": (255, 0, 255),
        }
        color = cmd_colors.get(command, (255, 255, 255))
        cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"ROVER: {command}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Mock robot commands — replace with real Pi GPIO later
        mock_robot_command(command)


def draw_status(frame, status, mode):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 36), (w, h), (0, 0, 0), cv2.FILLED)
    mode_color = (0, 255, 255) if mode == "NAVIGATE" else (0, 255, 0)
    cv2.putText(frame, f"[{mode}] {status}", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "M=mode Q=quit", (w - 150, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)


# ── Mock Robot Commands (replace with real GPIO on Pi) ────────────────────────
def mock_robot_command(command: str):
    """Prints mock commands. Replace with real motor/servo control on Pi."""
    actions = {
        "FORWARD":    "🚗 Motors: LEFT=FWD RIGHT=FWD",
        "STOP":       "🛑 Motors: LEFT=STOP RIGHT=STOP",
        "TURN_LEFT":  "↰  Motors: LEFT=BACK RIGHT=FWD",
        "TURN_RIGHT": "↱  Motors: LEFT=FWD RIGHT=BACK",
        "PICK":       "🦾 Arm: extending → grip → retract",
    }
    if command in actions:
        print(f"[robot] {actions[command]}", flush=True)


# ── Gemini API ────────────────────────────────────────────────────────────────
def request_detections(client, model_name, frame):
    _, buffer = cv2.imencode(".jpg", frame)
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"),
            PROMPT_DETECT,
        ],
        config=types.GenerateContentConfig(
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise ValueError("Empty response.")
    return parse_detections(text)


def request_navigation(client, model_name, frame):
    _, buffer = cv2.imencode(".jpg", frame)
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"),
            PROMPT_NAVIGATE,
        ],
        config=types.GenerateContentConfig(
            temperature=0.5,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = (response.text or "").strip()
    if not text:
        raise ValueError("Empty response.")
    return parse_nav(text)


# ── Detector Thread ───────────────────────────────────────────────────────────
class Detector:
    def __init__(self, client, model_name, state):
        self.client = client
        self.model_name = model_name
        self.state = state

    def submit_frame(self, frame, mode):
        threading.Thread(
            target=self._run,
            args=(frame.copy(), mode),
            daemon=True,
        ).start()

    def _run(self, frame, mode):
        try:
            if mode == "NAVIGATE":
                nav = request_navigation(self.client, self.model_name, frame)
                cmd = nav.get("rover_command", "?")
                tgts = len(nav.get("targets", []))
                obs = len(nav.get("obstacles", []))
                self.state.set_nav(
                    nav,
                    f"CMD:{cmd} | Targets:{tgts} | Obstacles:{obs} | Press Q to quit"
                )
            else:
                detections = request_detections(self.client, self.model_name, frame)
                labels = [d["label"] for d in detections]
                self.state.set_detections(
                    detections,
                    f"{len(detections)} object(s): {', '.join(labels[:4])}{'...' if len(labels) > 4 else ''} | Press M to navigate"
                )
        except json.JSONDecodeError as e:
            self.state.set_detections([], f"JSON error: {e}")
        except ValueError as e:
            self.state.set_detections([], f"Parse error: {e}")
        except Exception as e:
            self.state.set_detections([], f"API error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key in .env")

    model_name = os.getenv("GEMINI_MODEL", MODEL_NAME)
    client = genai.Client(api_key=api_key)

    print(f"[startup] Model: {model_name}", flush=True)
    print("[startup] Controls: M = toggle mode | Q = quit", flush=True)

    # Connectivity test with fallback
    try:
        test = client.models.generate_content(model=model_name, contents="ping")
        print(f"[startup] ✅ Model reachable: {test.text[:50]}", flush=True)
    except Exception as e:
        print(f"[startup] ⚠️  Robotics model failed: {e}", flush=True)
        model_name = "gemini-2.0-flash"
        print(f"[startup] Falling back to {model_name}", flush=True)

    state = DetectionState()
    detector = Detector(client, model_name, state)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam. Check macOS camera permissions.")

    print("[startup] 🎥 Webcam open. Press M to toggle DETECT/NAVIGATE mode.", flush=True)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Failed to read frame.")

            now = time.time()
            detections, nav_data, status, mode = state.snapshot()

            if state.should_request(now):
                state.mark_requested(now)
                detector.submit_frame(frame, mode)

            display_frame = frame.copy()

            if mode == "NAVIGATE":
                draw_nav(display_frame, nav_data)
            else:
                draw_detections(display_frame, detections)

            draw_status(display_frame, status, mode)
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                new_mode = state.toggle_mode()
                print(f"[mode] Switched to {new_mode}", flush=True)

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("[shutdown] Done.")


if __name__ == "__main__":
    main()