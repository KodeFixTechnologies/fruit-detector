#!/usr/bin/env python3
"""
AgroPick - Gemini Vision + Arm Test
Full display version for Raspberry Pi + Pi Camera + Monitor
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

import cv2
import numpy as np
import serial
from dotenv import load_dotenv
from google import genai
from google.genai import types
from picamera2 import Picamera2


# Config
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-robotics-er-1.5-preview"
ESP32_PORT = "/dev/ttyUSB0"
ESP32_BAUD = 115200
REQUEST_EVERY = 5.0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PICAMERA_OUTPUT_FORMAT = os.getenv("PICAMERA_OUTPUT_FORMAT", "RGB888")
CAMERA_COLOR_ORDER = os.getenv("CAMERA_COLOR_ORDER", "BGR").upper()
CAMERA_SWAP_RED_BLUE = os.getenv("CAMERA_SWAP_RED_BLUE", "false").lower() == "true"

# Arm limits
BASE_MIN, BASE_MAX = 70, 160
SHOULDER_MIN, SHOULDER_MAX = 120, 160
WRIST_FIXED = 160
GRIPPER_MIN, GRIPPER_MAX = 30, 90
ROTGRIP_MIN, ROTGRIP_MAX = 130, 160

HOME = dict(base=110, shoulder=130, wrist=160, gripper=40, rotgripper=130)

# IK params
IK_BASE_CENTER = 110
IK_SHOULDER_OFFSET = 110.0
IK_SHOULDER_MULT = 1.2
INVERT_BASE = True

SERVO_DELAY = 0.3
TWIST_ANGLE = 15
TWIST_CYCLES = 3
TWIST_DELAY = 0.25
APPROACH_H = 8.0
GRAB_H = 3.0
LIFT_H = 10.0


# Logging
log_lines: list[str] = []
log_lock = threading.Lock()


def log(msg: str, level: str = "INFO") -> None:
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}] [{level}] {msg}"
    print(entry, flush=True)
    with log_lock:
        log_lines.append(entry)
        if len(log_lines) > 20:
            log_lines.pop(0)


# Serial
class SerialManager:
    def __init__(self) -> None:
        self.ser = None
        self.connected = False

    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=1)
            time.sleep(2)
            while self.ser.in_waiting:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    log(f"ESP32: {line}")
            self.connected = True
            log("ESP32 connected", "SUCCESS")
            return True
        except Exception as exc:
            log(f"ESP32 failed: {exc} - MOCK mode", "WARN")
            self.connected = False
            return False

    def send(self, cmd: str) -> None:
        if self.connected and self.ser:
            try:
                self.ser.write(f"{cmd}\n".encode())
                time.sleep(0.02)
            except Exception as exc:
                log(f"Serial error: {exc}", "ERROR")
        else:
            log(f"[MOCK] -> {cmd}", "INFO")


serial_mgr = SerialManager()


# Arm Controller
class ArmController:
    def __init__(self) -> None:
        self.positions = dict(HOME)
        self.busy = False

    def _send(self, servo: str, angle: int) -> None:
        limits = {
            "base": (BASE_MIN, BASE_MAX),
            "shoulder": (SHOULDER_MIN, SHOULDER_MAX),
            "wrist": (WRIST_FIXED, WRIST_FIXED),
            "gripper": (GRIPPER_MIN, GRIPPER_MAX),
            "rotgripper": (ROTGRIP_MIN, ROTGRIP_MAX),
        }
        if servo in limits:
            angle = int(np.clip(angle, limits[servo][0], limits[servo][1]))
        serial_mgr.send(f"{servo}:{angle}")
        self.positions[servo] = angle
        time.sleep(0.15)

    def solve_ik(self, x: float, y: float, z: float) -> dict[str, int]:
        x = float(np.clip(x, 5, 35))
        y = float(np.clip(y, -20, 20))
        z = float(np.clip(z, 0, 25))
        angle_deg = np.degrees(np.arctan2(y, x))
        if INVERT_BASE:
            base = int(IK_BASE_CENTER - angle_deg)
        else:
            base = int(IK_BASE_CENTER + angle_deg)
        base = int(np.clip(base, BASE_MIN, BASE_MAX))
        shoulder = int(IK_SHOULDER_OFFSET + z * IK_SHOULDER_MULT)
        shoulder = int(np.clip(shoulder, SHOULDER_MIN, SHOULDER_MAX))
        return {"base": base, "shoulder": shoulder, "wrist": WRIST_FIXED}

    def move_to_xyz(self, x: float, y: float, z: float) -> None:
        angles = self.solve_ik(x, y, z)
        log(
            f"IK ({x:.1f},{y:.1f},{z:.1f}) -> base:{angles['base']} shoulder:{angles['shoulder']}",
            "ARM",
        )
        self._send("base", angles["base"])
        time.sleep(SERVO_DELAY)
        self._send("shoulder", angles["shoulder"])
        time.sleep(SERVO_DELAY)
        self._send("wrist", WRIST_FIXED)
        time.sleep(SERVO_DELAY)

    def home(self) -> None:
        log("Moving to HOME", "ARM")
        self.busy = True
        for servo, angle in HOME.items():
            self._send(servo, angle)
            time.sleep(0.3)
        self.positions = dict(HOME)
        self.busy = False
        log("HOME complete", "SUCCESS")

    def open_gripper(self) -> None:
        log("Open gripper", "ARM")
        self._send("gripper", GRIPPER_MIN)
        time.sleep(0.3)

    def close_gripper(self, diameter_cm: float = 5.0) -> None:
        angle_range = GRIPPER_MAX - GRIPPER_MIN
        angle = GRIPPER_MIN + int((1 - min(diameter_cm, 8.0) / 8.0) * angle_range)
        angle = int(np.clip(angle, GRIPPER_MIN, GRIPPER_MAX))
        log(f"Close gripper diameter {diameter_cm:.1f}cm -> {angle}", "ARM")
        self._send("gripper", angle)
        time.sleep(0.3)

    def twist(self) -> None:
        log("Twisting", "ARM")
        home_rot = HOME["rotgripper"]
        for _ in range(TWIST_CYCLES):
            self._send("rotgripper", home_rot + TWIST_ANGLE)
            time.sleep(TWIST_DELAY)
            self._send("rotgripper", home_rot - TWIST_ANGLE)
            time.sleep(TWIST_DELAY)
        self._send("rotgripper", home_rot)
        time.sleep(TWIST_DELAY)

    def pick(self, x: float, y: float, diameter_cm: float = 5.0) -> bool:
        log(f"PICK at ({x:.1f},{y:.1f}) diameter {diameter_cm:.1f}cm", "PICK")
        self.busy = True
        try:
            log("Step 1/6: Approach high", "ARM")
            self.move_to_xyz(x, y, GRAB_H + APPROACH_H)
            time.sleep(0.5)

            log("Step 2/6: Open gripper", "ARM")
            self.open_gripper()
            time.sleep(0.5)

            log("Step 3/6: Lower to target", "ARM")
            self.move_to_xyz(x, y, GRAB_H)
            time.sleep(0.5)

            log("Step 4/6: Grip", "ARM")
            self.close_gripper(diameter_cm)
            time.sleep(0.5)

            log("Step 5/6: Twist", "ARM")
            self.twist()
            time.sleep(0.5)

            log("Step 6/6: Lift and home", "ARM")
            self.move_to_xyz(x, y, GRAB_H + LIFT_H)
            time.sleep(0.5)
            self.home()
            self.open_gripper()

            log("PICK COMPLETE", "SUCCESS")
            return True
        except Exception as exc:
            log(f"Pick failed: {exc}", "ERROR")
            return False
        finally:
            self.busy = False


arm = ArmController()


# Robot API for Gemini
def execute_robot_command(fn: str, args: list[Any]) -> str:
    log(f"Execute: {fn}({args})", "ARM")

    if fn == "move":
        x, y = float(args[0]), float(args[1])
        high = bool(args[2]) if len(args) > 2 else True
        arm.move_to_xyz(x, y, APPROACH_H if high else GRAB_H)
        return f"Moved ({x:.1f},{y:.1f}) {'high' if high else 'low'}"

    if fn == "setGripperState":
        opened = bool(args[0])
        if opened:
            arm.open_gripper()
        else:
            arm.close_gripper()
        return f"Gripper {'opened' if opened else 'closed'}"

    if fn == "returnToOrigin":
        arm.home()
        return "Home"

    if fn == "twist":
        arm.twist()
        return "Twisted"

    if fn == "pick":
        x, y = float(args[0]), float(args[1])
        diameter = float(args[2]) if len(args) > 2 else 5.0
        arm.pick(x, y, diameter)
        return f"Picked ({x:.1f},{y:.1f})"

    log(f"Unknown command: {fn}", "WARN")
    return f"Unknown: {fn}"


# Gemini
DETECT_PROMPT = """
You are controlling a robot arm for agricultural harvesting.
Analyze this image and detect any visible fruits or vegetables that could be
picked.

Return ONLY a JSON object, with no explanation and no markdown:
{
  "detections": [
    {
      "label": "tomato",
      "pick_point": [y, x],
      "box_2d": [ymin, xmin, ymax, xmax],
      "estimated_diameter_cm": 5.0,
      "is_ripe": true
    }
  ],
  "scene_description": "brief one line description",
  "recommended_target": "label of best fruit or vegetable to pick, or null"
}

Rules:
- Include fruits and vegetables such as tomato, brinjal, eggplant, okra,
  cucumber, chili, capsicum, pepper, bean, carrot, onion, potato, banana,
  apple, mango, orange, and similar produce when visible.
- Use normalized coordinates from 0 to 1000.
- If nothing suitable is visible, return an empty detections list and null
  recommended_target.
- If ripeness is not relevant for a vegetable, set is_ripe to true when it
  looks harvestable.
""".strip()

PICK_PROMPT_TEMPLATE = """
You are controlling a robot arm for harvesting.

Available functions:
def move(x, y, high: bool):
    # x,y in cm (x:5-35, y:-20 to 20). high=True lifts arm, False lowers.

def setGripperState(opened: bool):
    # True=open, False=close

def twist():
    # Twist to detach produce from stem

def returnToOrigin():
    # Return arm home

Target: {label}
Position: x={x_cm:.1f}cm, y={y_cm:.1f}cm
Diameter: {diameter:.1f}cm

Return ONLY a JSON array of function calls, no explanation:
[{{"function": "move", "args": [x, y, true]}}]
""".strip()


def parse_json_safe(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    candidates: list[str] = [cleaned]
    object_match = re.search(r"\{[\s\S]*\}", cleaned)
    array_match = re.search(r"\[[\s\S]*\]", cleaned)
    if object_match:
        candidates.append(object_match.group(0))
    if array_match:
        candidates.append(array_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"No JSON in: {cleaned[:120]}")


def normalize_detection_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        detections = result.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        recommended_target = result.get("recommended_target")
        if recommended_target is not None and not isinstance(recommended_target, str):
            recommended_target = None
        scene_description = result.get("scene_description", "")
        if not isinstance(scene_description, str):
            scene_description = ""
        return {
            "detections": detections,
            "scene_description": scene_description,
            "recommended_target": recommended_target,
        }

    if isinstance(result, list):
        detections = [item for item in result if isinstance(item, dict)]
        target = detections[0].get("label") if detections else None
        return {
            "detections": detections,
            "scene_description": "",
            "recommended_target": target if isinstance(target, str) else None,
        }

    raise ValueError(f"Unexpected Gemini payload type: {type(result).__name__}")


def clamp_norm(value: Any) -> int:
    try:
        return int(np.clip(int(value), 0, 1000))
    except Exception:
        return 0


class GeminiController:
    def __init__(self) -> None:
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = MODEL_NAME
        self.picks_attempted = 0
        self.picks_successful = 0
        self.last_detections: list[dict[str, Any]] = []
        self.scene_desc = "No scan yet"

    def _call(self, image_bytes: bytes, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.3,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
            ),
        )
        return (response.text or "").strip()

    def _norm_to_cm(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        x_cm = 5 + (x_norm / 1000.0) * 25
        y_cm = ((y_norm - 500) / 500.0) * 15
        return x_cm, y_cm

    def run_pipeline(self, image_bytes: bytes) -> None:
        if arm.busy:
            log("Arm busy, skip", "WARN")
            return

        log("Detecting fruits and vegetables...", "GEMINI")
        try:
            raw_result = parse_json_safe(self._call(image_bytes, DETECT_PROMPT))
            result = normalize_detection_result(raw_result)
        except Exception as exc:
            log(f"Detection error: {exc}", "ERROR")
            return

        detections = result["detections"]
        self.scene_desc = result["scene_description"]
        target = result["recommended_target"]
        self.last_detections = detections

        log(f"Scene: {self.scene_desc or 'No description'}", "GEMINI")
        log(f"Found {len(detections)} items | Target: {target}", "GEMINI")

        for detection in detections:
            label = str(detection.get("label", "?"))
            is_ripe = detection.get("is_ripe", True)
            diameter = detection.get("estimated_diameter_cm", "?")
            log(f"{label} | ripe:{is_ripe} | diameter:{diameter}cm", "INFO")

        if not detections or not target:
            log("Nothing to pick", "INFO")
            return

        best = next(
            (
                item
                for item in detections
                if item.get("label") == target and item.get("is_ripe", True)
            ),
            detections[0],
        )

        label = str(best.get("label", "target"))
        point = best.get("pick_point", [500, 500])
        if not isinstance(point, list) or len(point) != 2:
            point = [500, 500]
        diameter = float(best.get("estimated_diameter_cm", 5.0))
        x_norm = clamp_norm(point[1])
        y_norm = clamp_norm(point[0])
        x_cm, y_cm = self._norm_to_cm(x_norm, y_norm)

        log(
            f"Target: {label} norm=({x_norm},{y_norm}) cm=({x_cm:.1f},{y_cm:.1f}) diameter:{diameter:.1f}cm",
            "PICK",
        )
        self.picks_attempted += 1

        try:
            prompt = PICK_PROMPT_TEMPLATE.format(
                label=label,
                x_cm=x_cm,
                y_cm=y_cm,
                diameter=diameter,
            )
            calls = parse_json_safe(self._call(image_bytes, prompt))
            if not isinstance(calls, list):
                raise ValueError("Planning response was not a list")
            log(f"Gemini planned {len(calls)} steps", "GEMINI")
        except Exception as exc:
            log(f"Planning failed: {exc} - direct pick", "WARN")

            def direct_pick() -> None:
                if arm.pick(x_cm, y_cm, diameter):
                    self.picks_successful += 1

            threading.Thread(target=direct_pick, daemon=True).start()
            return

        def execute_all() -> None:
            for index, call in enumerate(calls, start=1):
                function_name = call.get("function", "")
                args = call.get("args", [])
                result_text = execute_robot_command(function_name, args)
                log(f"[{index}/{len(calls)}] {function_name} -> {result_text}", "ARM")
                time.sleep(0.3)
            self.picks_successful += 1
            log(
                f"Done! {self.picks_successful}/{self.picks_attempted} picks",
                "SUCCESS",
            )

        threading.Thread(target=execute_all, daemon=True).start()


gemini = GeminiController()


# Pi Camera
class PiCamera:
    def __init__(self) -> None:
        self.cam = None
        self._pixel_format = PICAMERA_OUTPUT_FORMAT
        self._color_order = CAMERA_COLOR_ORDER
        self._swap_red_blue = CAMERA_SWAP_RED_BLUE

    def start(self) -> bool:
        try:
            log("Starting Pi Camera...", "INFO")
            self.cam = Picamera2()
            config = self.cam.create_preview_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": self._pixel_format}
            )
            self.cam.configure(config)
            self.cam.start()
            time.sleep(2)
            log(
                f"Pi Camera ready ({self._pixel_format}, channel order {self._color_order}, swap_rb={self._swap_red_blue})",
                "SUCCESS",
            )
            return True
        except Exception as exc:
            log(f"Camera failed: {exc}", "ERROR")
            return False

    def toggle_color_swap(self) -> None:
        self._swap_red_blue = not self._swap_red_blue
        log(f"Camera red/blue swap: {'ON' if self._swap_red_blue else 'OFF'}", "INFO")

    def capture_frame(self) -> Any:
        frame = self.cam.capture_array()
        if frame.ndim != 3:
            return frame

        if frame.shape[2] == 4:
            if self._color_order == "RGB":
                return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if frame.shape[2] == 3 and self._color_order == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self._swap_red_blue:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def frame_to_jpeg(self, frame: Any) -> bytes:
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Failed to encode JPEG frame.")
        return buffer.tobytes()

    def stop(self) -> None:
        if self.cam:
            self.cam.stop()


camera = PiCamera()


# Draw UI
def draw_ui(frame: Any, auto_mode: bool) -> Any:
    out = frame.copy()
    height, width = out.shape[:2]

    for detection in gemini.last_detections:
        box = detection.get("box_2d")
        label = str(detection.get("label", "?"))
        ripe = bool(detection.get("is_ripe", True))
        diameter = float(detection.get("estimated_diameter_cm", 0) or 0)

        if isinstance(box, list) and len(box) == 4:
            ymin, xmin, ymax, xmax = [clamp_norm(value) for value in box]
            x1 = int(min(xmin, xmax) / 1000 * width)
            y1 = int(min(ymin, ymax) / 1000 * height)
            x2 = int(max(xmin, xmax) / 1000 * width)
            y2 = int(max(ymin, ymax) / 1000 * height)
            color = (0, 255, 0) if ripe else (0, 165, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {'RIPE' if ripe else 'UNRIPE'} {diameter:.1f}cm"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            top = max(0, y1 - text_height - 10)
            cv2.rectangle(out, (x1, top), (x1 + text_width + 6, y1), color, cv2.FILLED)
            cv2.putText(
                out,
                text,
                (x1 + 3, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )

        point = detection.get("pick_point")
        if isinstance(point, list) and len(point) == 2:
            px = int(clamp_norm(point[1]) / 1000 * width)
            py = int(clamp_norm(point[0]) / 1000 * height)
            cv2.drawMarker(out, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)

    cv2.rectangle(out, (0, 0), (width, 40), (20, 20, 20), cv2.FILLED)

    esp_color = (0, 255, 0) if serial_mgr.connected else (0, 0, 255)
    arm_color = (0, 0, 255) if arm.busy else (0, 255, 0)
    auto_color = (0, 255, 255) if auto_mode else (100, 100, 100)

    cv2.putText(out, "ESP32", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, esp_color, 2)
    cv2.putText(
        out,
        f"ARM:{'BUSY' if arm.busy else 'READY'}",
        (90, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        arm_color,
        2,
    )
    cv2.putText(
        out,
        f"AUTO:{'ON' if auto_mode else 'OFF'}",
        (230, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        auto_color,
        2,
    )
    cv2.putText(
        out,
        f"Picks:{gemini.picks_successful}/{gemini.picks_attempted}",
        (360, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        out,
        gemini.model.replace("gemini-", "").replace("-preview", ""),
        (520, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
    )

    if gemini.scene_desc:
        cv2.rectangle(out, (0, 40), (width, 62), (10, 10, 40), cv2.FILLED)
        cv2.putText(
            out,
            gemini.scene_desc[:80],
            (10, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 255),
            1,
        )

    panel_width = 280
    overlay = out.copy()
    cv2.rectangle(overlay, (width - panel_width, 65), (width, height), (10, 10, 10), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

    cv2.putText(
        out,
        "LOG",
        (width - panel_width + 8, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 200, 200),
        1,
    )

    with log_lock:
        lines = list(log_lines[-15:])

    for index, line in enumerate(lines):
        y_pos = 98 + index * 17
        if y_pos > height - 60:
            break

        color = (200, 200, 200)
        if "[SUCCESS]" in line:
            color = (0, 255, 100)
        elif "[ERROR]" in line:
            color = (0, 80, 255)
        elif "[WARN]" in line:
            color = (0, 165, 255)
        elif "[ARM]" in line:
            color = (255, 200, 0)
        elif "[GEMINI]" in line:
            color = (200, 100, 255)
        elif "[PICK]" in line:
            color = (0, 255, 200)

        cv2.putText(
            out,
            line[:38],
            (width - panel_width + 5, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
        )

    cv2.rectangle(out, (0, height - 30), (width - panel_width, height), (20, 20, 20), cv2.FILLED)
    controls = "SPACE=detect  A=auto  H=home  O=open  C=close  T=twist  V=color  Q=quit"
    cv2.putText(
        out,
        controls,
        (8, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (150, 150, 150),
        1,
    )

    return out


# Main
def main() -> None:
    print(
        """
+------------------------------------------------------+
| AgroPick - Gemini Vision + Arm (Pi + Display)        |
+------------------------------------------------------+
| SPACE = detect once and pick                         |
| A     = toggle auto mode (every 5 sec)              |
| H     = home arm                                     |
| O     = open gripper                                 |
| C     = close gripper                                |
| T     = twist                                        |
| Q     = quit                                         |
+------------------------------------------------------+
"""
    )

    if not GOOGLE_API_KEY:
        log("No API key in .env", "ERROR")
        return

    serial_mgr.connect()

    log("Testing Gemini...", "GEMINI")
    try:
        test = gemini.client.models.generate_content(model=MODEL_NAME, contents="ping")
        log(f"Gemini OK: {(test.text or '')[:40]}", "SUCCESS")
    except Exception as exc:
        log(f"Robotics model failed: {exc}", "WARN")
        gemini.model = "gemini-2.0-flash"
        log(f"Fallback: {gemini.model}", "WARN")

    if not camera.start():
        log("Camera failed", "ERROR")
        return

    arm.home()

    auto_mode = False
    last_auto = 0.0

    log("Ready! SPACE=detect  A=auto  Q=quit", "SUCCESS")

    try:
        while True:
            now = time.time()
            frame = camera.capture_frame()

            if auto_mode and not arm.busy and (now - last_auto) >= REQUEST_EVERY:
                last_auto = now
                image_bytes = camera.frame_to_jpeg(frame)
                threading.Thread(
                    target=gemini.run_pipeline,
                    args=(image_bytes,),
                    daemon=True,
                ).start()

            display = draw_ui(frame, auto_mode)
            cv2.imshow("AgroPick - Gemini Vision", display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key == ord(" "):
                if arm.busy:
                    log("Arm busy", "WARN")
                else:
                    log("Manual detect", "INFO")
                    image_bytes = camera.frame_to_jpeg(frame)
                    threading.Thread(
                        target=gemini.run_pipeline,
                        args=(image_bytes,),
                        daemon=True,
                    ).start()

            elif key in (ord("a"), ord("A")):
                auto_mode = not auto_mode
                log(f"Auto mode: {'ON' if auto_mode else 'OFF'}", "INFO")
                if auto_mode:
                    last_auto = 0

            elif key in (ord("h"), ord("H")) and not arm.busy:
                threading.Thread(target=arm.home, daemon=True).start()

            elif key in (ord("o"), ord("O")) and not arm.busy:
                arm.open_gripper()

            elif key in (ord("c"), ord("C")) and not arm.busy:
                arm.close_gripper()

            elif key in (ord("t"), ord("T")) and not arm.busy:
                threading.Thread(target=arm.twist, daemon=True).start()

            elif key in (ord("v"), ord("V")):
                camera.toggle_color_swap()

    except KeyboardInterrupt:
        log("Ctrl+C", "INFO")
    finally:
        log("Shutting down...", "INFO")
        cv2.destroyAllWindows()
        camera.stop()
        arm.home()
        log("Goodbye", "SUCCESS")


if __name__ == "__main__":
    main()
