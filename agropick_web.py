#!/usr/bin/env python3
"""
AgroPick - Gemini Vision + Rover + Arm Web App
Raspberry Pi + Pi Camera + Unified ESP32 + browser control
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Optional

import cv2
import numpy as np
import serial
import serial.tools.list_ports
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request
from google import genai
from google.genai import types
from picamera2 import Picamera2

try:
    from libcamera import controls
except ImportError:
    controls = None


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
CONFIG_FILE = "agropick_web_config.json"


@dataclass
class Config:
    camera_width: int = 640
    camera_height: int = 480
    stream_quality: int = 75
    detect_interval_seconds: float = 2.5
    control_loop_delay_seconds: float = 0.05
    pi_camera_output_format: str = "RGB888"
    camera_color_order: str = "BGR"
    camera_swap_red_blue: bool = False

    model_name: str = "gemini-robotics-er-1.5-preview"
    fallback_model_name: str = "gemini-2.0-flash"

    esp32_port: str = "/dev/ttyUSB0"
    esp32_baudrate: int = 115200

    rover_speed: int = 150
    rover_turn_speed: int = 130
    auto_forward_speed: int = 120
    auto_turn_speed: int = 110
    target_center_tolerance_norm: int = 130
    pick_box_width_threshold_norm: int = 150
    min_detection_width_norm: int = 40
    search_turn_direction: str = "left"
    side_camera_mode: bool = True

    base_min: int = 70
    base_max: int = 160
    shoulder_min: int = 120
    shoulder_max: int = 160
    wrist_fixed: int = 160
    gripper_min: int = 30
    gripper_max: int = 90
    rotgripper_min: int = 130
    rotgripper_max: int = 160

    home_base: int = 110
    home_shoulder: int = 130
    home_wrist: int = 160
    home_gripper: int = 40
    home_rotgripper: int = 130

    ik_base_center: int = 110
    ik_shoulder_offset: float = 110.0
    ik_shoulder_multiplier: float = 1.2
    invert_base: bool = False

    servo_delay: float = 0.3
    twist_angle: int = 15
    twist_cycles: int = 3
    twist_delay: float = 0.25
    approach_height_cm: float = 8.0
    grab_height_cm: float = 3.0
    lift_height_cm: float = 10.0
    gripper_max_capacity_cm: float = 8.0
    use_firmware_home_command: bool = True
    home_command_timeout_seconds: float = 8.0

    web_port: int = 5000

    def save(self, filename: str = CONFIG_FILE) -> None:
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def load(cls, filename: str = CONFIG_FILE) -> "Config":
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as handle:
                    return cls(**json.load(handle))
            except Exception:
                pass
        return cls()


class SystemState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.mode = "manual"
        self.system_running = False
        self.esp32_connected = False
        self.rover_moving = False
        self.rover_direction = "stopped"
        self.arm_busy = False
        self.arm_positions: dict[str, int] = {}
        self.detections: list[dict[str, Any]] = []
        self.scene_description = "No scan yet"
        self.recommended_target: Optional[str] = None
        self.last_frame: Optional[bytes] = None
        self.picks_attempted = 0
        self.picks_successful = 0
        self.ripe_count = 0
        self.unripe_count = 0
        self.logs: list[str] = []
        self.max_logs = 120
        self.detection_inflight = False
        self.last_detection_started_at = 0.0
        self.last_detection_completed_at = 0.0
        self.last_auto_action_at = 0.0


config = Config.load()
state = SystemState()


def log(message: str, level: str = "INFO") -> None:
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{level}] {message}"
    print(entry, flush=True)
    with state.lock:
        state.logs.append(entry)
        if len(state.logs) > state.max_logs:
            state.logs.pop(0)


class SerialManager:
    def __init__(self) -> None:
        self.serial: Optional[serial.Serial] = None
        self._write_lock = threading.Lock()
        self._line_condition = threading.Condition()
        self._recent_lines: deque[tuple[float, str]] = deque(maxlen=200)
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_running = False

    def find_esp32(self) -> Optional[str]:
        for port in serial.tools.list_ports.comports():
            description = port.description.lower()
            if any(token in description for token in ["cp210", "ch340", "usb", "uart"]):
                return port.device
            if "/dev/ttyUSB" in port.device or "/dev/ttyACM" in port.device:
                return port.device
        return None

    def connect(self) -> bool:
        port = config.esp32_port
        if not os.path.exists(port):
            detected = self.find_esp32()
            if detected:
                port = detected

        if not port:
            log("ESP32 not found", "ERROR")
            return False

        try:
            log(f"Connecting to ESP32 on {port}", "INFO")
            self.serial = serial.Serial(port, config.esp32_baudrate, timeout=0.2)
            time.sleep(2)
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            self._start_reader()

            with state.lock:
                state.esp32_connected = True
            log("ESP32 connected", "SUCCESS")
            return True
        except Exception as exc:
            with state.lock:
                state.esp32_connected = False
            log(f"ESP32 connection failed: {exc}", "WARN")
            return False

    def _start_reader(self) -> None:
        if self._reader_thread and self._reader_thread.is_alive():
            return
        self._reader_running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        while self._reader_running:
            if self.serial is None:
                time.sleep(0.1)
                continue
            try:
                raw = self.serial.readline()
            except Exception as exc:
                with state.lock:
                    state.esp32_connected = False
                log(f"Serial read failed: {exc}", "ERROR")
                return
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            with self._line_condition:
                self._recent_lines.append((time.time(), line))
                self._line_condition.notify_all()
            log(f"ESP32: {line}", "INFO")

    def send(self, command: str) -> bool:
        if not self.serial:
            return False
        try:
            with self._write_lock:
                self.serial.write(f"{command}\n".encode())
                self.serial.flush()
            time.sleep(0.02)
            return True
        except Exception as exc:
            with state.lock:
                state.esp32_connected = False
            log(f"Serial send failed: {exc}", "ERROR")
            return False

    def wait_for_line_contains(self, needle: str, timeout: float, since: float) -> bool:
        deadline = time.time() + timeout
        target = needle.lower()
        with self._line_condition:
            while True:
                if any(
                    timestamp >= since and target in line.lower()
                    for timestamp, line in self._recent_lines
                ):
                    return True
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                self._line_condition.wait(timeout=remaining)


serial_mgr = SerialManager()


class RoverController:
    def __init__(self) -> None:
        self._last_command = ""
        self._last_command_at = 0.0

    def _send_motion(self, command: str, direction: str, speed: int = 0) -> None:
        now = time.time()
        encoded = f"{command}:{speed}"
        if encoded == self._last_command and now - self._last_command_at < 0.25:
            return

        payload = command if speed <= 0 else f"{command}{speed}"
        if serial_mgr.send(payload):
            self._last_command = encoded
            self._last_command_at = now
            with state.lock:
                state.rover_moving = direction != "stopped"
                state.rover_direction = direction
            log(f"Rover -> {direction} ({speed})", "ROVER")

    def forward(self, speed: Optional[int] = None) -> None:
        self._send_motion("f", "forward", speed or config.rover_speed)

    def backward(self, speed: Optional[int] = None) -> None:
        self._send_motion("b", "backward", speed or config.rover_speed)

    def left(self, speed: Optional[int] = None) -> None:
        self._send_motion("l", "left", speed or config.rover_turn_speed)

    def right(self, speed: Optional[int] = None) -> None:
        self._send_motion("r", "right", speed or config.rover_turn_speed)

    def stop(self) -> None:
        self._send_motion("s", "stopped", 0)


rover = RoverController()


class ArmController:
    def __init__(self) -> None:
        self._positions = {
            "base": config.home_base,
            "shoulder": config.home_shoulder,
            "wrist": config.home_wrist,
            "gripper": config.home_gripper,
            "rotgripper": config.home_rotgripper,
        }
        with state.lock:
            state.arm_positions = self._positions.copy()

    def _set_busy(self, busy: bool) -> None:
        with state.lock:
            state.arm_busy = busy

    def _sync_home_state(self) -> None:
        self._positions = {
            "base": config.home_base,
            "shoulder": config.home_shoulder,
            "wrist": config.home_wrist,
            "gripper": config.home_gripper,
            "rotgripper": config.home_rotgripper,
        }
        with state.lock:
            state.arm_positions = self._positions.copy()

    def _send(self, servo: str, angle: int) -> bool:
        limits = {
            "base": (config.base_min, config.base_max),
            "shoulder": (config.shoulder_min, config.shoulder_max),
            "wrist": (config.wrist_fixed, config.wrist_fixed),
            "gripper": (config.gripper_min, config.gripper_max),
            "rotgripper": (config.rotgripper_min, config.rotgripper_max),
        }
        if servo in limits:
            angle = int(np.clip(angle, limits[servo][0], limits[servo][1]))

        success = serial_mgr.send(f"{servo}:{angle}")
        if success:
            self._positions[servo] = angle
            with state.lock:
                state.arm_positions = self._positions.copy()
            time.sleep(0.15)
        return success

    def solve_ik(self, x: float, y: float, z: float) -> dict[str, int]:
        x = float(np.clip(x, 5, 35))
        y = float(np.clip(y, -20, 20))
        z = float(np.clip(z, 0, 25))
        angle_deg = np.degrees(np.arctan2(y, x))
        if config.invert_base:
            base = int(config.ik_base_center - angle_deg)
        else:
            base = int(config.ik_base_center + angle_deg)
        base = int(np.clip(base, config.base_min, config.base_max))
        shoulder = int(config.ik_shoulder_offset + z * config.ik_shoulder_multiplier)
        shoulder = int(np.clip(shoulder, config.shoulder_min, config.shoulder_max))
        return {"base": base, "shoulder": shoulder, "wrist": config.wrist_fixed}

    def move_to_xyz(self, x: float, y: float, z: float) -> None:
        angles = self.solve_ik(x, y, z)
        log(
            f"IK ({x:.1f}, {y:.1f}, {z:.1f}) -> base:{angles['base']} shoulder:{angles['shoulder']}",
            "ARM",
        )
        self._send("base", angles["base"])
        time.sleep(config.servo_delay)
        self._send("shoulder", angles["shoulder"])
        time.sleep(config.servo_delay)
        self._send("wrist", config.wrist_fixed)
        time.sleep(config.servo_delay)

    def home(self, preserve_busy: bool = False) -> None:
        if not preserve_busy:
            self._set_busy(True)
        try:
            log("Arm -> HOME", "ARM")
            if config.use_firmware_home_command:
                started_at = time.time()
                if serial_mgr.send("home"):
                    if serial_mgr.wait_for_line_contains(
                        "HOME complete",
                        timeout=config.home_command_timeout_seconds,
                        since=started_at,
                    ):
                        self._sync_home_state()
                        log("Arm home complete", "SUCCESS")
                        return
                    log("Firmware home timed out, falling back to direct servo commands", "WARN")
            self._send("gripper", config.home_gripper)
            time.sleep(0.3)
            self._send("wrist", config.home_wrist)
            time.sleep(0.3)
            self._send("shoulder", config.home_shoulder)
            time.sleep(0.3)
            self._send("base", config.home_base)
            time.sleep(0.3)
            self._send("rotgripper", config.home_rotgripper)
            time.sleep(0.3)
            self._sync_home_state()
            log("Arm home complete", "SUCCESS")
        finally:
            if not preserve_busy:
                self._set_busy(False)

    def open_gripper(self) -> None:
        self._send("gripper", config.gripper_min)
        time.sleep(0.3)

    def close_gripper(self, diameter_cm: float = 5.0) -> None:
        angle_range = config.gripper_max - config.gripper_min
        angle = config.gripper_min + int(
            (1 - min(diameter_cm, config.gripper_max_capacity_cm) / config.gripper_max_capacity_cm)
            * angle_range
        )
        angle = int(np.clip(angle, config.gripper_min, config.gripper_max))
        self._send("gripper", angle)
        time.sleep(0.3)

    def twist(self) -> None:
        home_rot = config.home_rotgripper
        for _ in range(config.twist_cycles):
            self._send("rotgripper", home_rot + config.twist_angle)
            time.sleep(config.twist_delay)
            self._send("rotgripper", home_rot - config.twist_angle)
            time.sleep(config.twist_delay)
        self._send("rotgripper", home_rot)
        time.sleep(config.twist_delay)

    def pick(self, x: float, y: float, diameter_cm: float = 5.0) -> bool:
        if diameter_cm > config.gripper_max_capacity_cm:
            log(f"Target too large to grip ({diameter_cm:.1f}cm)", "WARN")
            return False

        self._set_busy(True)
        with state.lock:
            state.picks_attempted += 1

        try:
            log(f"Picking target at ({x:.1f}, {y:.1f}) diameter {diameter_cm:.1f}cm", "PICK")
            self.move_to_xyz(x, y, config.grab_height_cm + config.approach_height_cm)
            time.sleep(0.5)
            self.open_gripper()
            time.sleep(0.5)
            self.move_to_xyz(x, y, config.grab_height_cm)
            time.sleep(0.5)
            self.close_gripper(diameter_cm)
            time.sleep(0.5)
            self.twist()
            time.sleep(0.5)
            self.move_to_xyz(x, y, config.grab_height_cm + config.lift_height_cm)
            time.sleep(0.5)
            self.home(preserve_busy=True)
            self.open_gripper()
            with state.lock:
                state.picks_successful += 1
            log("Pick complete", "SUCCESS")
            return True
        except Exception as exc:
            log(f"Pick failed: {exc}", "ERROR")
            return False
        finally:
            self._set_busy(False)


arm = ArmController()


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

    raise ValueError(f"No JSON in Gemini response: {cleaned[:120]}")


def clamp_norm(value: Any) -> int:
    try:
        return int(np.clip(int(value), 0, 1000))
    except Exception:
        return 0


def get_detection_point(det: dict[str, Any]) -> tuple[int, int]:
    point = det.get("pick_point")
    if isinstance(point, list) and len(point) == 2:
        return clamp_norm(point[0]), clamp_norm(point[1])
    box = det.get("box_2d")
    if isinstance(box, list) and len(box) == 4:
        ymin, xmin, ymax, xmax = [clamp_norm(value) for value in box]
        return (ymin + ymax) // 2, (xmin + xmax) // 2
    return 500, 500


def get_detection_width_norm(det: dict[str, Any]) -> int:
    box = det.get("box_2d")
    if not isinstance(box, list) or len(box) != 4:
        return 0
    xmin = clamp_norm(box[1])
    xmax = clamp_norm(box[3])
    return abs(xmax - xmin)


def overlay_detections(frame: Any, detections: list[dict[str, Any]], scene_description: str, mode: str) -> Any:
    display = frame.copy()
    height, width = display.shape[:2]

    for detection in detections:
        label = str(detection.get("label", "?"))
        ripe = bool(detection.get("is_ripe", True))
        diameter = float(detection.get("estimated_diameter_cm", 0) or 0)
        box = detection.get("box_2d")

        if isinstance(box, list) and len(box) == 4:
            ymin, xmin, ymax, xmax = [clamp_norm(value) for value in box]
            x1 = int(min(xmin, xmax) / 1000 * width)
            y1 = int(min(ymin, ymax) / 1000 * height)
            x2 = int(max(xmin, xmax) / 1000 * width)
            y2 = int(max(ymin, ymax) / 1000 * height)
            color = (0, 255, 0) if ripe else (0, 165, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            title = f"{label} {diameter:.1f}cm"
            cv2.putText(
                display,
                title[:32],
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        point_y, point_x = get_detection_point(detection)
        px = int(point_x / 1000 * width)
        py = int(point_y / 1000 * height)
        cv2.drawMarker(display, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)

    cv2.rectangle(display, (0, 0), (width, 62), (18, 18, 18), cv2.FILLED)
    with state.lock:
        top_text = (
            f"{mode.upper()} | Rover:{state.rover_direction.upper()} | "
            f"Arm:{'BUSY' if state.arm_busy else 'READY'} | "
            f"Picks:{state.picks_successful}/{state.picks_attempted}"
        )
    cv2.putText(display, top_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    cv2.putText(display, scene_description[:90], (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
    return display


DETECT_PROMPT = """
You are controlling a rover and robot arm for agricultural harvesting.
Analyze this camera image and detect visible fruits or vegetables in the scene.

Return ONLY this JSON object:
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
  "recommended_target": "label of the best fruit or vegetable to pick, or null"
}

Rules:
- Include fruits and vegetables only.
- box_2d and pick_point coordinates are normalized 0 to 1000.
- If the item is harvestable, set is_ripe to true even for vegetables.
- If nothing useful is visible, return an empty detections list and null recommended_target.
""".strip()


class GeminiVision:
    def __init__(self) -> None:
        if not GOOGLE_API_KEY:
            raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY")
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model = config.model_name
        self._pick_lock = threading.Lock()

    def test_connection(self) -> None:
        try:
            response = self.client.models.generate_content(model=self.model, contents="ping")
            log(f"Gemini OK: {(response.text or '')[:40]}", "SUCCESS")
        except Exception as exc:
            log(f"Model {self.model} unavailable: {exc}", "WARN")
            self.model = config.fallback_model_name
            log(f"Falling back to {self.model}", "WARN")

    def detect_scene(self, image_bytes: bytes) -> None:
        with state.lock:
            if state.detection_inflight or state.arm_busy:
                return
            state.detection_inflight = True
            state.last_detection_started_at = time.time()

        try:
            log("Gemini detecting scene...", "GEMINI")
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    DETECT_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )

            parsed = parse_json_safe((response.text or "").strip())
            if not isinstance(parsed, dict):
                raise ValueError("Gemini detection payload was not a JSON object")

            detections = parsed.get("detections", [])
            if not isinstance(detections, list):
                detections = []

            detections = [item for item in detections if isinstance(item, dict)]
            scene_description = parsed.get("scene_description", "")
            if not isinstance(scene_description, str):
                scene_description = ""
            recommended_target = parsed.get("recommended_target")
            if recommended_target is not None and not isinstance(recommended_target, str):
                recommended_target = None

            with state.lock:
                state.detections = detections
                state.scene_description = scene_description
                state.recommended_target = recommended_target
                state.ripe_count = sum(1 for item in detections if item.get("is_ripe", True))
                state.unripe_count = max(0, len(detections) - state.ripe_count)
                state.last_detection_completed_at = time.time()

            log(
                f"Detected {len(detections)} item(s). Target: {recommended_target}",
                "GEMINI",
            )
        except Exception as exc:
            log(f"Gemini detection failed: {exc}", "ERROR")
        finally:
            with state.lock:
                state.detection_inflight = False

    def pick_detection(self, detection: dict[str, Any]) -> bool:
        if not self._pick_lock.acquire(blocking=False):
            return False

        def worker() -> None:
            try:
                point_y, point_x = get_detection_point(detection)
                x_cm = 5 + (point_x / 1000.0) * 25
                y_cm = ((point_y - 500) / 500.0) * 15
                diameter = float(detection.get("estimated_diameter_cm", 5.0))
                log(
                    f"Pick target {detection.get('label', '?')} at norm=({point_x},{point_y}) cm=({x_cm:.1f},{y_cm:.1f})",
                    "PICK",
                )
                success = arm.pick(x_cm, y_cm, diameter)
                if success:
                    with state.lock:
                        state.detections = []
                        state.recommended_target = None
                        state.scene_description = "Last pick completed"
            finally:
                self._pick_lock.release()

        threading.Thread(target=worker, daemon=True).start()
        return True


gemini: Optional[GeminiVision] = None


class PiCamera:
    def __init__(self) -> None:
        self.camera: Optional[Picamera2] = None
        self._color_order = config.camera_color_order.upper()
        self._swap_rb = config.camera_swap_red_blue

    def start(self) -> bool:
        try:
            self.camera = Picamera2()
            cam_config = self.camera.create_preview_configuration(
                main={
                    "size": (config.camera_width, config.camera_height),
                    "format": config.pi_camera_output_format,
                }
            )
            self.camera.configure(cam_config)
            self.camera.start()
            time.sleep(1)
            if controls is not None:
                try:
                    self.camera.set_controls(
                        {
                            "AfMode": controls.AfModeEnum.Continuous,
                            "AfSpeed": controls.AfSpeedEnum.Fast,
                        }
                    )
                except Exception:
                    pass
            log(
                f"Camera ready ({config.pi_camera_output_format}, order={self._color_order}, swap_rb={self._swap_rb})",
                "SUCCESS",
            )
            return True
        except Exception as exc:
            log(f"Camera failed: {exc}", "ERROR")
            return False

    def capture_frame(self) -> Any:
        frame = self.camera.capture_array()
        if frame.ndim == 3 and frame.shape[2] == 4:
            if self._color_order == "RGB":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3 and self._color_order == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self._swap_rb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def frame_to_jpeg(self, frame: Any) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, config.stream_quality],
        )
        if not ok:
            raise RuntimeError("Failed to encode JPEG frame")
        return encoded.tobytes()

    def stop(self) -> None:
        if self.camera:
            self.camera.stop()


camera = PiCamera()


def get_best_detection() -> Optional[dict[str, Any]]:
    with state.lock:
        detections = list(state.detections)
        recommended_target = state.recommended_target

    if not detections:
        return None

    ripe = [item for item in detections if item.get("is_ripe", True)]
    candidates = ripe or detections

    if recommended_target:
        for item in candidates:
            if item.get("label") == recommended_target:
                return item

    candidates = [
        item
        for item in candidates
        if get_detection_width_norm(item) >= config.min_detection_width_norm
    ] or candidates
    candidates.sort(key=lambda item: get_detection_width_norm(item), reverse=True)
    return candidates[0]


def start_detection_for_frame(frame: Any) -> None:
    if gemini is None:
        return
    image_bytes = camera.frame_to_jpeg(frame)
    threading.Thread(target=gemini.detect_scene, args=(image_bytes,), daemon=True).start()


def handle_autonomous_behavior() -> None:
    if not state.system_running or state.mode != "autonomous":
        return

    if state.arm_busy:
        rover.stop()
        return

    best = get_best_detection()
    now = time.time()

    if best is None:
        if now - state.last_auto_action_at > 0.5:
            rover.forward(config.auto_forward_speed)
            with state.lock:
                state.last_auto_action_at = now
        return

    width_norm = get_detection_width_norm(best)

    if width_norm < config.pick_box_width_threshold_norm:
        rover.forward(config.auto_forward_speed)
        with state.lock:
            state.last_auto_action_at = now
        return

    rover.stop()
    if gemini is not None and gemini.pick_detection(best):
        with state.lock:
            state.last_auto_action_at = now


def handle_semi_auto_behavior() -> None:
    if not state.system_running or state.mode != "semi-auto" or state.arm_busy:
        return
    if state.rover_moving:
        return

    best = get_best_detection()
    if not best:
        return

    width_norm = get_detection_width_norm(best)
    if gemini is not None and width_norm >= config.pick_box_width_threshold_norm:
        gemini.pick_detection(best)


def control_loop() -> None:
    log("Control loop started", "INFO")
    last_detection_request = 0.0

    while True:
        if camera.camera is None:
            time.sleep(0.2)
            continue

        frame = camera.capture_frame()
        with state.lock:
            detections = list(state.detections)
            scene_description = state.scene_description
            mode = state.mode

        display = overlay_detections(frame, detections, scene_description, mode)
        try:
            with state.lock:
                state.last_frame = camera.frame_to_jpeg(display)
        except Exception as exc:
            log(f"Frame encode failed: {exc}", "ERROR")

        now = time.time()
        with state.lock:
            should_detect = (
                state.system_running
                and not state.detection_inflight
                and not state.arm_busy
                and now - last_detection_request >= config.detect_interval_seconds
            )

        if should_detect:
            last_detection_request = now
            start_detection_for_frame(frame)

        handle_autonomous_behavior()
        handle_semi_auto_behavior()

        time.sleep(config.control_loop_delay_seconds)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AgroPick Web</title>
  <style>
    :root {
      --bg: #0b1324;
      --panel: #121b32;
      --line: #233253;
      --ink: #ebf1ff;
      --muted: #8aa0c8;
      --cyan: #2bd9ff;
      --green: #56f08d;
      --red: #ff607f;
      --amber: #ffbe55;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #152448 0%, var(--bg) 55%);
      color: var(--ink);
    }
    .shell {
      max-width: 1500px;
      margin: 0 auto;
      padding: 20px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
    }
    .title {
      font-size: 30px;
      font-weight: 700;
      letter-spacing: 0.04em;
    }
    .chips {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .chip {
      border: 1px solid var(--line);
      background: rgba(18, 27, 50, 0.85);
      border-radius: 999px;
      padding: 8px 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 18px;
    }
    .panel {
      background: rgba(18, 27, 50, 0.9);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 14px 45px rgba(0, 0, 0, 0.18);
    }
    .video {
      width: 100%;
      border-radius: 14px;
      display: block;
      background: #000;
      border: 1px solid #1d2a48;
    }
    .section-title {
      font-size: 14px;
      color: var(--cyan);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }
    .modes, .actions, .arm-grid {
      display: grid;
      gap: 10px;
    }
    .modes { grid-template-columns: repeat(3, 1fr); }
    .actions { grid-template-columns: repeat(2, 1fr); margin-top: 14px; }
    .arm-grid { grid-template-columns: repeat(2, 1fr); margin-top: 14px; }
    button {
      border: 1px solid var(--line);
      background: #16223f;
      color: var(--ink);
      border-radius: 12px;
      padding: 12px 14px;
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { border-color: var(--cyan); }
    .mode.active { border-color: var(--cyan); color: var(--cyan); }
    .mode.auto.active { border-color: var(--green); color: var(--green); }
    .start { background: linear-gradient(135deg, #1d6f4a, #23b269); }
    .stop { background: linear-gradient(135deg, #833049, #c24767); }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin-bottom: 14px;
    }
    .stat {
      background: #0e1730;
      border: 1px solid #1f2d4d;
      border-radius: 14px;
      padding: 14px;
    }
    .stat .value {
      font-size: 28px;
      font-weight: 700;
    }
    .stat .label {
      font-size: 12px;
      color: var(--muted);
      margin-top: 4px;
    }
    .rover-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-top: 14px;
    }
    .rover-grid button {
      min-height: 70px;
      font-size: 26px;
    }
    .empty {
      border: none;
      background: transparent;
      cursor: default;
    }
    .stop-btn { background: #5d2431; }
    .slider-row { margin-top: 16px; }
    input[type=range] { width: 100%; }
    .ik-block {
      margin-top: 18px;
      padding: 14px;
      border: 1px solid #1d2a48;
      border-radius: 14px;
      background: #0c152b;
    }
    .ik-block label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      margin-top: 10px;
    }
    .ik-block input[type=number] {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid #233253;
      background: #101a33;
      color: var(--ink);
    }
    .ik-check {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .logs {
      margin-top: 14px;
      min-height: 220px;
      max-height: 260px;
      overflow: auto;
      background: #091124;
      border-radius: 14px;
      border: 1px solid #1a2846;
      padding: 12px;
      font-family: ui-monospace, SFMono-Regular, monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    .log-line { padding: 3px 0; color: #cdd9f5; }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <div class="title">AgroPick Web</div>
      <div class="chips">
        <div class="chip" id="esp32Status">ESP32: offline</div>
        <div class="chip" id="modeStatus">Mode: manual</div>
        <div class="chip" id="roverStatus">Rover: stopped</div>
        <div class="chip" id="armStatus">Arm: ready</div>
      </div>
    </div>

    <div class="layout">
      <div class="panel">
        <div class="section-title">Live Feed</div>
        <img class="video" src="/video_feed" alt="Live video feed">

        <div class="modes" style="margin-top: 16px;">
          <button class="mode active" id="mode-manual" onclick="setMode('manual', this)">Manual</button>
          <button class="mode" id="mode-semi-auto" onclick="setMode('semi-auto', this)">Semi-Auto</button>
          <button class="mode auto" id="mode-autonomous" onclick="setMode('autonomous', this)">Autonomous</button>
        </div>

        <div class="actions">
          <button id="systemButton" class="start" onclick="toggleSystem()">Start Vision</button>
          <button onclick="api('scan', {force: true})">Scan Now</button>
        </div>

        <div class="section-title" style="margin-top: 16px;">Rover</div>
        <div class="rover-grid">
          <button class="empty"></button>
          <button onmousedown="rover('f')" onmouseup="rover('s')" ontouchstart="rover('f')" ontouchend="rover('s')">▲</button>
          <button class="empty"></button>
          <button onmousedown="rover('l')" onmouseup="rover('s')" ontouchstart="rover('l')" ontouchend="rover('s')">◀</button>
          <button class="stop-btn" onclick="rover('s')">■</button>
          <button onmousedown="rover('r')" onmouseup="rover('s')" ontouchstart="rover('r')" ontouchend="rover('s')">▶</button>
          <button class="empty"></button>
          <button onmousedown="rover('b')" onmouseup="rover('s')" ontouchstart="rover('b')" ontouchend="rover('s')">▼</button>
          <button class="empty"></button>
        </div>

        <div class="slider-row">
          <label for="speedSlider">Rover speed: <span id="speedValue">150</span></label>
          <input id="speedSlider" type="range" min="60" max="255" value="150" oninput="updateSpeed(this.value)">
        </div>
      </div>

      <div class="panel">
        <div class="section-title">Status</div>
        <div class="stats">
          <div class="stat">
            <div class="value" id="ripeCount">0</div>
            <div class="label">Harvestable</div>
          </div>
          <div class="stat">
            <div class="value" id="unripeCount">0</div>
            <div class="label">Other Detections</div>
          </div>
          <div class="stat">
            <div class="value" id="pickSuccess">0</div>
            <div class="label">Successful Picks</div>
          </div>
          <div class="stat">
            <div class="value" id="pickAttempts">0</div>
            <div class="label">Pick Attempts</div>
          </div>
        </div>

        <div class="section-title">Arm</div>
        <div class="arm-grid">
          <button onclick="api('arm/home')">Home</button>
          <button onclick="api('arm/pick')">Pick Best</button>
          <button onclick="api('arm/open')">Open</button>
          <button onclick="api('arm/close')">Close</button>
          <button onclick="api('arm/twist')">Twist</button>
          <button onclick="api('camera/color')">Swap Colors</button>
        </div>

        <div class="ik-block">
          <div class="section-title" style="margin-bottom: 8px;">IK Tuning</div>
          <label for="ikBaseCenter">Base Center</label>
          <input id="ikBaseCenter" type="number" min="70" max="160">
          <label for="ikShoulderOffset">Shoulder Offset</label>
          <input id="ikShoulderOffset" type="number" min="90" max="140" step="0.1">
          <label for="ikShoulderMultiplier">Shoulder Multiplier</label>
          <input id="ikShoulderMultiplier" type="number" min="0.5" max="2.5" step="0.1">
          <label class="ik-check">
            <input id="ikInvertBase" type="checkbox">
            Invert Base
          </label>
          <button style="margin-top: 12px;" onclick="saveIK()">Save IK Settings</button>
        </div>

        <div class="section-title" style="margin-top: 18px;">Logs</div>
        <div class="logs" id="logs"></div>
      </div>
    </div>
  </div>

  <script>
    let running = false;

    function api(endpoint, data) {
      const options = data ? {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(data)
      } : {};
      return fetch('/api/' + endpoint, options).then((response) => response.json());
    }

    function clearModes() {
      document.querySelectorAll('.mode').forEach((el) => el.classList.remove('active'));
    }

    function setMode(mode, button) {
      api('mode', {mode});
      clearModes();
      button.classList.add('active');
    }

    function toggleSystem() {
      running = !running;
      api('system', {running});
      const button = document.getElementById('systemButton');
      button.textContent = running ? 'Stop Vision' : 'Start Vision';
      button.className = running ? 'stop' : 'start';
    }

    function rover(cmd) {
      api('rover/' + cmd);
    }

    function saveIK() {
      api('ik', {
        base_center: parseInt(document.getElementById('ikBaseCenter').value, 10),
        shoulder_offset: parseFloat(document.getElementById('ikShoulderOffset').value),
        shoulder_multiplier: parseFloat(document.getElementById('ikShoulderMultiplier').value),
        invert_base: document.getElementById('ikInvertBase').checked
      });
    }

    function updateSpeed(value) {
      document.getElementById('speedValue').textContent = value;
      api('speed', {speed: parseInt(value, 10)});
    }

    function updateStatus() {
      api('status').then((data) => {
        document.getElementById('esp32Status').textContent = 'ESP32: ' + (data.esp32_connected ? 'online' : 'offline');
        document.getElementById('modeStatus').textContent = 'Mode: ' + data.mode;
        document.getElementById('roverStatus').textContent = 'Rover: ' + data.rover_direction;
        document.getElementById('armStatus').textContent = 'Arm: ' + (data.arm_busy ? 'busy' : 'ready');
        document.getElementById('ripeCount').textContent = data.ripe_count;
        document.getElementById('unripeCount').textContent = data.unripe_count;
        document.getElementById('pickSuccess').textContent = data.picks_successful;
        document.getElementById('pickAttempts').textContent = data.picks_attempted;
        document.getElementById('speedValue').textContent = data.rover_speed;
        document.getElementById('speedSlider').value = data.rover_speed;
        running = data.system_running;
        const button = document.getElementById('systemButton');
        button.textContent = running ? 'Stop Vision' : 'Start Vision';
        button.className = running ? 'stop' : 'start';
        clearModes();
        const active = document.getElementById('mode-' + data.mode);
        if (active) active.classList.add('active');
      });
    }

    function updateLogs() {
      api('logs').then((data) => {
        const el = document.getElementById('logs');
        el.innerHTML = data.logs.map((line) => `<div class="log-line">${line}</div>`).join('');
        el.scrollTop = el.scrollHeight;
      });
    }

    updateStatus();
    updateLogs();
    api('ik').then((data) => {
      document.getElementById('ikBaseCenter').value = data.base_center;
      document.getElementById('ikShoulderOffset').value = data.shoulder_offset;
      document.getElementById('ikShoulderMultiplier').value = data.shoulder_multiplier;
      document.getElementById('ikInvertBase').checked = data.invert_base;
    });
    setInterval(updateStatus, 700);
    setInterval(updateLogs, 1100);
  </script>
</body>
</html>
"""


app = Flask(__name__)


@app.route("/")
def index() -> str:
    return render_template_string(HTML_TEMPLATE)


@app.route("/video_feed")
def video_feed() -> Response:
    def generate() -> Any:
        while True:
            with state.lock:
                frame = state.last_frame
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status() -> Response:
    with state.lock:
        return jsonify(
            {
                "mode": state.mode,
                "system_running": state.system_running,
                "esp32_connected": state.esp32_connected,
                "rover_moving": state.rover_moving,
                "rover_direction": state.rover_direction,
                "rover_speed": config.rover_speed,
                "arm_busy": state.arm_busy,
                "arm_positions": state.arm_positions,
                "ripe_count": state.ripe_count,
                "unripe_count": state.unripe_count,
                "picks_attempted": state.picks_attempted,
                "picks_successful": state.picks_successful,
                "scene_description": state.scene_description,
                "recommended_target": state.recommended_target,
            }
        )


@app.route("/api/logs")
def api_logs() -> Response:
    with state.lock:
        return jsonify({"logs": state.logs[-60:]})


@app.route("/api/mode", methods=["POST"])
def api_mode() -> Response:
    data = request.get_json(force=True) or {}
    mode = data.get("mode", "manual")
    if mode not in {"manual", "semi-auto", "autonomous"}:
        return jsonify({"success": False, "error": "invalid mode"}), 400
    with state.lock:
        state.mode = mode
    log(f"Mode -> {mode}", "INFO")
    if mode == "manual":
        rover.stop()
    return jsonify({"success": True})


@app.route("/api/system", methods=["POST"])
def api_system() -> Response:
    data = request.get_json(force=True) or {}
    running = bool(data.get("running", False))
    with state.lock:
        state.system_running = running
    log(f"System {'STARTED' if running else 'STOPPED'}", "INFO")
    if not running:
        rover.stop()
    return jsonify({"success": True})


@app.route("/api/speed", methods=["POST"])
def api_speed() -> Response:
    data = request.get_json(force=True) or {}
    speed = int(data.get("speed", config.rover_speed))
    speed = int(np.clip(speed, 60, 255))
    config.rover_speed = speed
    config.rover_turn_speed = max(60, speed - 20)
    config.auto_forward_speed = max(70, speed - 20)
    config.auto_turn_speed = max(60, speed - 40)
    config.save()
    log(f"Rover speed -> {config.rover_speed}", "INFO")
    return jsonify({"success": True})


@app.route("/api/ik", methods=["GET", "POST"])
def api_ik() -> Response:
    if request.method == "GET":
        return jsonify(
            {
                "base_center": config.ik_base_center,
                "shoulder_offset": config.ik_shoulder_offset,
                "shoulder_multiplier": config.ik_shoulder_multiplier,
                "invert_base": config.invert_base,
            }
        )

    data = request.get_json(force=True) or {}
    config.ik_base_center = int(data.get("base_center", config.ik_base_center))
    config.ik_shoulder_offset = float(data.get("shoulder_offset", config.ik_shoulder_offset))
    config.ik_shoulder_multiplier = float(
        data.get("shoulder_multiplier", config.ik_shoulder_multiplier)
    )
    config.invert_base = bool(data.get("invert_base", config.invert_base))
    config.save()
    log(
        "IK updated: "
        f"base={config.ik_base_center}, "
        f"shoulder={config.ik_shoulder_offset}/{config.ik_shoulder_multiplier}, "
        f"invert={config.invert_base}",
        "INFO",
    )
    return jsonify({"success": True})


@app.route("/api/scan", methods=["POST"])
def api_scan() -> Response:
    if camera.camera is None:
        return jsonify({"success": False, "error": "camera not started"}), 400
    frame = camera.capture_frame()
    start_detection_for_frame(frame)
    return jsonify({"success": True})


@app.route("/api/rover/<cmd>")
def api_rover(cmd: str) -> Response:
    if cmd == "f":
        rover.forward()
    elif cmd == "b":
        rover.backward()
    elif cmd == "l":
        rover.left()
    elif cmd == "r":
        rover.right()
    elif cmd == "s":
        rover.stop()
    else:
        return jsonify({"success": False, "error": "invalid rover command"}), 400
    return jsonify({"success": True})


@app.route("/api/arm/home")
def api_arm_home() -> Response:
    threading.Thread(target=arm.home, daemon=True).start()
    return jsonify({"success": True})


@app.route("/api/arm/open")
def api_arm_open() -> Response:
    threading.Thread(target=arm.open_gripper, daemon=True).start()
    return jsonify({"success": True})


@app.route("/api/arm/close")
def api_arm_close() -> Response:
    threading.Thread(target=arm.close_gripper, daemon=True).start()
    return jsonify({"success": True})


@app.route("/api/arm/twist")
def api_arm_twist() -> Response:
    threading.Thread(target=arm.twist, daemon=True).start()
    return jsonify({"success": True})


@app.route("/api/arm/pick")
def api_arm_pick() -> Response:
    if gemini is None:
        return jsonify({"success": False, "error": "gemini not initialized"}), 400
    best = get_best_detection()
    if not best:
        return jsonify({"success": False, "error": "no current target"}), 400
    started = gemini.pick_detection(best)
    return jsonify({"success": started})


@app.route("/api/camera/color")
def api_camera_color() -> Response:
    camera._swap_rb = not camera._swap_rb
    config.camera_swap_red_blue = camera._swap_rb
    config.save()
    log(f"Camera red/blue swap -> {camera._swap_rb}", "INFO")
    return jsonify({"success": True, "swap_rb": camera._swap_rb})


def main() -> None:
    global gemini
    print(
        """
+------------------------------------------------------------------+
| AgroPick Web                                                     |
| Gemini detection + rover control + arm picking + browser UI      |
+------------------------------------------------------------------+
"""
    )

    if not GOOGLE_API_KEY:
        log("Missing GOOGLE_API_KEY or GEMINI_API_KEY", "ERROR")
        return

    serial_mgr.connect()

    gemini = GeminiVision()
    gemini.test_connection()

    if not camera.start():
        return

    if state.esp32_connected:
        arm.home()

    threading.Thread(target=control_loop, daemon=True).start()

    log(f"Open http://0.0.0.0:{config.web_port} from the Pi network", "SUCCESS")
    app.run(host="0.0.0.0", port=config.web_port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
