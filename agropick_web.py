#!/usr/bin/env python3
"""
AgroPick simple web controller.

Architecture:
  - Raspberry Pi owns camera, Gemini calls, web UI, and mode logic.
  - ESP32 is a simple serial actuator for rover + arm.
  - If ESP32 is missing, the app still runs in mock mode for vision/UI testing.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

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
    from libcamera import controls as libcam_controls

    HAS_AF = True
except ImportError:
    HAS_AF = False



load_dotenv()

GOOGLE_API_KEY =  os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

CONFIG_FILE = "agropick_config.json"


@dataclass
class Config:
    cam_w: int = 640
    cam_h: int = 480
    jpeg_quality: int = 70
    detect_interval: float = 3.0
    auto_pick_box_width_norm: int = 140

    esp_port: str = "/dev/ttyUSB0"
    esp_baud: int = 115200

    rover_speed: int = 60
    rover_turn_speed: int = 45
    auto_rover_speed: int = 45
    auto_stop_settle: float = 0.5
    auto_resume_delay: float = 1.0
    # Autonomous step-scan timing (all values in SECONDS, not milliseconds)
    scan_step_duration: float = 0.8   # how long the rover drives between scans (0.8 s = 800 ms)
    scan_settle_time: float = 0.3     # camera settle pause after stopping before scanning (300 ms)
    gemini_timeout: float = 10.0      # max seconds to wait for Gemini API response before giving up

    base_min: int = 70
    base_max: int = 160
    shoulder_min: int = 120
    shoulder_max: int = 160
    wrist_fixed: int = 160
    gripper_min: int = 30
    gripper_max: int = 90
    rotgrip_min: int = 130
    rotgrip_max: int = 160

    home_base: int = 110
    home_shoulder: int = 130
    home_wrist: int = 160
    home_gripper: int = 40
    home_rotgrip: int = 130

    ik_base_center: int = 110
    ik_shoulder_offset: float = 110.0
    ik_shoulder_mult: float = 1.2
    invert_base: bool = False

    servo_delay: float = 0.3
    twist_angle: int = 15
    twist_cycles: int = 3
    twist_delay: float = 0.25
    approach_h: float = 8.0
    grab_h: float = 3.0
    pick_drop_extra: float = 2.0
    grip_extra_close: int = 10
    lift_h: float = 10.0

    port: int = 5000

    def save(self) -> None:
        with open(CONFIG_FILE, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def load(cls) -> "Config":
        if not os.path.exists(CONFIG_FILE):
            return cls()
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            return cls()
        allowed = {item.name for item in fields(cls)}
        data = {key: value for key, value in raw.items() if key in allowed}
        try:
            return cls(**data)
        except Exception:
            return cls()


cfg = Config.load()

MODEL_NAME = "gemini-robotics-er-1.5-preview"
PICAMERA_OUTPUT_FORMAT = os.getenv("PICAMERA_OUTPUT_FORMAT", "RGB888")
CAMERA_COLOR_ORDER = os.getenv("CAMERA_COLOR_ORDER", "BGR").upper()
CAMERA_SWAP_RED_BLUE = os.getenv("CAMERA_SWAP_RED_BLUE", "false").lower() == "true"


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.mode = "manual"
        self.running = False
        self.esp_ok = False
        self.gemini_ok = False
        self.rover_moving = False
        self.rover_dir = "stopped"
        self.arm_busy = False
        self.arm_pos: Dict[str, int] = {}
        self.last_jpeg: Optional[bytes] = None
        self.detections: List[Dict[str, Any]] = []
        self.scene_desc = ""
        self.ripe_count = 0
        self.unripe_count = 0
        self.picks_ok = 0
        self.picks_try = 0
        self.logs: List[str] = []


S = State()


def log(msg: str, lvl: str = "INFO") -> None:
    entry = f"[{time.strftime('%H:%M:%S')}][{lvl}] {msg}"
    print(entry, flush=True)
    with S.lock:
        S.logs.append(entry)
        if len(S.logs) > 120:
            S.logs.pop(0)


class SerialManager:
    def __init__(self) -> None:
        self.ser: Optional[serial.Serial] = None
        self.connected = False
        self._lock = threading.Lock()
        self._next_command_id = 1

    def _command_id(self) -> str:
        command_id = str(self._next_command_id)
        self._next_command_id += 1
        return command_id

    def _readline(self, timeout: float = 0.25) -> str:
        if self.ser is None:
            return ""
        original_timeout = self.ser.timeout
        try:
            self.ser.timeout = timeout
            raw = self.ser.readline()
        finally:
            self.ser.timeout = original_timeout
        return raw.decode("utf-8", errors="ignore").strip() if raw else ""

    def _flush_input(self) -> None:
        if self.ser is None:
            return
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

    def _probe_protocol(self) -> bool:
        if self.ser is None:
            return False
        command_id = "probe"
        try:
            with self._lock:
                self._flush_input()
                self.ser.write(f"GET|{command_id}|VERSION\n".encode())
                self.ser.flush()
                deadline = time.time() + 3.0
                while time.time() < deadline:
                    line = self._readline(0.2)
                    if not line:
                        continue
                    if line == f"ACK|{command_id}":
                        continue
                    if line.startswith(f"VAL|{command_id}|VERSION|"):
                        return True
                    if line.startswith(f"ERR|{command_id}|"):
                        return False
        except Exception:
            return False
        return False

    def _protocol_call(self, opcode: str, *parts: Any, expect_value: Optional[str] = None, timeout: float = 8.0) -> List[str]:
        if not self.connected or self.ser is None:
            log(f"[MOCK] -> {opcode}|{'|'.join(str(part) for part in parts)}", "INFO")
            return []

        command_id = self._command_id()
        frame = "|".join([opcode, command_id] + [str(part) for part in parts if part is not None])

        with self._lock:
            self._flush_input()
            self.ser.write(f"{frame}\n".encode())
            self.ser.flush()

            deadline = time.time() + timeout
            saw_ack = False
            while time.time() < deadline:
                line = self._readline(0.15)
                if not line:
                    continue
                if line == f"ACK|{command_id}":
                    saw_ack = True
                    if expect_value is None and opcode == "PING":
                        return []
                    continue
                if line.startswith(f"ERR|{command_id}|"):
                    raise RuntimeError(line)
                if line.startswith(f"VAL|{command_id}|"):
                    tokens = line.split("|")
                    if expect_value is None or (len(tokens) >= 3 and tokens[2] == expect_value):
                        return tokens[3:]
                if line.startswith(f"DONE|{command_id}|"):
                    return line.split("|")[2:]

            if not saw_ack:
                raise TimeoutError(f"Protocol ACK timeout for {frame}")
            raise TimeoutError(f"Protocol completion timeout for {frame}")

    def _find_port(self) -> Optional[str]:
        if os.path.exists(cfg.esp_port):
            return cfg.esp_port
        for port in serial.tools.list_ports.comports():
            description = port.description.lower()
            if any(token in description for token in ("cp210", "ch340", "usb", "uart")):
                return port.device
            if "/dev/ttyUSB" in port.device or "/dev/ttyACM" in port.device:
                return port.device
        return None

    def connect(self) -> bool:
        port = self._find_port()
        if not port:
            log("ESP32 not found - MOCK mode", "WARN")
            with S.lock:
                S.esp_ok = False
            return False
        try:
            self.ser = serial.Serial(port, cfg.esp_baud, timeout=1)
            time.sleep(2)
            firmware_ready = False
            deadline = time.time() + 3.0
            while time.time() < deadline and not firmware_ready:
                line = self._readline(0.2)
                if line:
                    log(f"ESP32: {line}")
                    if line.startswith("READY|actuator-v1"):
                        firmware_ready = True
            if not firmware_ready:
                firmware_ready = self._probe_protocol()
            if not firmware_ready:
                if self.ser is not None:
                    self.ser.close()
                self.ser = None
                self.connected = False
                with S.lock:
                    S.esp_ok = False
                log("ESP32 firmware is not agropick_unified_optimized.ino - MOCK mode", "WARN")
                return False
            self.connected = True
            with S.lock:
                S.esp_ok = True
            log(f"ESP32 connected on {port} (optimized protocol)", "SUCCESS")
            return True
        except Exception as exc:
            log(f"ESP32 failed: {exc} - MOCK mode", "WARN")
            self.connected = False
            self.ser = None
            with S.lock:
                S.esp_ok = False
            return False

    def rover(self, direction: str, speed: Optional[int] = None) -> None:
        if direction == "STOP":
            self._protocol_call("ROVER", "STOP", timeout=2.0)
        else:
            self._protocol_call("ROVER", direction, int(speed or 0), timeout=2.0)

    def pose(self, base: int, shoulder: int, wrist: int) -> None:
        self._protocol_call("POSE", int(base), int(shoulder), int(wrist), timeout=8.0)

    def gripper(self, angle: int) -> None:
        self._protocol_call("GRIPPER", int(angle), timeout=5.0)

    def rotgripper(self, angle: int) -> None:
        self._protocol_call("ROTGRIPPER", int(angle), timeout=5.0)

    def home(self) -> None:
        self._protocol_call("HOME", timeout=12.0)


serial_mgr = SerialManager()


class RoverController:
    def _set_state(self, direction: str, moving: bool) -> None:
        with S.lock:
            S.rover_dir = direction
            S.rover_moving = moving

    def forward(self, speed: int = 0) -> None:
        serial_mgr.rover("REV", speed or cfg.rover_speed)
        self._set_state("forward", True)

    def backward(self, speed: int = 0) -> None:
        serial_mgr.rover("FWD", speed or cfg.rover_speed)
        self._set_state("backward", True)

    def left(self, speed: int = 0) -> None:
        serial_mgr.rover("LEFT", speed or cfg.rover_turn_speed)
        self._set_state("left", True)

    def right(self, speed: int = 0) -> None:
        serial_mgr.rover("RIGHT", speed or cfg.rover_turn_speed)
        self._set_state("right", True)

    def stop(self) -> None:
        serial_mgr.rover("STOP")
        self._set_state("stopped", False)


rover = RoverController()


class ArmController:
    def __init__(self) -> None:
        self.positions = {
            "base": cfg.home_base,
            "shoulder": cfg.home_shoulder,
            "wrist": cfg.home_wrist,
            "gripper": cfg.home_gripper,
            "rotgripper": cfg.home_rotgrip,
        }
        self.busy = False
        with S.lock:
            S.arm_pos = self.positions.copy()

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        with S.lock:
            S.arm_busy = busy

    def _send(self, servo: str, angle: int) -> None:
        limits = {
            "base": (cfg.base_min, cfg.base_max),
            "shoulder": (cfg.shoulder_min, cfg.shoulder_max),
            "wrist": (cfg.wrist_fixed, cfg.wrist_fixed),
            "gripper": (cfg.gripper_min, cfg.gripper_max),
            "rotgripper": (cfg.rotgrip_min, cfg.rotgrip_max),
        }
        if servo in limits:
            lo, hi = limits[servo]
            angle = int(np.clip(angle, lo, hi))
        if servo in ("base", "shoulder", "wrist"):
            pose = {
                "base": self.positions["base"],
                "shoulder": self.positions["shoulder"],
                "wrist": self.positions["wrist"],
            }
            pose[servo] = angle
            serial_mgr.pose(pose["base"], pose["shoulder"], pose["wrist"])
        elif servo == "gripper":
            serial_mgr.gripper(angle)
        elif servo == "rotgripper":
            serial_mgr.rotgripper(angle)
        else:
            raise RuntimeError(f"Unknown servo: {servo}")
        self.positions[servo] = angle
        with S.lock:
            S.arm_pos = self.positions.copy()
        time.sleep(0.15)

    def solve_ik(self, x: float, y: float, z: float) -> Dict[str, int]:
        x = float(np.clip(x, 5, 35))
        y = float(np.clip(y, -20, 20))
        z = float(np.clip(z, 0, 25))
        angle_deg = np.degrees(np.arctan2(y, x))
        if cfg.invert_base:
            base = int(cfg.ik_base_center - angle_deg)
        else:
            base = int(cfg.ik_base_center + angle_deg)
        base = int(np.clip(base, cfg.base_min, cfg.base_max))
        shoulder = int(cfg.ik_shoulder_offset + z * cfg.ik_shoulder_mult)
        shoulder = int(np.clip(shoulder, cfg.shoulder_min, cfg.shoulder_max))
        return {"base": base, "shoulder": shoulder, "wrist": cfg.wrist_fixed}

    def move_to_xyz(self, x: float, y: float, z: float) -> None:
        angles = self.solve_ik(x, y, z)
        log(
            f"IK ({x:.1f},{y:.1f},{z:.1f}) -> base:{angles['base']} shoulder:{angles['shoulder']}",
            "ARM",
        )
        serial_mgr.pose(angles["base"], angles["shoulder"], cfg.wrist_fixed)
        self.positions.update(
            base=angles["base"],
            shoulder=angles["shoulder"],
            wrist=cfg.wrist_fixed,
        )
        with S.lock:
            S.arm_pos = self.positions.copy()
        time.sleep(cfg.servo_delay)

    def home(self) -> None:
        self._set_busy(True)
        try:
            log("Moving to HOME", "ARM")
            serial_mgr.home()
            self.positions = {
                "base": cfg.home_base,
                "shoulder": max(cfg.home_shoulder, 150),
                "wrist": cfg.home_wrist,
                "gripper": cfg.home_gripper,
                "rotgripper": cfg.home_rotgrip,
            }
            with S.lock:
                S.arm_pos = self.positions.copy()
            log("HOME complete", "SUCCESS")
        finally:
            self._set_busy(False)

    def open_gripper(self) -> None:
        self._set_busy(True)
        try:
            log("Open gripper", "ARM")
            self._send("gripper", cfg.gripper_min)
            time.sleep(0.3)
        finally:
            self._set_busy(False)

    def close_gripper(self, diameter_cm: float = 5.0) -> None:
        angle_range = cfg.gripper_max - cfg.gripper_min
        angle = cfg.gripper_min + int(
            (1 - min(diameter_cm, 8.0) / 8.0) * angle_range
        )
        angle = int(np.clip(angle, cfg.gripper_min, cfg.gripper_max))
        self._set_busy(True)
        try:
            log(f"Close gripper diameter {diameter_cm:.1f}cm -> {angle}", "ARM")
            self._send("gripper", angle)
            time.sleep(0.3)
        finally:
            self._set_busy(False)

    def twist(self) -> None:
        self._set_busy(True)
        try:
            log("Twisting", "ARM")
            home_rot = cfg.home_rotgrip
            for _ in range(cfg.twist_cycles):
                self._send("rotgripper", home_rot + cfg.twist_angle)
                time.sleep(cfg.twist_delay)
                self._send("rotgripper", home_rot - cfg.twist_angle)
                time.sleep(cfg.twist_delay)
            self._send("rotgripper", home_rot)
            time.sleep(cfg.twist_delay)
        finally:
            self._set_busy(False)

    def pick(self, x: float, y: float, diameter_cm: float = 5.0) -> bool:
        log(f"PICK at ({x:.1f},{y:.1f}) diameter {diameter_cm:.1f}cm", "PICK")
        self._set_busy(True)
        with S.lock:
            S.picks_try += 1
        try:
            log("Step 1/6: Approach high", "ARM")
            self.move_to_xyz(x, y, cfg.grab_h + cfg.approach_h)
            time.sleep(0.5)

            log("Step 2/6: Open gripper", "ARM")
            self._send("gripper", cfg.gripper_min)
            time.sleep(0.5)

            log("Step 3/6: Lower to target", "ARM")
            self.move_to_xyz(x, y, cfg.grab_h - cfg.pick_drop_extra)
            time.sleep(0.5)

            log("Step 4/6: Grip", "ARM")
            angle_range = cfg.gripper_max - cfg.gripper_min
            grip_angle = cfg.gripper_min + int(
                (1 - min(diameter_cm, 8.0) / 8.0) * angle_range
            )
            grip_angle = int(np.clip(grip_angle + cfg.grip_extra_close, cfg.gripper_min, cfg.gripper_max))
            self._send("gripper", grip_angle)
            time.sleep(0.5)

            log("Step 5/6: Twist", "ARM")
            home_rot = cfg.home_rotgrip
            for _ in range(cfg.twist_cycles):
                self._send("rotgripper", home_rot + cfg.twist_angle)
                time.sleep(cfg.twist_delay)
                self._send("rotgripper", home_rot - cfg.twist_angle)
                time.sleep(cfg.twist_delay)
            self._send("rotgripper", home_rot)
            time.sleep(0.5)

            log("Step 6/6: Lift and home", "ARM")
            self.move_to_xyz(x, y, cfg.grab_h + cfg.lift_h)
            time.sleep(0.5)
            # Safe return: raise shoulder first so arm lifts up before swinging base back.
            # Avoids the sudden backward jerk caused by snapping all servos at once via HOME.
            log("Moving to HOME", "ARM")
            self._send("shoulder", cfg.shoulder_max)
            time.sleep(0.6)
            serial_mgr.pose(cfg.home_base, cfg.shoulder_max, cfg.home_wrist)
            time.sleep(0.6)
            serial_mgr.pose(cfg.home_base, cfg.home_shoulder, cfg.home_wrist)
            time.sleep(0.4)
            self._send("rotgripper", cfg.home_rotgrip)
            time.sleep(0.3)
            self.positions = {
                "base": cfg.home_base,
                "shoulder": cfg.home_shoulder,
                "wrist": cfg.home_wrist,
                "gripper": cfg.home_gripper,
                "rotgripper": cfg.home_rotgrip,
            }
            with S.lock:
                S.arm_pos = self.positions.copy()
            log("HOME complete", "SUCCESS")
            self._send("gripper", cfg.gripper_min)

            with S.lock:
                S.picks_ok += 1
                S.detections = []
                S.ripe_count = 0
                S.unripe_count = 0
                S.scene_desc = "Last pick completed"
            log("PICK COMPLETE", "SUCCESS")
            return True
        except Exception as exc:
            log(f"Pick failed: {exc}", "ERROR")
            return False
        finally:
            self._set_busy(False)


arm = ArmController()


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


def parse_json_safe(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    candidates: List[str] = [cleaned]
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


def normalize_detection_result(result: Any) -> Dict[str, Any]:
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
        self.client: Optional[genai.Client] = None
        self.model = MODEL_NAME

    def init(self) -> bool:
        if not GOOGLE_API_KEY:
            log("No GOOGLE_API_KEY in .env", "WARN")
            return False
        try:
            self.client = genai.Client(api_key=GOOGLE_API_KEY)
            test = self.client.models.generate_content(model=self.model, contents="ping")
            with S.lock:
                S.gemini_ok = True
            log(f"Gemini OK: {(test.text or '')[:40]}", "SUCCESS")
            return True
        except Exception as exc:
            with S.lock:
                S.gemini_ok = False
            log(f"Gemini failed: {exc}", "ERROR")
            return False

    def _call(self, image_bytes: bytes, prompt: str) -> str:
        if self.client is None:
            raise RuntimeError("Gemini client not initialized")
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

    def _call_with_timeout(self, image_bytes: bytes, prompt: str) -> str:
        """Run _call in a thread with a hard timeout (cfg.gemini_timeout seconds).
        If the network is slow or the API hangs, raises TimeoutError so the
        caller can skip this scan and continue without blocking the rover."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(self._call, image_bytes, prompt)
            return future.result(timeout=cfg.gemini_timeout)

    def _norm_to_cm(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        x_cm = 5 + (x_norm / 1000.0) * 25
        y_cm = ((y_norm - 500) / 500.0) * 15
        return x_cm, y_cm

    def detect(self, image_bytes: bytes) -> Tuple[List[Dict[str, Any]], str]:
        if self.client is None:
            return [], ""
        raw_result = parse_json_safe(self._call_with_timeout(image_bytes, DETECT_PROMPT))
        result = normalize_detection_result(raw_result)
        detections: List[Dict[str, Any]] = []
        target = result["recommended_target"]

        for item in result["detections"]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "?"))
            is_ripe = bool(item.get("is_ripe", True))
            diameter = float(item.get("estimated_diameter_cm", 5.0) or 5.0)
            box = item.get("box_2d")
            point = item.get("pick_point", [500, 500])
            if not isinstance(point, list) or len(point) != 2:
                point = [500, 500]
            y_norm = clamp_norm(point[0])
            x_norm = clamp_norm(point[1])
            x_cm, y_cm = self._norm_to_cm(x_norm, y_norm)

            width_norm = 0
            if isinstance(box, list) and len(box) == 4:
                ymin, xmin, ymax, xmax = [clamp_norm(v) for v in box]
                box = [ymin, xmin, ymax, xmax]
                width_norm = abs(xmax - xmin)
            else:
                box = None

            detections.append(
                {
                    "label": label,
                    "is_ripe": is_ripe,
                    "diameter_cm": diameter,
                    "can_grip": diameter <= 8.0,
                    "box_norm": box,
                    "pick_norm": [y_norm, x_norm],
                    "width_norm": width_norm,
                    "robot_coords": (x_cm, y_cm),
                    "is_target": label == target,
                }
            )

        detections.sort(
            key=lambda item: (
                not item["is_target"],
                not item["is_ripe"],
                not item["can_grip"],
                -item["width_norm"],
            )
        )
        return detections, result["scene_description"]


gemini = GeminiController()


class PiCamera:
    def __init__(self) -> None:
        self.cam: Optional[Picamera2] = None
        self.ok = False
        self._pixel_format = PICAMERA_OUTPUT_FORMAT
        self._color_order = CAMERA_COLOR_ORDER
        self._swap_red_blue = CAMERA_SWAP_RED_BLUE

    def start(self) -> bool:
        try:
            self.cam = Picamera2()
            camera_config = self.cam.create_preview_configuration(
                main={"size": (cfg.cam_w, cfg.cam_h), "format": self._pixel_format}
            )
            self.cam.configure(camera_config)
            self.cam.start()
            time.sleep(1)
            if HAS_AF:
                try:
                    self.cam.set_controls(
                        {
                            "AfMode": libcam_controls.AfModeEnum.Continuous,
                            "AfSpeed": libcam_controls.AfSpeedEnum.Fast,
                        }
                    )
                except Exception:
                    pass
            self.ok = True
            log(
                f"Camera ready ({self._pixel_format}, order {self._color_order}, swap_rb={self._swap_red_blue})",
                "SUCCESS",
            )
            return True
        except Exception as exc:
            log(f"Camera failed: {exc}", "ERROR")
            return False

    def toggle_color_swap(self) -> None:
        self._swap_red_blue = not self._swap_red_blue
        log(f"Camera red/blue swap: {'ON' if self._swap_red_blue else 'OFF'}", "INFO")

    def capture_frame(self) -> Optional[np.ndarray]:
        if self.cam is None:
            return None
        frame = self.cam.capture_array()
        if frame.ndim != 3:
            return frame
        if frame.shape[2] == 4:
            if self._color_order == "RGB":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] == 3 and self._color_order == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self._swap_red_blue:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def frame_to_jpeg(self, frame: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Failed to encode JPEG")
        return buffer.tobytes()

    def stop(self) -> None:
        if self.cam is not None:
            self.cam.stop()


camera = PiCamera()


def update_detection_state(detections: List[Dict[str, Any]], scene_desc: str) -> None:
    with S.lock:
        S.detections = detections
        S.scene_desc = scene_desc
        S.ripe_count = sum(1 for item in detections if item["is_ripe"])
        S.unripe_count = max(0, len(detections) - S.ripe_count)


def best_pickable_detection() -> Optional[Dict[str, Any]]:
    with S.lock:
        detections = list(S.detections)
    ripe = [item for item in detections if item["is_ripe"] and item["can_grip"]]
    if ripe:
        return ripe[0]
    return detections[0] if detections else None


def trigger_detect(frame: np.ndarray) -> None:
    if not S.gemini_ok or arm.busy:
        return
    try:
        detections, scene_desc = gemini.detect(camera.frame_to_jpeg(frame))
        update_detection_state(detections, scene_desc)
        if detections:
            ripe_count = sum(1 for item in detections if item["is_ripe"])
            log(f"Detected {len(detections)} items ({ripe_count} ripe)", "GEMINI")
        else:
            log("No harvestable produce visible", "GEMINI")
    except concurrent.futures.TimeoutError:
        log(f"Gemini timeout after {cfg.gemini_timeout:.0f}s (bad network) — skipping scan", "WARN")
    except Exception as exc:
        log(f"Detection failed: {exc}", "ERROR")


def draw_overlay(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    out = frame.copy()
    height, width = out.shape[:2]

    for det in detections:
        box = det.get("box_norm")
        if isinstance(box, list) and len(box) == 4:
            ymin, xmin, ymax, xmax = [clamp_norm(v) for v in box]
            x1 = int(min(xmin, xmax) / 1000 * width)
            y1 = int(min(ymin, ymax) / 1000 * height)
            x2 = int(max(xmin, xmax) / 1000 * width)
            y2 = int(max(ymin, ymax) / 1000 * height)
            color = (0, 255, 0) if det["is_ripe"] else (0, 165, 255)
            if not det["can_grip"]:
                color = (0, 100, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{det['label']} {det['diameter_cm']:.1f}cm"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                out,
                text,
                (x1 + 2, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        pick_norm = det.get("pick_norm")
        if isinstance(pick_norm, list) and len(pick_norm) == 2:
            px = int(clamp_norm(pick_norm[1]) / 1000 * width)
            py = int(clamp_norm(pick_norm[0]) / 1000 * height)
            cv2.drawMarker(out, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)

    cv2.rectangle(out, (0, 0), (width, 32), (20, 20, 20), -1)
    with S.lock:
        status = f"{S.mode.upper()} | Ripe:{S.ripe_count} | Picks:{S.picks_ok}/{S.picks_try}"
        scene_desc = S.scene_desc
        esp_ok = S.esp_ok
    mode_color = {
        "manual": (150, 150, 150),
        "semi-auto": (0, 165, 255),
        "autonomous": (0, 255, 0),
    }.get(S.mode, (255, 255, 255))
    cv2.putText(out, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)
    cv2.circle(out, (width - 20, 16), 6, (0, 255, 0) if esp_ok else (0, 0, 255), -1)

    if scene_desc:
        cv2.rectangle(out, (0, 32), (width, 50), (10, 10, 40), -1)
        cv2.putText(
            out,
            scene_desc[:70],
            (8, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 255),
            1,
        )

    return out


def control_loop() -> None:
    log("Control loop started", "INFO")
    last_detect = 0.0
    scan_step_start: float = 0.0  # when the current autonomous step started

    while True:
        if not camera.ok:
            time.sleep(0.1)
            continue

        frame = camera.capture_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        now = time.time()

        # Semi-auto and manual use the timer-based detection as before.
        # Autonomous mode manages its own scan cadence via the step-scan cycle below.
        with S.lock:
            should_detect = (
                S.running
                and S.gemini_ok
                and not S.arm_busy
                and S.mode != "autonomous"
                and (now - last_detect) >= cfg.detect_interval
            )
            detections = list(S.detections)

        if should_detect:
            last_detect = now
            trigger_detect(frame)
            with S.lock:
                detections = list(S.detections)

        if S.mode == "autonomous" and S.running and not arm.busy:
            det = best_pickable_detection()

            if det and det["width_norm"] >= cfg.auto_pick_box_width_norm:
                # Confirmed fruit in range — stop and pick.
                if S.rover_moving:
                    rover.stop()
                    time.sleep(cfg.auto_stop_settle)
                x_cm, y_cm = det["robot_coords"]
                arm.pick(x_cm, y_cm, det["diameter_cm"])
                # After pick, reset so the next step starts fresh.
                last_detect = time.time()
                scan_step_start = time.time()
                time.sleep(cfg.auto_resume_delay)
                if S.running and S.mode == "autonomous":
                    rover.forward(cfg.auto_rover_speed)
                    scan_step_start = time.time()

            elif S.rover_moving:
                # Mid-step — check if the step duration has elapsed.
                if now - scan_step_start >= cfg.scan_step_duration:
                    # Step done: stop, let the camera settle, then scan with Gemini.
                    # Rover is stationary for the entire Gemini round-trip, so the
                    # detected position exactly matches the rover's actual position.
                    rover.stop()
                    time.sleep(cfg.scan_settle_time)
                    fresh = camera.capture_frame()
                    if fresh is not None:
                        trigger_detect(fresh)
                        with S.lock:
                            detections = list(S.detections)
                    last_detect = time.time()
                    # Don't restart rover here — next iteration checks detections and
                    # either picks (if det found) or kicks off the next step (else branch).

            else:
                # Rover is stopped and no in-range fruit — start the next step.
                rover.forward(cfg.auto_rover_speed)
                scan_step_start = time.time()

        elif S.mode == "semi-auto" and S.running and not arm.busy:
            det = best_pickable_detection()
            if det and not S.rover_moving and det["width_norm"] >= cfg.auto_pick_box_width_norm:
                x_cm, y_cm = det["robot_coords"]
                arm.pick(x_cm, y_cm, det["diameter_cm"])

        display = draw_overlay(frame, detections)
        try:
            encoded = camera.frame_to_jpeg(display)
            with S.lock:
                S.last_jpeg = encoded
        except Exception as exc:
            log(f"Preview encode failed: {exc}", "WARN")

        time.sleep(0.03)


app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>AgroPick</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#080c1a;--card:#111827;--border:#1e293b;--accent:#22d3ee;
  --green:#10b981;--red:#ef4444;--amber:#f59e0b;--text:#e2e8f0;
  --muted:#64748b;--surface:#1e293b;
}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
.header{
  background:var(--card);padding:12px 24px;
  display:flex;justify-content:space-between;align-items:center;
  border-bottom:1px solid var(--border);
}
.logo{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;color:var(--accent)}
.header-status{display:flex;gap:16px;font-size:13px;font-family:'JetBrains Mono',monospace}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:4px;background:var(--red)}
.dot.on{background:var(--green)}
.layout{display:grid;grid-template-columns:1fr 360px;gap:16px;padding:16px;max-width:1400px;margin:0 auto}
@media(max-width:900px){.layout{grid-template-columns:1fr}}
.card{background:var(--card);border-radius:10px;padding:16px;border:1px solid var(--border)}
.card h3{font-size:13px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:12px;font-family:'JetBrains Mono',monospace}
.feed{width:100%;border-radius:8px;background:#000;display:block}
.modes{display:flex;gap:8px;margin-bottom:12px}
.mode-btn{
  flex:1;padding:10px 8px;border:1.5px solid var(--border);border-radius:8px;
  background:transparent;color:var(--muted);cursor:pointer;font-size:13px;font-weight:600;
}
.mode-btn.active{border-color:var(--accent);color:var(--accent);background:rgba(34,211,238,.08)}
.mode-btn.active[data-mode="autonomous"]{border-color:var(--green);color:var(--green);background:rgba(16,185,129,.08)}
.ctrl-row{display:flex;gap:8px;margin-bottom:16px}
.btn{
  flex:1;padding:12px;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;
}
.btn-go{background:var(--green);color:#000}
.btn-stop{background:var(--red);color:#fff}
.btn-home{background:var(--surface);color:var(--text)}
.rover-pad{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:16px}
.r-btn{
  aspect-ratio:1;border:none;border-radius:8px;background:var(--surface);
  color:var(--text);font-size:20px;cursor:pointer;display:flex;align-items:center;justify-content:center;
}
.r-btn.r-stop{background:var(--red)}
.r-btn.empty{background:transparent;cursor:default}
.slider-wrap{margin-bottom:12px}
.slider-top{display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;font-family:'JetBrains Mono',monospace}
.slider-top .val{color:var(--accent)}
input[type=range]{width:100%;height:4px;border-radius:2px;background:var(--surface);outline:none;-webkit-appearance:none}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--accent);cursor:pointer;
}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.stat{background:var(--surface);padding:14px;border-radius:8px;text-align:center}
.stat-val{font-size:28px;font-weight:700;color:var(--accent);font-family:'JetBrains Mono',monospace}
.stat-lbl{font-size:11px;color:var(--muted);margin-top:2px;text-transform:uppercase;letter-spacing:.5px}
.arm-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.arm-btn{
  padding:10px;border:1px solid var(--border);border-radius:8px;background:transparent;color:var(--text);cursor:pointer;font-size:13px;
}
.ik-panel{background:var(--bg);border-radius:8px;padding:12px;margin-top:12px}
.ik-panel label{font-size:11px;color:var(--muted);display:block;margin-bottom:2px}
.ik-input{
  width:100%;padding:7px 8px;border:1px solid var(--border);border-radius:6px;
  background:var(--card);color:var(--text);font-size:13px;margin-bottom:8px;font-family:'JetBrains Mono',monospace;
}
.btn-save{
  width:100%;padding:9px;border:none;border-radius:6px;background:var(--green);color:#000;font-weight:600;cursor:pointer;
}
.log-box{
  height:180px;overflow-y:auto;background:var(--bg);border-radius:8px;padding:8px;
  font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.5;
}
.log-line{border-bottom:1px solid var(--border);padding:2px 0}
</style>
</head>
<body>
<div class="header">
  <div class="logo">AGROPICK</div>
  <div class="header-status">
    <span><span class="dot" id="espDot"></span>ESP32</span>
    <span><span class="dot" id="gemDot"></span>Gemini</span>
    <span id="modeLabel">MANUAL</span>
  </div>
</div>
<div class="layout">
  <div>
    <div class="card">
      <img src="/video_feed" class="feed">
    </div>
    <div class="card" style="margin-top:12px">
      <div class="modes">
        <button class="mode-btn active" data-mode="manual" onclick="setMode('manual',this)">Manual</button>
        <button class="mode-btn" data-mode="semi-auto" onclick="setMode('semi-auto',this)">Semi-Auto</button>
        <button class="mode-btn" data-mode="autonomous" onclick="setMode('autonomous',this)">Autonomous</button>
      </div>
      <div class="ctrl-row">
        <button class="btn btn-go" id="startBtn" onclick="toggleSys()">START</button>
        <button class="btn btn-home" onclick="api('arm/home')">HOME</button>
        <button class="btn btn-home" onclick="api('scan')">SCAN</button>
      </div>
      <h3>Rover</h3>
      <div class="rover-pad">
        <div class="r-btn empty"></div>
        <button class="r-btn" onmousedown="api('rover/f')" onmouseup="api('rover/s')" ontouchstart="api('rover/f')" ontouchend="api('rover/s')">&#9650;</button>
        <div class="r-btn empty"></div>
        <button class="r-btn" onmousedown="api('rover/l')" onmouseup="api('rover/s')" ontouchstart="api('rover/l')" ontouchend="api('rover/s')">&#9664;</button>
        <button class="r-btn r-stop" onclick="api('rover/s')">&#9632;</button>
        <button class="r-btn" onmousedown="api('rover/r')" onmouseup="api('rover/s')" ontouchstart="api('rover/r')" ontouchend="api('rover/s')">&#9654;</button>
        <div class="r-btn empty"></div>
        <button class="r-btn" onmousedown="api('rover/b')" onmouseup="api('rover/s')" ontouchstart="api('rover/b')" ontouchend="api('rover/s')">&#9660;</button>
        <div class="r-btn empty"></div>
      </div>
      <div class="slider-wrap">
        <div class="slider-top"><span>Speed</span><span class="val" id="spdVal">60</span></div>
        <input type="range" min="30" max="255" value="60" oninput="setSpeed(this.value)">
      </div>
      <div class="ctrl-row">
        <button class="btn btn-home" onclick="api('camera/color')">SWAP COLORS</button>
      </div>
    </div>
  </div>
  <div>
    <div class="card">
      <h3>Stats</h3>
      <div class="stats">
        <div class="stat"><div class="stat-val" id="sOk">0</div><div class="stat-lbl">Picked</div></div>
        <div class="stat"><div class="stat-val" id="sTry">0</div><div class="stat-lbl">Attempts</div></div>
        <div class="stat"><div class="stat-val" id="sRipe">0</div><div class="stat-lbl">Ripe</div></div>
        <div class="stat"><div class="stat-val" id="sUnripe">0</div><div class="stat-lbl">Other</div></div>
      </div>
    </div>
    <div class="card" style="margin-top:12px">
      <h3>Arm</h3>
      <div class="arm-grid">
        <button class="arm-btn" onclick="api('arm/home')">Home</button>
        <button class="arm-btn" onclick="api('arm/pick')">Pick Best</button>
        <button class="arm-btn" onclick="api('arm/open')">Open</button>
        <button class="arm-btn" onclick="api('arm/close')">Close</button>
        <button class="arm-btn" onclick="api('arm/twist')">Twist</button>
        <button class="arm-btn" id="armSt" style="cursor:default">Ready</button>
      </div>
      <div class="ik-panel">
        <h3 style="color:var(--accent)">IK Tuning</h3>
        <label>Base Center</label>
        <input type="number" class="ik-input" id="ikBC" value="110" min="70" max="160">
        <label>Shoulder Offset</label>
        <input type="number" class="ik-input" id="ikSO" value="110" min="90" max="130" step="0.1">
        <label>Shoulder Multiplier</label>
        <input type="number" class="ik-input" id="ikSM" value="1.2" min="0.5" max="2.0" step="0.1">
        <label style="margin-bottom:8px;display:flex;align-items:center;gap:6px">
          <input type="checkbox" id="ikIB">Invert Base
        </label>
        <label>Extra Drop (cm)</label>
        <input type="number" class="ik-input" id="ikDE" value="2.0" min="0" max="10" step="0.5">
        <label>Extra Grip Close (deg)</label>
        <input type="number" class="ik-input" id="ikGE" value="10" min="0" max="40" step="1">
        <button class="btn-save" onclick="saveIK()">Save IK</button>
      </div>
    </div>
    <div class="card" style="margin-top:12px">
      <h3>Log</h3>
      <div class="log-box" id="logBox"></div>
    </div>
  </div>
</div>
<script>
let running=false;
function api(ep,data){
  const opts=data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{};
  return fetch('/api/'+ep,opts).then(r=>r.json());
}
function setMode(mode,el){
  api('mode',{mode});
  document.querySelectorAll('.mode-btn').forEach(btn=>btn.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('modeLabel').textContent=mode.toUpperCase();
}
function toggleSys(){
  running=!running;
  api('system',{running});
  const btn=document.getElementById('startBtn');
  btn.textContent=running?'STOP':'START';
  btn.className='btn '+(running?'btn-stop':'btn-go');
}
function setSpeed(v){
  document.getElementById('spdVal').textContent=v;
  api('speed',{speed:+v});
}
function saveIK(){
  api('ik',{
    base_center:+document.getElementById('ikBC').value,
    shoulder_offset:+document.getElementById('ikSO').value,
    shoulder_mult:+document.getElementById('ikSM').value,
    invert_base:document.getElementById('ikIB').checked,
    pick_drop_extra:+document.getElementById('ikDE').value,
    grip_extra_close:+document.getElementById('ikGE').value
  });
}
function poll(){
  api('status').then(d=>{
    document.getElementById('espDot').className='dot'+(d.esp_ok?' on':'');
    document.getElementById('gemDot').className='dot'+(d.gemini_ok?' on':'');
    document.getElementById('sOk').textContent=d.picks_ok;
    document.getElementById('sTry').textContent=d.picks_try;
    document.getElementById('sRipe').textContent=d.ripe_count;
    document.getElementById('sUnripe').textContent=d.unripe_count;
    document.getElementById('armSt').textContent=d.arm_busy?'Busy...':'Ready';
    document.getElementById('modeLabel').textContent=d.mode.toUpperCase();
    running=d.running;
    const btn=document.getElementById('startBtn');
    btn.textContent=running?'STOP':'START';
    btn.className='btn '+(running?'btn-stop':'btn-go');
  }).catch(()=>{});
}
function pollLogs(){
  api('logs').then(d=>{
    const box=document.getElementById('logBox');
    box.innerHTML=d.logs.map(line=>'<div class="log-line">'+line+'</div>').join('');
    box.scrollTop=box.scrollHeight;
  }).catch(()=>{});
}
api('ik').then(d=>{
  document.getElementById('ikBC').value=d.base_center;
  document.getElementById('ikSO').value=d.shoulder_offset;
  document.getElementById('ikSM').value=d.shoulder_mult;
  document.getElementById('ikIB').checked=d.invert_base;
  document.getElementById('ikDE').value=d.pick_drop_extra;
  document.getElementById('ikGE').value=d.grip_extra_close;
}).catch(()=>{});
setInterval(poll,600);
setInterval(pollLogs,1200);
</script>
</body>
</html>
"""


@app.route("/")
def index() -> str:
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed() -> Response:
    def gen() -> Any:
        while True:
            with S.lock:
                frame = S.last_jpeg
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.033)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status() -> Response:
    with S.lock:
        return jsonify(
            {
                "mode": S.mode,
                "running": S.running,
                "esp_ok": S.esp_ok,
                "gemini_ok": S.gemini_ok,
                "rover_moving": S.rover_moving,
                "rover_dir": S.rover_dir,
                "arm_busy": S.arm_busy,
                "arm_pos": S.arm_pos,
                "scene_desc": S.scene_desc,
                "ripe_count": S.ripe_count,
                "unripe_count": S.unripe_count,
                "picks_ok": S.picks_ok,
                "picks_try": S.picks_try,
            }
        )


@app.route("/api/logs")
def api_logs() -> Response:
    with S.lock:
        return jsonify({"logs": S.logs[-60:]})


@app.route("/api/mode", methods=["POST"])
def api_mode() -> Response:
    data = request.get_json(force=True) or {}
    new_mode = data.get("mode", "manual")
    if S.mode == "autonomous" and new_mode != "autonomous":
        rover.stop()
    with S.lock:
        S.mode = new_mode
    log(f"Mode: {new_mode}")
    return jsonify({"ok": True})


@app.route("/api/system", methods=["POST"])
def api_system() -> Response:
    data = request.get_json(force=True) or {}
    running = bool(data.get("running", False))
    with S.lock:
        S.running = running
    log(f"System {'ON' if running else 'OFF'}")
    if not running:
        rover.stop()
    return jsonify({"ok": True})


@app.route("/api/speed", methods=["POST"])
def api_speed() -> Response:
    data = request.get_json(force=True) or {}
    speed = int(np.clip(int(data.get("speed", cfg.rover_speed)), 30, 255))
    cfg.rover_speed = speed
    cfg.rover_turn_speed = max(30, speed - 15)
    cfg.auto_rover_speed = max(30, speed - 15)
    cfg.save()
    return jsonify({"ok": True})


@app.route("/api/ik", methods=["GET", "POST"])
def api_ik() -> Response:
    if request.method == "GET":
        return jsonify(
            {
                "base_center": cfg.ik_base_center,
                "shoulder_offset": cfg.ik_shoulder_offset,
                "shoulder_mult": cfg.ik_shoulder_mult,
                "invert_base": cfg.invert_base,
                "pick_drop_extra": cfg.pick_drop_extra,
                "grip_extra_close": cfg.grip_extra_close,
            }
        )
    data = request.get_json(force=True) or {}
    cfg.ik_base_center = int(data.get("base_center", cfg.ik_base_center))
    cfg.ik_shoulder_offset = float(data.get("shoulder_offset", cfg.ik_shoulder_offset))
    cfg.ik_shoulder_mult = float(data.get("shoulder_mult", cfg.ik_shoulder_mult))
    cfg.invert_base = bool(data.get("invert_base", cfg.invert_base))
    cfg.pick_drop_extra = float(data.get("pick_drop_extra", cfg.pick_drop_extra))
    cfg.grip_extra_close = int(data.get("grip_extra_close", cfg.grip_extra_close))
    cfg.save()
    log("IK saved", "INFO")
    return jsonify({"ok": True})


@app.route("/api/scan", methods=["GET", "POST"])
def api_scan() -> Response:
    if not camera.ok or not S.gemini_ok or arm.busy:
        return jsonify({"ok": False})

    def worker() -> None:
        frame = camera.capture_frame()
        if frame is not None:
            trigger_detect(frame)

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/camera/color")
def api_camera_color() -> Response:
    camera.toggle_color_swap()
    return jsonify({"ok": True})


@app.route("/api/rover/<cmd>")
def api_rover(cmd: str) -> Response:
    {"f": rover.forward, "b": rover.backward, "l": rover.left, "r": rover.right, "s": rover.stop}.get(cmd, rover.stop)()
    return jsonify({"ok": True})


@app.route("/api/arm/home")
def api_arm_home() -> Response:
    threading.Thread(target=arm.home, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/arm/open")
def api_arm_open() -> Response:
    threading.Thread(target=arm.open_gripper, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/arm/close")
def api_arm_close() -> Response:
    threading.Thread(target=arm.close_gripper, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/arm/twist")
def api_arm_twist() -> Response:
    threading.Thread(target=arm.twist, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/arm/pick")
def api_arm_pick() -> Response:
    def do_pick() -> None:
        det = best_pickable_detection()
        if det is None:
            log("Nothing to pick", "WARN")
            return
        x_cm, y_cm = det["robot_coords"]
        arm.pick(x_cm, y_cm, det["diameter_cm"])

    threading.Thread(target=do_pick, daemon=True).start()
    return jsonify({"ok": True})


def main() -> None:
    log("AgroPick starting", "INFO")
    serial_mgr.connect()
    if not camera.start():
        log("Camera required - exiting", "ERROR")
        return
    gemini.init()
    if serial_mgr.connected:
        arm.home()
    else:
        log("ESP32 not connected - running in mock mode", "WARN")
    threading.Thread(target=control_loop, daemon=True).start()
    log(f"Ready - http://0.0.0.0:{cfg.port}", "SUCCESS")
    app.run(host="0.0.0.0", port=cfg.port, threaded=True, debug=False)


if __name__ == "__main__":
    try:
        main()
    finally:
        camera.stop()
