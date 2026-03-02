#!/usr/bin/env python3
"""
AgroPick Production — Gemini Vision + Flask + Minimal ESP32 Actuator

Hardware:
  Raspberry Pi 4 + Pi Camera Module 3
  ESP32 minimal actuator (single-char ACK protocol over USB serial)
  PCA9685 5-DOF arm (base, shoulder, wrist[fixed], rotgripper, gripper)
  L298N dual motor driver for rover

Three Modes:
  Manual     — control rover + arm entirely from web UI
  Semi-Auto  — drive rover from UI, arm auto-picks when produce detected
  Autonomous — rover drives forward, stops on detection, arm picks, resumes

ESP32 Protocol (115200 baud):
  Servo:  b:110  s:130  w:160  g:40  r:130   → 'K' (ok) or 'E' (error)
  Home:   H                                    → 'K' then 'D' when done
  Query:  ?                                    → 'I' (idle) or 'M' (moving)
  Rover:  F120  B120  L100  R100  S            → 'K'
  Pos:    P                                    → csv: base,shld,wrst,rotg,grip,enc1,enc2

Motor wiring note:
  Physical forward = ESP32 'B' command (motors wired inverted)
  Physical backward = ESP32 'F' command
"""

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import serial
import serial.tools.list_ports
from flask import Flask, Response, jsonify, render_template_string, request
from flask_cors import CORS
from picamera2 import Picamera2

try:
    from libcamera import controls as libcam_controls
    HAS_AF = True
except ImportError:
    HAS_AF = False

# Gemini
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

CFG_FILE = "agropick_config.json"


@dataclass
class Cfg:
    # Camera
    cam_w: int = 640
    cam_h: int = 480
    jpeg_quality: int = 70
    focal_length: float = 1400.0
    cam_x: float = 0.0
    cam_y: float = -2.0
    cam_z: float = 45.0
    y_scale: float = 0.9

    # Serial
    esp_port: str = "/dev/ttyUSB0"
    esp_baud: int = 115200

    # Rover
    rover_speed: int = 120
    rover_turn_speed: int = 100
    auto_rover_speed: int = 100
    auto_stop_settle: float = 0.5
    auto_resume_delay: float = 1.0

    # Arm servo limits
    base_min: int = 70;       base_max: int = 160
    shoulder_min: int = 120;  shoulder_max: int = 160
    wrist_fixed: int = 160
    gripper_min: int = 30;    gripper_max: int = 90
    rotgrip_min: int = 130;   rotgrip_max: int = 160

    # Home
    home_base: int = 110;     home_shoulder: int = 130
    home_wrist: int = 160;    home_rotgrip: int = 130
    home_gripper: int = 40

    # Gripper calibration
    grip_open_angle: int = 30;   grip_open_gap: float = 8.0
    grip_closed_angle: int = 90; grip_closed_gap: float = 0.0
    grip_squeeze: float = 0.5;   grip_max_cap: float = 8.0

    # Pick geometry
    approach_h: float = 8.0
    grab_h: float = 3.0
    table_z: float = 4.8
    lift_h: float = 10.0
    twist_angle: int = 15
    twist_cycles: int = 3
    twist_delay: float = 0.25

    # IK
    ik_base_center: int = 110
    ik_shoulder_offset: float = 110.0
    ik_shoulder_mult: float = 1.2
    invert_base: bool = True
    invert_y: bool = True

    # Timing
    pick_step_delay: float = 0.4

    # Detection
    tomato_diam_ref: float = 6.0
    detect_interval: float = 3.0
    min_det_px: int = 50

    # Server
    port: int = 5000

    def save(self):
        with open(CFG_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls):
        if os.path.exists(CFG_FILE):
            try:
                with open(CFG_FILE) as f:
                    return cls(**json.load(f))
            except Exception:
                pass
        return cls()


cfg = Cfg.load()


# ═══════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════

class State:
    def __init__(self):
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
        self.detections: List[Dict] = []
        self.scene_desc = ""
        self.ripe_count = 0
        self.unripe_count = 0
        self.picks_ok = 0
        self.picks_try = 0
        self.logs: List[str] = []


S = State()


def log(msg: str, lvl: str = "INFO"):
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}][{lvl}] {msg}"
    print(entry, flush=True)
    with S.lock:
        S.logs.append(entry)
        if len(S.logs) > 100:
            S.logs.pop(0)


# ═══════════════════════════════════════════════════════════════
# ESP32 SERIAL
# ═══════════════════════════════════════════════════════════════

class ESP32:
    def __init__(self):
        self.ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()

    def find_port(self) -> Optional[str]:
        for p in serial.tools.list_ports.comports():
            d = p.description.lower()
            if any(x in d for x in ("cp210", "ch340", "usb", "uart")):
                return p.device
            if "/dev/ttyUSB" in p.device or "/dev/ttyACM" in p.device:
                return p.device
        return None

    def connect(self) -> bool:
        port = cfg.esp_port
        if not os.path.exists(port):
            port = self.find_port()
        if not port:
            log("ESP32 not found", "ERR")
            return False
        try:
            self.ser = serial.Serial(port, cfg.esp_baud, timeout=0.15)
            time.sleep(2.5)
            deadline = time.time() + 4.0
            while time.time() < deadline:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if line == "D":
                        S.esp_ok = True
                        log(f"ESP32 ready on {port}", "OK")
                        return True
                time.sleep(0.01)
            S.esp_ok = True
            log(f"ESP32 on {port} (no boot ACK)", "WARN")
            return True
        except Exception as e:
            log(f"ESP32 fail: {e}", "ERR")
            return False

    def cmd(self, command: str) -> str:
        with self._lock:
            if not self.ser or not S.esp_ok:
                return "K"
            try:
                while self.ser.in_waiting:
                    self.ser.read(self.ser.in_waiting)
                self.ser.write(f"{command}\n".encode())
                self.ser.flush()
                resp = self.ser.readline().decode("utf-8", errors="ignore").strip()
                return resp
            except Exception as e:
                log(f"Serial: {e}", "ERR")
                return ""

    def is_moving(self) -> bool:
        return self.cmd("?") == "M"

    def wait_idle(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_moving():
                return True
            time.sleep(0.03)
        return False


esp = ESP32()


# ═══════════════════════════════════════════════════════════════
# ROVER — motor wiring is inverted: B=physical forward
# ═══════════════════════════════════════════════════════════════

class Rover:
    def forward(self, speed: int = 0):
        spd = speed or cfg.rover_speed
        esp.cmd(f"B{spd}")  # B = physical forward (wiring inverted)
        S.rover_moving = True
        S.rover_dir = "forward"

    def backward(self, speed: int = 0):
        spd = speed or cfg.rover_speed
        esp.cmd(f"F{spd}")  # F = physical backward
        S.rover_moving = True
        S.rover_dir = "backward"

    def left(self, speed: int = 0):
        spd = speed or cfg.rover_turn_speed
        esp.cmd(f"L{spd}")
        S.rover_moving = True
        S.rover_dir = "left"

    def right(self, speed: int = 0):
        spd = speed or cfg.rover_turn_speed
        esp.cmd(f"R{spd}")
        S.rover_moving = True
        S.rover_dir = "right"

    def stop(self):
        esp.cmd("S")
        S.rover_moving = False
        S.rover_dir = "stopped"


rover = Rover()


# ═══════════════════════════════════════════════════════════════
# ARM
# ═══════════════════════════════════════════════════════════════

class Arm:
    # Map servo names to single-char ESP32 codes
    CODE = {"base": "b", "shoulder": "s", "wrist": "w",
            "gripper": "g", "rotgripper": "r"}

    def __init__(self):
        self._pos = {
            "base": cfg.home_base, "shoulder": cfg.home_shoulder,
            "wrist": cfg.home_wrist, "gripper": cfg.home_gripper,
            "rotgripper": cfg.home_rotgrip,
        }

    def _limits(self, servo: str) -> Tuple[int, int]:
        return {
            "base": (cfg.base_min, cfg.base_max),
            "shoulder": (cfg.shoulder_min, cfg.shoulder_max),
            "wrist": (cfg.wrist_fixed, cfg.wrist_fixed),
            "gripper": (cfg.gripper_min, cfg.gripper_max),
            "rotgripper": (cfg.rotgrip_min, cfg.rotgrip_max),
        }.get(servo, (0, 180))

    def _send(self, servo: str, angle: int) -> bool:
        lo, hi = self._limits(servo)
        angle = int(np.clip(angle, lo, hi))
        code = self.CODE.get(servo, servo)
        resp = esp.cmd(f"{code}:{angle}")
        if resp == "K":
            self._pos[servo] = angle
            S.arm_pos = self._pos.copy()
            return True
        return False

    def _send_wait(self, servo: str, angle: int):
        self._send(servo, angle)
        esp.wait_idle(5.0)

    # IK — ported from production code
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

    def move_xyz(self, x: float, y: float, z: float):
        angles = self.solve_ik(x, y, z)
        log(f"IK ({x:.1f},{y:.1f},{z:.1f}) b:{angles['base']} s:{angles['shoulder']}")
        self._send("base", angles["base"])
        self._send("shoulder", angles["shoulder"])
        self._send("wrist", cfg.wrist_fixed)
        esp.wait_idle(6.0)

    def home(self):
        log("Arm home")
        S.arm_busy = True
        # Explicit individual commands — keeps Python in sync
        self._send("gripper", cfg.home_gripper)
        esp.wait_idle(3.0)
        self._send("wrist", cfg.home_wrist)
        esp.wait_idle(3.0)
        self._send("shoulder", cfg.home_shoulder)
        esp.wait_idle(3.0)
        self._send("base", cfg.home_base)
        esp.wait_idle(3.0)
        self._send("rotgripper", cfg.home_rotgrip)
        esp.wait_idle(3.0)
        self._pos = {
            "base": cfg.home_base, "shoulder": cfg.home_shoulder,
            "wrist": cfg.home_wrist, "gripper": cfg.home_gripper,
            "rotgripper": cfg.home_rotgrip,
        }
        S.arm_pos = self._pos.copy()
        S.arm_busy = False
        log("Home done", "OK")

    def open_gripper(self):
        self._send_wait("gripper", cfg.grip_open_angle)

    def close_gripper(self, diameter_cm: float = 5.0):
        angle_range = cfg.grip_closed_angle - cfg.grip_open_angle
        gap_range = cfg.grip_open_gap - cfg.grip_closed_gap
        if gap_range == 0:
            angle = cfg.grip_open_angle
        else:
            deg_per_cm = angle_range / gap_range
            target_gap = max(0.5, diameter_cm - cfg.grip_squeeze)
            angle = cfg.grip_open_angle + (cfg.grip_open_gap - target_gap) * deg_per_cm
        angle = int(np.clip(angle, cfg.gripper_min, cfg.gripper_max))
        self._send_wait("gripper", angle)

    def twist(self):
        home_rot = cfg.home_rotgrip
        for _ in range(cfg.twist_cycles):
            self._send("rotgripper", home_rot + cfg.twist_angle)
            time.sleep(cfg.twist_delay)
            self._send("rotgripper", home_rot - cfg.twist_angle)
            time.sleep(cfg.twist_delay)
        self._send_wait("rotgripper", home_rot)

    def pick(self, x: float, y: float, z: float, diameter: float) -> bool:
        if diameter > cfg.grip_max_cap:
            log(f"Too big: {diameter:.1f}cm", "WARN")
            return False
        log(f"PICK ({x:.1f},{y:.1f},{z:.1f}) d={diameter:.1f}cm")
        S.arm_busy = True
        S.picks_try += 1
        try:
            grab_z = cfg.grab_h
            approach_z = grab_z + cfg.approach_h

            log("1/6 approach")
            self.move_xyz(x, y, approach_z)
            time.sleep(cfg.pick_step_delay)

            log("2/6 open")
            self.open_gripper()
            time.sleep(cfg.pick_step_delay)

            log("3/6 lower")
            self.move_xyz(x, y, grab_z)
            time.sleep(cfg.pick_step_delay)

            log("4/6 grip")
            self.close_gripper(diameter)
            time.sleep(cfg.pick_step_delay)

            log("5/6 twist")
            self.twist()
            time.sleep(cfg.pick_step_delay)

            log("6/6 lift+home")
            self.move_xyz(x, y, grab_z + cfg.lift_h)
            time.sleep(0.3)

            # Return home and drop
            self._send("base", cfg.home_base)
            esp.wait_idle(3.0)
            self._send("shoulder", cfg.home_shoulder)
            esp.wait_idle(3.0)
            self._send("rotgripper", cfg.home_rotgrip)
            esp.wait_idle(3.0)
            self.open_gripper()
            time.sleep(0.3)

            self._pos = {
                "base": cfg.home_base, "shoulder": cfg.home_shoulder,
                "wrist": cfg.home_wrist, "gripper": cfg.home_gripper,
                "rotgripper": cfg.home_rotgrip,
            }
            S.arm_pos = self._pos.copy()
            S.picks_ok += 1
            log("PICK OK", "OK")
            return True
        except Exception as e:
            log(f"Pick fail: {e}", "ERR")
            return False
        finally:
            S.arm_busy = False


arm = Arm()


# ═══════════════════════════════════════════════════════════════
# CAMERA
# ═══════════════════════════════════════════════════════════════

class Camera:
    def __init__(self):
        self.cam: Optional[Picamera2] = None
        self.ok = False

    def start(self) -> bool:
        try:
            self.cam = Picamera2()
            c = self.cam.create_preview_configuration(
                main={"size": (cfg.cam_w, cfg.cam_h), "format": "RGB888"}
            )
            self.cam.configure(c)
            self.cam.start()
            time.sleep(1)
            if HAS_AF:
                self.cam.set_controls({
                    "AfMode": libcam_controls.AfModeEnum.Continuous,
                    "AfSpeed": libcam_controls.AfSpeedEnum.Fast,
                })
            self.ok = True
            log("Camera ready", "OK")
            return True
        except Exception as e:
            log(f"Camera fail: {e}", "ERR")
            return False

    def frame(self) -> Optional[np.ndarray]:
        if not self.cam:
            return None
        f = self.cam.capture_array()
        if f.ndim == 3 and f.shape[2] == 3:
            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        return f

    def jpeg(self, f: np.ndarray) -> bytes:
        _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        return buf.tobytes()

    def stop(self):
        if self.cam:
            self.cam.stop()


camera = Camera()


# ═══════════════════════════════════════════════════════════════
# PIXEL → ROBOT COORDS (ported from production code)
# ═══════════════════════════════════════════════════════════════

def pixel_to_robot(cx: int, cy: int, distance_cm: float) -> Tuple[float, float, float]:
    """Convert pixel center + distance to robot arm coordinates."""
    px_off_x = cx - (cfg.cam_w // 2)
    px_off_y = cy - (cfg.cam_h // 2)
    real_off_x = (px_off_x * distance_cm) / cfg.focal_length
    real_off_y = (px_off_y * distance_cm) / cfg.focal_length
    robot_x = cfg.cam_x + real_off_y
    if cfg.invert_y:
        robot_y = cfg.cam_y - real_off_x * cfg.y_scale
    else:
        robot_y = cfg.cam_y + real_off_x
    robot_z = cfg.table_z
    return (robot_x, robot_y, robot_z)


# ═══════════════════════════════════════════════════════════════
# GEMINI VISION
# ═══════════════════════════════════════════════════════════════

DETECT_PROMPT = """
You are controlling a robot arm for agricultural harvesting.
Analyze this image and detect any visible fruits or vegetables.

Return ONLY a JSON object with no explanation and no markdown:
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
  "scene_description": "brief one line",
  "recommended_target": "label of best to pick, or null"
}

Rules:
- Detect tomato, brinjal, eggplant, okra, cucumber, chili, capsicum,
  pepper, bean, carrot, onion, banana, apple, mango, orange, etc.
- Use normalized coordinates 0-1000.
- Empty detections and null target if nothing suitable.
- is_ripe=true when harvestable.
""".strip()


def parse_json_safe(text: str) -> Any:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
                     flags=re.IGNORECASE | re.DOTALL).strip()
    for candidate in [cleaned,
                      (m.group(0) if (m := re.search(r"\{[\s\S]*\}", cleaned)) else None),
                      (m.group(0) if (m := re.search(r"\[[\s\S]*\]", cleaned)) else None)]:
        if candidate:
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
    raise ValueError(f"No JSON: {cleaned[:100]}")


def clamp_norm(v: Any) -> int:
    try:
        return int(np.clip(int(v), 0, 1000))
    except Exception:
        return 0


class Vision:
    def __init__(self):
        self.client = None
        self.model = GEMINI_MODEL

    def init(self) -> bool:
        if not GOOGLE_API_KEY:
            log("No GOOGLE_API_KEY in .env", "ERR")
            return False
        try:
            self.client = genai.Client(api_key=GOOGLE_API_KEY)
            test = self.client.models.generate_content(model=self.model, contents="ping")
            S.gemini_ok = True
            log(f"Gemini OK ({self.model})", "OK")
            return True
        except Exception as e:
            log(f"Gemini fail: {e}", "ERR")
            return False

    def detect(self, jpeg_bytes: bytes) -> List[Dict]:
        """Run Gemini detection, return list of detection dicts with robot coords."""
        if not self.client:
            return []
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
                    DETECT_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )
            raw = parse_json_safe((resp.text or "").strip())
        except Exception as e:
            log(f"Detect err: {e}", "ERR")
            return []

        # Normalize
        if isinstance(raw, dict):
            dets = raw.get("detections", [])
            if not isinstance(dets, list):
                dets = []
            S.scene_desc = str(raw.get("scene_description", ""))
        elif isinstance(raw, list):
            dets = [d for d in raw if isinstance(d, dict)]
            S.scene_desc = ""
        else:
            return []

        results = []
        for det in dets:
            label = str(det.get("label", "?"))
            is_ripe = bool(det.get("is_ripe", True))
            diameter = float(det.get("estimated_diameter_cm", 5.0) or 5.0)
            can_grip = diameter <= cfg.grip_max_cap

            # Bounding box → pixel center + size for distance estimation
            box = det.get("box_2d")
            point = det.get("pick_point", [500, 500])
            if not isinstance(point, list) or len(point) != 2:
                point = [500, 500]

            # Convert normalized coords to pixel coords
            y_norm, x_norm = clamp_norm(point[0]), clamp_norm(point[1])
            cx_px = int(x_norm / 1000 * cfg.cam_w)
            cy_px = int(y_norm / 1000 * cfg.cam_h)

            # Estimate distance from bbox width
            if isinstance(box, list) and len(box) == 4:
                ymin, xmin, ymax, xmax = [clamp_norm(v) for v in box]
                w_px = abs(xmax - xmin) / 1000 * cfg.cam_w
                h_px = abs(ymax - ymin) / 1000 * cfg.cam_h
            else:
                w_px = 80
                h_px = 80

            if w_px > cfg.min_det_px:
                distance = (cfg.tomato_diam_ref * cfg.focal_length) / w_px
            else:
                distance = 30.0

            robot_coords = pixel_to_robot(cx_px, cy_px, distance)

            results.append({
                "label": label,
                "is_ripe": is_ripe,
                "diameter": diameter,
                "can_grip": can_grip,
                "box_norm": box,
                "pick_norm": [y_norm, x_norm],
                "center_px": (cx_px, cy_px),
                "distance_cm": distance,
                "robot_coords": robot_coords,
            })

        # Sort: ripe+grippable first
        results.sort(key=lambda d: (not d["is_ripe"], not d["can_grip"]))
        return results


vision = Vision()


# ═══════════════════════════════════════════════════════════════
# CONTROL LOOP
# ═══════════════════════════════════════════════════════════════

def control_loop():
    """Main loop: capture → detect → draw → mode logic."""
    log("Control loop started")
    last_detect = 0.0

    while True:
        if not camera.ok:
            time.sleep(0.1)
            continue

        frame = camera.frame()
        if frame is None:
            time.sleep(0.1)
            continue

        now = time.time()
        detections = []

        # Run detection periodically when system is running
        # (or always in semi-auto/autonomous so we have fresh data)
        should_detect = (
            S.running
            and S.gemini_ok
            and not S.arm_busy
            and (now - last_detect) >= cfg.detect_interval
        )

        if should_detect:
            last_detect = now
            jpeg_bytes = camera.jpeg(frame)
            detections = vision.detect(jpeg_bytes)
            with S.lock:
                S.detections = detections
                S.ripe_count = sum(1 for d in detections if d["is_ripe"])
                S.unripe_count = sum(1 for d in detections if not d["is_ripe"])
            if detections:
                log(f"Detected {len(detections)} ({S.ripe_count} ripe)", "DET")
        else:
            with S.lock:
                detections = S.detections

        # Draw overlay
        display = draw_overlay(frame, detections)
        _, jpeg = cv2.imencode(".jpg", display,
                               [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        with S.lock:
            S.last_jpeg = jpeg.tobytes()

        # ─── MODE LOGIC ───

        if S.mode == "autonomous" and S.running and not S.arm_busy:
            ripe = [d for d in detections if d["is_ripe"] and d["can_grip"]]

            if ripe:
                # Stop rover, pick
                if S.rover_moving:
                    rover.stop()
                    time.sleep(cfg.auto_stop_settle)

                det = ripe[0]
                rx, ry, rz = det["robot_coords"]
                if 5 <= rx <= 35:
                    log(f"AUTO pick ({rx:.1f},{ry:.1f})")
                    arm.pick(rx, ry, rz, det["diameter"])
                    time.sleep(cfg.auto_resume_delay)
                    # Resume driving after pick
                    rover.forward(cfg.auto_rover_speed)
                else:
                    log(f"AUTO out of reach x={rx:.1f}")
                    if not S.rover_moving:
                        rover.forward(cfg.auto_rover_speed)
            else:
                # Nothing detected, keep driving
                if not S.rover_moving:
                    rover.forward(cfg.auto_rover_speed)

        elif S.mode == "semi-auto" and S.running and not S.arm_busy:
            # Rover is user-controlled; arm auto-picks when rover is stopped
            ripe = [d for d in detections if d["is_ripe"] and d["can_grip"]]

            if ripe and not S.rover_moving:
                det = ripe[0]
                rx, ry, rz = det["robot_coords"]
                if 5 <= rx <= 35:
                    log(f"SEMI pick ({rx:.1f},{ry:.1f})")
                    arm.pick(rx, ry, rz, det["diameter"])

        # manual mode: everything controlled from UI, no auto-logic

        time.sleep(0.03)


def draw_overlay(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw detection boxes and status bar on frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    for det in detections:
        box = det.get("box_norm")
        if isinstance(box, list) and len(box) == 4:
            ymin, xmin, ymax, xmax = [clamp_norm(v) for v in box]
            x1 = int(min(xmin, xmax) / 1000 * w)
            y1 = int(min(ymin, ymax) / 1000 * h)
            x2 = int(max(xmin, xmax) / 1000 * w)
            y2 = int(max(ymin, ymax) / 1000 * h)

            col = (0, 255, 0) if det["is_ripe"] else (0, 165, 255)
            if not det["can_grip"]:
                col = (0, 100, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)

            txt = f"{det['label']} {det['diameter']:.1f}cm"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), col, -1)
            cv2.putText(out, txt, (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Robot coords
            rx, ry, rz = det["robot_coords"]
            cv2.putText(out, f"({rx:.0f},{ry:.0f})", (x1, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Pick point marker
        pn = det.get("pick_norm")
        if isinstance(pn, list) and len(pn) == 2:
            px = int(clamp_norm(pn[1]) / 1000 * w)
            py = int(clamp_norm(pn[0]) / 1000 * h)
            cv2.drawMarker(out, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 14, 2)

    # Status bar
    cv2.rectangle(out, (0, 0), (w, 32), (20, 20, 20), -1)
    mode_col = {"manual": (150, 150, 150), "semi-auto": (0, 165, 255),
                "autonomous": (0, 255, 0)}.get(S.mode, (255, 255, 255))
    status = f"{S.mode.upper()} | Ripe:{S.ripe_count} | Picks:{S.picks_ok}/{S.picks_try}"
    cv2.putText(out, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_col, 2)

    esp_col = (0, 255, 0) if S.esp_ok else (0, 0, 255)
    cv2.circle(out, (w - 20, 16), 6, esp_col, -1)

    if S.scene_desc:
        cv2.rectangle(out, (0, 32), (w, 50), (10, 10, 40), -1)
        cv2.putText(out, S.scene_desc[:70], (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 255), 1)

    return out


# ═══════════════════════════════════════════════════════════════
# FLASK WEB SERVER
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

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
.logo{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;color:var(--accent);letter-spacing:-0.5px}
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
  background:transparent;color:var(--muted);cursor:pointer;
  font-size:13px;font-weight:600;font-family:'DM Sans',sans-serif;transition:all .15s;
}
.mode-btn:hover{border-color:var(--accent);color:var(--accent)}
.mode-btn.active{border-color:var(--accent);color:var(--accent);background:rgba(34,211,238,.08)}
.mode-btn.active[data-mode="autonomous"]{border-color:var(--green);color:var(--green);background:rgba(16,185,129,.08)}

.ctrl-row{display:flex;gap:8px;margin-bottom:16px}
.btn{
  flex:1;padding:12px;border:none;border-radius:8px;font-size:14px;font-weight:600;
  cursor:pointer;font-family:'DM Sans',sans-serif;transition:background .15s;
}
.btn-go{background:var(--green);color:#000}
.btn-go:hover{background:#34d399}
.btn-stop{background:var(--red);color:#fff}
.btn-stop:hover{background:#f87171}
.btn-home{background:var(--surface);color:var(--text)}
.btn-home:hover{background:#334155}

.rover-pad{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:16px}
.r-btn{
  aspect-ratio:1;border:none;border-radius:8px;background:var(--surface);
  color:var(--text);font-size:20px;cursor:pointer;transition:background .1s;
  display:flex;align-items:center;justify-content:center;
}
.r-btn:active{background:var(--accent);color:#000}
.r-btn.r-stop{background:var(--red)}
.r-btn.empty{background:transparent;cursor:default}

.slider-wrap{margin-bottom:12px}
.slider-top{display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;font-family:'JetBrains Mono',monospace}
.slider-top .val{color:var(--accent)}
input[type=range]{
  width:100%;height:4px;border-radius:2px;background:var(--surface);
  outline:none;-webkit-appearance:none;
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:16px;height:16px;border-radius:50%;
  background:var(--accent);cursor:pointer;
}

.stats{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.stat{background:var(--surface);padding:14px;border-radius:8px;text-align:center}
.stat-val{font-size:28px;font-weight:700;color:var(--accent);font-family:'JetBrains Mono',monospace}
.stat-lbl{font-size:11px;color:var(--muted);margin-top:2px;text-transform:uppercase;letter-spacing:.5px}

.arm-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.arm-btn{
  padding:10px;border:1px solid var(--border);border-radius:8px;
  background:transparent;color:var(--text);cursor:pointer;font-size:13px;
  font-family:'DM Sans',sans-serif;transition:all .15s;
}
.arm-btn:hover{border-color:var(--accent);background:rgba(34,211,238,.06)}

.ik-panel{background:var(--bg);border-radius:8px;padding:12px;margin-top:12px}
.ik-panel label{font-size:11px;color:var(--muted);display:block;margin-bottom:2px}
.ik-input{
  width:100%;padding:7px 8px;border:1px solid var(--border);border-radius:6px;
  background:var(--card);color:var(--text);font-size:13px;margin-bottom:8px;
  font-family:'JetBrains Mono',monospace;
}
.btn-save{
  width:100%;padding:9px;border:none;border-radius:6px;
  background:var(--green);color:#000;font-weight:600;cursor:pointer;
  font-family:'DM Sans',sans-serif;
}

.log-box{
  height:160px;overflow-y:auto;background:var(--bg);border-radius:8px;padding:8px;
  font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.5;
}
.log-line{border-bottom:1px solid var(--border);padding:2px 0}

::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:var(--card)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
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
        <div class="slider-top"><span>Speed</span><span class="val" id="spdVal">120</span></div>
        <input type="range" min="50" max="255" value="120" oninput="setSpeed(this.value)">
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
        <div class="stat"><div class="stat-val" id="sUnripe">0</div><div class="stat-lbl">Unripe</div></div>
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
          <input type="checkbox" id="ikIB" checked> Invert Base
        </label>
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
  const o=data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{};
  return fetch('/api/'+ep,o).then(r=>r.json());
}

function setMode(m,el){
  api('mode',{mode:m});
  document.querySelectorAll('.mode-btn').forEach(b=>b.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('modeLabel').textContent=m.toUpperCase();
}

function toggleSys(){
  running=!running;
  api('system',{running});
  const b=document.getElementById('startBtn');
  b.textContent=running?'STOP':'START';
  b.className='btn '+(running?'btn-stop':'btn-go');
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
    invert_base:document.getElementById('ikIB').checked
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
    document.getElementById('armSt').style.borderColor=d.arm_busy?'var(--amber)':'var(--border)';
  }).catch(()=>{});
}

function pollLogs(){
  api('logs').then(d=>{
    const box=document.getElementById('logBox');
    box.innerHTML=d.logs.map(l=>'<div class="log-line">'+l+'</div>').join('');
    box.scrollTop=box.scrollHeight;
  }).catch(()=>{});
}

// Load IK
api('ik').then(d=>{
  document.getElementById('ikBC').value=d.base_center;
  document.getElementById('ikSO').value=d.shoulder_offset;
  document.getElementById('ikSM').value=d.shoulder_mult;
  document.getElementById('ikIB').checked=d.invert_base;
}).catch(()=>{});

setInterval(poll,600);
setInterval(pollLogs,1200);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with S.lock:
                f = S.last_jpeg
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.033)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with S.lock:
        return jsonify({
            "mode": S.mode, "running": S.running,
            "esp_ok": S.esp_ok, "gemini_ok": S.gemini_ok,
            "rover_moving": S.rover_moving, "rover_dir": S.rover_dir,
            "arm_busy": S.arm_busy, "arm_pos": S.arm_pos,
            "ripe_count": S.ripe_count, "unripe_count": S.unripe_count,
            "picks_ok": S.picks_ok, "picks_try": S.picks_try,
        })


@app.route("/api/mode", methods=["POST"])
def api_mode():
    d = request.json
    new_mode = d.get("mode", "manual")
    # Stop rover when switching away from autonomous
    if S.mode == "autonomous" and new_mode != "autonomous":
        rover.stop()
    S.mode = new_mode
    log(f"Mode: {S.mode}")
    return jsonify({"ok": True})


@app.route("/api/system", methods=["POST"])
def api_system():
    d = request.json
    S.running = d.get("running", False)
    log(f"System {'ON' if S.running else 'OFF'}")
    if not S.running:
        rover.stop()
    return jsonify({"ok": True})


@app.route("/api/speed", methods=["POST"])
def api_speed():
    cfg.rover_speed = request.json.get("speed", 120)
    return jsonify({"ok": True})


@app.route("/api/ik", methods=["GET", "POST"])
def api_ik():
    if request.method == "POST":
        d = request.json
        cfg.ik_base_center = d.get("base_center", cfg.ik_base_center)
        cfg.ik_shoulder_offset = d.get("shoulder_offset", cfg.ik_shoulder_offset)
        cfg.ik_shoulder_mult = d.get("shoulder_mult", cfg.ik_shoulder_mult)
        cfg.invert_base = d.get("invert_base", cfg.invert_base)
        cfg.save()
        log(f"IK saved: bc={cfg.ik_base_center} so={cfg.ik_shoulder_offset} sm={cfg.ik_shoulder_mult} inv={cfg.invert_base}")
        return jsonify({"ok": True})
    return jsonify({
        "base_center": cfg.ik_base_center,
        "shoulder_offset": cfg.ik_shoulder_offset,
        "shoulder_mult": cfg.ik_shoulder_mult,
        "invert_base": cfg.invert_base,
    })


@app.route("/api/rover/<cmd>")
def api_rover(cmd):
    {"f": rover.forward, "b": rover.backward, "l": rover.left,
     "r": rover.right, "s": rover.stop}.get(cmd, rover.stop)()
    return jsonify({"ok": True})


@app.route("/api/arm/home")
def api_arm_home():
    threading.Thread(target=arm.home, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/arm/open")
def api_arm_open():
    arm.open_gripper()
    return jsonify({"ok": True})


@app.route("/api/arm/close")
def api_arm_close():
    arm.close_gripper()
    return jsonify({"ok": True})


@app.route("/api/arm/twist")
def api_arm_twist():
    threading.Thread(target=arm.twist, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/arm/pick")
def api_arm_pick():
    def do_pick():
        with S.lock:
            ripe = [d for d in S.detections if d["is_ripe"] and d["can_grip"]]
        if ripe:
            d = ripe[0]
            rx, ry, rz = d["robot_coords"]
            arm.pick(rx, ry, rz, d["diameter"])
        else:
            log("Nothing to pick", "WARN")
    threading.Thread(target=do_pick, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/arm/servo", methods=["POST"])
def api_arm_servo():
    d = request.json
    servo = d.get("servo", "")
    angle = d.get("angle", 0)
    if servo in arm.CODE:
        arm._send(servo, int(angle))
    return jsonify({"ok": True})


@app.route("/api/logs")
def api_logs():
    with S.lock:
        return jsonify({"logs": S.logs[-50:]})


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    log("AgroPick starting")

    if not esp.connect():
        log("ESP32 required — exiting", "ERR")
        return

    if not camera.start():
        log("Camera required — exiting", "ERR")
        return

    vision.init()  # Gemini is optional — works in manual mode without it

    if S.esp_ok:
        arm.home()

    # Start control loop thread
    threading.Thread(target=control_loop, daemon=True).start()

    log(f"Ready — http://0.0.0.0:{cfg.port}", "OK")
    app.run(host="0.0.0.0", port=cfg.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
