#!/usr/bin/env python3
"""
AgroPick low-latency Pi controller.

Architecture:
  - ESP32 is a pure actuator over USB serial.
  - Raspberry Pi owns vision, control logic, Flask UI, and arm sequencing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
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
CALIBRATION_FILE = "calibration.json"
SUPPORTED_MODES = {"manual", "semi-auto", "autonomous"}
LIVE_ACTIONS = {"continue_forward", "stop_and_confirm", "hold"}


@dataclass
class Config:
    camera_width: int = 640
    camera_height: int = 480
    stream_quality: int = 60
    control_loop_delay_seconds: float = 0.08
    model_input_max_dim: int = 448
    live_frame_interval_seconds: float = 0.2
    live_query_interval_seconds: float = 0.9
    live_reconnect_seconds: float = 2.0
    snapshot_fallback_interval_seconds: float = 1.5
    live_stale_timeout_seconds: float = 3.0
    pi_camera_output_format: str = "RGB888"
    camera_color_order: str = "BGR"
    camera_swap_red_blue: bool = False
    rotation_deg: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    roi_norm: list[int] = field(default_factory=lambda: [0, 0, 1000, 1000])

    live_model_name: str = "gemini-2.5-flash-native-audio-preview-12-2025"
    snapshot_model_name: str = "gemini-2.5-flash"
    confirm_model_name: str = "gemini-robotics-er-1.5-preview"
    confirm_fallback_model_name: str = "gemini-2.5-flash"
    genai_api_version: str = "v1alpha"

    esp32_port: str = "/dev/ttyUSB0"
    esp32_baudrate: int = 115200
    serial_ack_timeout_seconds: float = 1.0
    serial_done_timeout_seconds: float = 10.0
    home_command_timeout_seconds: float = 12.0

    rover_speed: int = 150
    rover_turn_speed: int = 130
    auto_forward_speed: int = 120

    stable_frames_required: int = 3
    autonomous_clear_frames_required: int = 2
    pick_box_width_threshold_norm: int = 150
    min_detection_width_norm: int = 40

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
    home_shoulder: int = 150
    home_wrist: int = 160
    home_gripper: int = 40
    home_rotgripper: int = 130

    ik_base_center: int = 110
    ik_shoulder_offset: float = 110.0
    ik_shoulder_multiplier: float = 1.2
    invert_base: bool = False
    approach_height_cm: float = 8.0
    grab_height_cm: float = 3.0
    lift_height_cm: float = 10.0
    twist_angle: int = 15
    twist_cycles: int = 3
    twist_delay: float = 0.2
    gripper_max_capacity_cm: float = 8.0

    web_port: int = 5000

    def save(self, filename: str = CONFIG_FILE) -> None:
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def load(cls, filename: str = CONFIG_FILE) -> "Config":
        if not os.path.exists(filename):
            return cls()
        try:
            with open(filename, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            return cls()

        allowed = {item.name for item in fields(cls)}
        data = {key: value for key, value in raw.items() if key in allowed}
        return cls(**data)


@dataclass
class Detection:
    label: str
    box_2d: list[int]
    pick_point: list[int]
    estimated_diameter_cm: float
    is_harvestable: bool
    reachable: bool
    source: str
    ts: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionTrace:
    scene_summary: str = "Vision idle"
    selected_target: Optional[str] = None
    action: str = "hold"
    reason: str = "System idle"
    live_latency_ms: Optional[int] = None
    confirm_latency_ms: Optional[int] = None
    stale: bool = True
    candidate_visible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationPoint:
    img_x_norm: int
    img_y_norm: int
    arm_x_cm: float
    arm_y_cm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Calibration:
    matrix: list[list[float]] = field(default_factory=list)
    points: list[CalibrationPoint] = field(default_factory=list)
    active: bool = False

    @property
    def ready(self) -> bool:
        return len(self.matrix) == 2 and all(len(row) == 3 for row in self.matrix)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "active": self.active,
            "matrix": self.matrix,
            "points": [point.to_dict() for point in self.points],
        }


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.mode = "manual"
        self.system_running = False
        self.esp32_connected = False
        self.vision_connected = False
        self.live_session_state = "disconnected"
        self.rover_moving = False
        self.rover_direction = "stopped"
        self.arm_busy = False
        self.pick_inflight = False
        self.arm_positions: dict[str, int] = {}
        self.detections: list[Detection] = []
        self.decision_trace = DecisionTrace()
        self.last_frame: Optional[bytes] = None
        self.picks_attempted = 0
        self.picks_successful = 0
        self.ripe_count = 0
        self.unripe_count = 0
        self.logs: list[str] = []
        self.max_logs = 160
        self.serial_debug_lines: list[str] = []
        self.max_serial_debug = 80
        self.serial_rtt_ms: Optional[int] = None
        self.last_confirm_latency_ms: Optional[int] = None
        self.fault_state: Optional[str] = None
        self.last_raw_live = ""
        self.last_raw_confirm = ""
        self.last_decision_at = 0.0
        self.candidate_signature: Optional[str] = None
        self.candidate_streak = 0
        self.awaiting_autonomous_clearance = False
        self.autonomous_clear_streak = 0


config = Config.load()
state = AppState()


def log(message: str, level: str = "INFO") -> None:
    entry = f"[{time.strftime('%H:%M:%S')}] [{level}] {message}"
    print(entry, flush=True)
    with state.lock:
        state.logs.append(entry)
        if len(state.logs) > state.max_logs:
            state.logs.pop(0)


def add_serial_debug(line: str) -> None:
    with state.lock:
        state.serial_debug_lines.append(line)
        if len(state.serial_debug_lines) > state.max_serial_debug:
            state.serial_debug_lines.pop(0)


def set_fault(message: str) -> None:
    with state.lock:
        state.fault_state = message
    log(message, "ERROR")


def clear_fault() -> None:
    with state.lock:
        state.fault_state = None


def clamp_norm(value: Any) -> int:
    try:
        return int(np.clip(int(round(float(value))), 0, 1000))
    except Exception:
        return 0


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
        except Exception:
            continue

    raise ValueError(f"No JSON found in payload: {cleaned[:160]}")


def sanitize_roi_norm(roi_norm: list[int]) -> list[int]:
    if not isinstance(roi_norm, list) or len(roi_norm) != 4:
        return [0, 0, 1000, 1000]
    ymin, xmin, ymax, xmax = [clamp_norm(value) for value in roi_norm]
    ymin, ymax = sorted((ymin, ymax))
    xmin, xmax = sorted((xmin, xmax))
    if ymax == ymin:
        ymax = min(1000, ymin + 1)
    if xmax == xmin:
        xmax = min(1000, xmin + 1)
    return [ymin, xmin, ymax, xmax]


def remap_norm_point_from_roi(point: list[int], roi_norm: list[int]) -> list[int]:
    ymin, xmin, ymax, xmax = sanitize_roi_norm(roi_norm)
    point_y = clamp_norm(point[0])
    point_x = clamp_norm(point[1])
    full_y = ymin + int((ymax - ymin) * point_y / 1000.0)
    full_x = xmin + int((xmax - xmin) * point_x / 1000.0)
    return [clamp_norm(full_y), clamp_norm(full_x)]


def remap_norm_box_from_roi(box: list[int], roi_norm: list[int]) -> list[int]:
    ymin, xmin, ymax, xmax = sanitize_roi_norm(roi_norm)
    box_ymin, box_xmin, box_ymax, box_xmax = [clamp_norm(value) for value in box]
    full_ymin = ymin + int((ymax - ymin) * box_ymin / 1000.0)
    full_xmin = xmin + int((xmax - xmin) * box_xmin / 1000.0)
    full_ymax = ymin + int((ymax - ymin) * box_ymax / 1000.0)
    full_xmax = xmin + int((xmax - xmin) * box_xmax / 1000.0)
    return [clamp_norm(full_ymin), clamp_norm(full_xmin), clamp_norm(full_ymax), clamp_norm(full_xmax)]


def normalize_detection(raw: dict[str, Any], source: str, roi_norm: Optional[list[int]] = None) -> Optional[Detection]:
    if not isinstance(raw, dict):
        return None

    label = str(raw.get("label") or raw.get("name") or "unknown").strip() or "unknown"
    box = raw.get("box_2d") or raw.get("box2d")
    point = raw.get("pick_point") or raw.get("point")

    if isinstance(box, list) and len(box) == 4:
        box = [clamp_norm(value) for value in box]
        box = [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
        if roi_norm is not None:
            box = remap_norm_box_from_roi(box, roi_norm)
    else:
        box = [0, 0, 0, 0]

    if isinstance(point, list) and len(point) == 2:
        point = [clamp_norm(point[0]), clamp_norm(point[1])]
        if roi_norm is not None:
            point = remap_norm_point_from_roi(point, roi_norm)
    elif box[2] > box[0] and box[3] > box[1]:
        point = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
    else:
        point = [500, 500]

    try:
        diameter = float(raw.get("estimated_diameter_cm", raw.get("diameter_cm", 0)) or 0)
    except Exception:
        diameter = 0.0

    is_harvestable = bool(raw.get("is_harvestable", raw.get("is_ripe", True)))
    reachable = bool(raw.get("reachable", True))

    return Detection(
        label=label,
        box_2d=box,
        pick_point=point,
        estimated_diameter_cm=diameter,
        is_harvestable=is_harvestable,
        reachable=reachable,
        source=source,
        ts=time.time(),
    )


def detection_width_norm(detection: Detection) -> int:
    return abs(clamp_norm(detection.box_2d[3]) - clamp_norm(detection.box_2d[1]))


def detection_signature(detection: Detection) -> str:
    center_y, center_x = detection.pick_point
    return (
        f"{detection.label}:"
        f"{round(center_x / 80)}:"
        f"{round(center_y / 80)}:"
        f"{round(detection_width_norm(detection) / 40)}"
    )


def choose_best_detection(
    detections: list[Detection],
    preferred_label: Optional[str],
) -> Optional[Detection]:
    if not detections:
        return None
    candidates = [item for item in detections if item.is_harvestable and item.reachable]
    if not candidates:
        candidates = [item for item in detections if item.reachable] or detections

    if preferred_label:
        for item in candidates:
            if item.label == preferred_label:
                return item

    candidates = [
        item for item in candidates if detection_width_norm(item) >= config.min_detection_width_norm
    ] or candidates
    candidates.sort(key=detection_width_norm, reverse=True)
    return candidates[0]


class CalibrationManager:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data = Calibration()
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            return

        points = []
        for item in raw.get("points", []):
            try:
                points.append(
                    CalibrationPoint(
                        img_x_norm=clamp_norm(item["img_x_norm"]),
                        img_y_norm=clamp_norm(item["img_y_norm"]),
                        arm_x_cm=float(item["arm_x_cm"]),
                        arm_y_cm=float(item["arm_y_cm"]),
                    )
                )
            except Exception:
                continue

        matrix = raw.get("matrix", [])
        if isinstance(matrix, list) and len(matrix) == 2:
            self.data.matrix = [[float(value) for value in row[:3]] for row in matrix]
        self.data.points = points
        self.data.active = False

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.data.to_dict(), handle, indent=2)

    def start(self) -> dict[str, Any]:
        with self.lock:
            self.data.active = True
            self.data.points = []
            self.data.matrix = []
            self.save()
            return self.data.to_dict()

    def add_point(self, img_x_norm: int, img_y_norm: int, arm_x_cm: float, arm_y_cm: float) -> dict[str, Any]:
        point = CalibrationPoint(
            img_x_norm=clamp_norm(img_x_norm),
            img_y_norm=clamp_norm(img_y_norm),
            arm_x_cm=float(arm_x_cm),
            arm_y_cm=float(arm_y_cm),
        )
        with self.lock:
            self.data.points.append(point)
            self.save()
            return self.data.to_dict()

    def finish(self) -> dict[str, Any]:
        with self.lock:
            if len(self.data.points) < 3:
                raise ValueError("Need at least 3 calibration points")

            src = np.float32([[point.img_x_norm, point.img_y_norm] for point in self.data.points])
            dst = np.float32([[point.arm_x_cm, point.arm_y_cm] for point in self.data.points])

            if len(self.data.points) == 3:
                matrix = cv2.getAffineTransform(src, dst)
            else:
                matrix, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=20.0)
                if matrix is None:
                    raise ValueError("Calibration fit failed")

            self.data.matrix = matrix.tolist()
            self.data.active = False
            self.save()
            return self.data.to_dict()

    def map_point(self, img_x_norm: int, img_y_norm: int) -> tuple[float, float]:
        with self.lock:
            if not self.data.ready:
                raise ValueError("Calibration not ready")
            matrix = np.array(self.data.matrix, dtype=np.float32)
        vector = np.array([float(img_x_norm), float(img_y_norm), 1.0], dtype=np.float32)
        result = matrix @ vector
        return float(result[0]), float(result[1])

    def as_dict(self) -> dict[str, Any]:
        with self.lock:
            return self.data.to_dict()


calibration_manager = CalibrationManager(CALIBRATION_FILE)


@dataclass
class PendingCommand:
    command_id: str
    frame: str
    wait_mode: str
    created_at: float = field(default_factory=time.time)
    ack_event: threading.Event = field(default_factory=threading.Event)
    complete_event: threading.Event = field(default_factory=threading.Event)
    ack_latency_ms: Optional[int] = None
    payload: list[str] = field(default_factory=list)
    error: Optional[str] = None


class SerialProtocolClient:
    def __init__(self) -> None:
        self.serial: Optional[serial.Serial] = None
        self.connected = False
        self._stop_event = threading.Event()
        self._write_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._pending: dict[str, PendingCommand] = {}
        self._outbound: queue.Queue[Optional[PendingCommand]] = queue.Queue()
        self._next_id = 1
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True)

    def start(self) -> None:
        if not self._reader_thread.is_alive():
            self._reader_thread.start()
        if not self._writer_thread.is_alive():
            self._writer_thread.start()
        if not self._reconnect_thread.is_alive():
            self._reconnect_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._outbound.put(None)
        if self.serial is not None:
            with contextlib.suppress(Exception):
                self.serial.close()

    def _find_port(self) -> Optional[str]:
        if os.path.exists(config.esp32_port):
            return config.esp32_port
        for port in serial.tools.list_ports.comports():
            description = port.description.lower()
            if any(token in description for token in ("cp210", "ch340", "usb", "uart")):
                return port.device
            if "/dev/ttyUSB" in port.device or "/dev/ttyACM" in port.device:
                return port.device
        return None

    def _reconnect_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self.connected:
                self._connect_once()
            time.sleep(2.0)

    def _connect_once(self) -> None:
        port = self._find_port()
        if not port:
            with state.lock:
                state.esp32_connected = False
            return

        try:
            serial_port = serial.Serial(
                port=port,
                baudrate=config.esp32_baudrate,
                timeout=0.1,
                write_timeout=0.1,
                dsrdtr=False,
                rtscts=False,
                xonxoff=False,
            )
            with contextlib.suppress(Exception):
                serial_port.setDTR(False)
                serial_port.setRTS(False)
            time.sleep(0.2)
            serial_port.reset_input_buffer()
            serial_port.reset_output_buffer()
            self.serial = serial_port
            self.connected = True
            with state.lock:
                state.esp32_connected = True
            clear_fault()
            log(f"ESP32 connected on {port}", "SUCCESS")
            threading.Thread(target=self._bootstrap_after_connect, daemon=True).start()
        except Exception as exc:
            with state.lock:
                state.esp32_connected = False
            self.connected = False
            self.serial = None
            log(f"ESP32 connect failed: {exc}", "WARN")

    def _handle_disconnect(self, reason: str) -> None:
        if self.serial is not None:
            with contextlib.suppress(Exception):
                self.serial.close()
        self.serial = None
        self.connected = False
        with state.lock:
            state.esp32_connected = False
        if reason:
            set_fault(reason)

        with self._pending_lock:
            pending_items = list(self._pending.values())
            self._pending.clear()
        for pending in pending_items:
            pending.error = "DISCONNECTED"
            pending.ack_event.set()
            pending.complete_event.set()

    def _bootstrap_after_connect(self) -> None:
        try:
            self.ping()
            busy = self.get_busy()
            positions = self.get_pos()
            with state.lock:
                state.arm_busy = busy
                state.arm_positions = positions
        except Exception as exc:
            log(f"Bootstrap sync failed: {exc}", "WARN")

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.serial is None:
                time.sleep(0.1)
                continue
            try:
                raw = self.serial.readline()
            except Exception as exc:
                self._handle_disconnect(f"Serial read failed: {exc}")
                continue
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            add_serial_debug(line)
            self._handle_line(line)

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            pending = self._outbound.get()
            if pending is None:
                return
            if self.serial is None or not self.connected:
                pending.error = "DISCONNECTED"
                pending.ack_event.set()
                pending.complete_event.set()
                with self._pending_lock:
                    self._pending.pop(pending.command_id, None)
                continue
            try:
                with self._write_lock:
                    self.serial.write(f"{pending.frame}\n".encode("ascii", errors="ignore"))
                    self.serial.flush()
            except Exception as exc:
                self._handle_disconnect(f"Serial write failed: {exc}")
                pending.error = "DISCONNECTED"
                pending.ack_event.set()
                pending.complete_event.set()
                with self._pending_lock:
                    self._pending.pop(pending.command_id, None)

    def _handle_line(self, line: str) -> None:
        tokens = line.split("|")
        if not tokens:
            return

        if tokens[0] == "READY":
            with state.lock:
                state.esp32_connected = True
            log(f"ESP32 ready: {line}", "INFO")
            return

        if len(tokens) < 2:
            log(f"Unparsed serial line: {line}", "WARN")
            return

        command_id = tokens[1]
        with self._pending_lock:
            pending = self._pending.get(command_id)

        if pending is None:
            log(f"Unexpected serial line: {line}", "WARN")
            return

        kind = tokens[0]
        if kind == "ACK":
            pending.ack_latency_ms = int((time.time() - pending.created_at) * 1000)
            pending.ack_event.set()
            with state.lock:
                state.serial_rtt_ms = pending.ack_latency_ms
            return

        if kind == "DONE":
            pending.payload = tokens[2:]
            pending.complete_event.set()
            return

        if kind == "VAL":
            pending.payload = tokens[2:]
            pending.complete_event.set()
            return

        if kind == "ERR":
            pending.error = tokens[2] if len(tokens) > 2 else "UNKNOWN"
            pending.ack_event.set()
            pending.complete_event.set()
            return

    def _next_command_id(self) -> str:
        with self._command_lock:
            command_id = str(self._next_id)
            self._next_id += 1
        return command_id

    def _submit(self, opcode: str, *parts: Any, wait_mode: str, timeout: float) -> PendingCommand:
        if not self.connected or self.serial is None:
            raise RuntimeError("ESP32 not connected")

        command_id = self._next_command_id()
        frame = "|".join([opcode, command_id] + [str(part) for part in parts if part is not None])
        pending = PendingCommand(command_id=command_id, frame=frame, wait_mode=wait_mode)
        with self._pending_lock:
            self._pending[command_id] = pending
        self._outbound.put(pending)

        if not pending.ack_event.wait(config.serial_ack_timeout_seconds):
            with self._pending_lock:
                self._pending.pop(command_id, None)
            raise TimeoutError(f"ACK timeout for {frame}")
        if pending.error:
            with self._pending_lock:
                self._pending.pop(command_id, None)
            raise RuntimeError(pending.error)

        if wait_mode == "ack":
            with self._pending_lock:
                self._pending.pop(command_id, None)
            return pending

        if not pending.complete_event.wait(timeout):
            with self._pending_lock:
                self._pending.pop(command_id, None)
            raise TimeoutError(f"Completion timeout for {frame}")
        if pending.error:
            with self._pending_lock:
                self._pending.pop(command_id, None)
            raise RuntimeError(pending.error)

        with self._pending_lock:
            self._pending.pop(command_id, None)
        return pending

    def ping(self) -> None:
        self._submit("PING", wait_mode="ack", timeout=config.serial_ack_timeout_seconds)

    def estop(self) -> None:
        self._submit("ESTOP", wait_mode="ack", timeout=config.serial_ack_timeout_seconds)

    def rover(self, direction: str, speed: Optional[int] = None) -> None:
        timeout = config.serial_done_timeout_seconds
        if direction == "STOP":
            self._submit("ROVER", direction, wait_mode="done", timeout=timeout)
        else:
            self._submit("ROVER", direction, int(speed or config.rover_speed), wait_mode="done", timeout=timeout)

    def pose(self, base: int, shoulder: int, wrist: int) -> None:
        self._submit("POSE", int(base), int(shoulder), int(wrist), wait_mode="done", timeout=config.serial_done_timeout_seconds)

    def gripper(self, angle: int) -> None:
        self._submit("GRIPPER", int(angle), wait_mode="done", timeout=config.serial_done_timeout_seconds)

    def rotgripper(self, angle: int) -> None:
        self._submit("ROTGRIPPER", int(angle), wait_mode="done", timeout=config.serial_done_timeout_seconds)

    def home(self) -> None:
        self._submit("HOME", wait_mode="done", timeout=config.home_command_timeout_seconds)

    def get_busy(self) -> bool:
        response = self._submit("GET", "BUSY", wait_mode="val", timeout=config.serial_ack_timeout_seconds)
        return len(response.payload) >= 2 and response.payload[0] == "BUSY" and response.payload[1] == "1"

    def get_pos(self) -> dict[str, int]:
        response = self._submit("GET", "POS", wait_mode="val", timeout=config.serial_ack_timeout_seconds)
        if len(response.payload) < 6 or response.payload[0] != "POS":
            raise RuntimeError("Invalid POS response")
        return {
            "base": int(response.payload[1]),
            "shoulder": int(response.payload[2]),
            "wrist": int(response.payload[3]),
            "gripper": int(response.payload[4]),
            "rotgripper": int(response.payload[5]),
        }

    def get_encoders(self) -> tuple[int, int]:
        response = self._submit("GET", "ENC", wait_mode="val", timeout=config.serial_ack_timeout_seconds)
        if len(response.payload) < 3 or response.payload[0] != "ENC":
            raise RuntimeError("Invalid ENC response")
        return int(response.payload[1]), int(response.payload[2])


serial_client = SerialProtocolClient()


class RoverController:
    def __init__(self) -> None:
        self._last_command = ""
        self._last_command_at = 0.0

    def _update_state(self, direction: str, moving: bool) -> None:
        with state.lock:
            state.rover_direction = direction
            state.rover_moving = moving

    def _send(self, direction: str, speed: Optional[int] = None) -> None:
        encoded = f"{direction}:{speed}"
        now = time.time()
        if encoded == self._last_command and now - self._last_command_at < 0.25:
            return
        self._last_command = encoded
        self._last_command_at = now

        if direction == "FWD":
            serial_client.rover("FWD", speed or config.rover_speed)
            self._update_state("forward", True)
        elif direction == "REV":
            serial_client.rover("REV", speed or config.rover_speed)
            self._update_state("backward", True)
        elif direction == "LEFT":
            serial_client.rover("LEFT", speed or config.rover_turn_speed)
            self._update_state("left", True)
        elif direction == "RIGHT":
            serial_client.rover("RIGHT", speed or config.rover_turn_speed)
            self._update_state("right", True)
        elif direction == "STOP":
            serial_client.rover("STOP")
            self._update_state("stopped", False)
        else:
            raise RuntimeError(f"Invalid rover direction: {direction}")

    def forward(self, speed: Optional[int] = None) -> None:
        self._send("FWD", speed)

    def backward(self, speed: Optional[int] = None) -> None:
        self._send("REV", speed)

    def left(self, speed: Optional[int] = None) -> None:
        self._send("LEFT", speed)

    def right(self, speed: Optional[int] = None) -> None:
        self._send("RIGHT", speed)

    def stop(self) -> None:
        self._send("STOP")

    def estop(self) -> None:
        serial_client.estop()
        self._update_state("stopped", False)


rover = RoverController()


class ArmController:
    def __init__(self) -> None:
        with state.lock:
            state.arm_positions = {
                "base": config.home_base,
                "shoulder": config.home_shoulder,
                "wrist": config.home_wrist,
                "gripper": config.home_gripper,
                "rotgripper": config.home_rotgripper,
            }

    def _set_busy(self, busy: bool) -> None:
        with state.lock:
            state.arm_busy = busy

    def _set_positions(self, **kwargs: int) -> None:
        with state.lock:
            state.arm_positions.update(kwargs)

    def solve_pose(self, x_cm: float, y_cm: float, z_cm: float) -> tuple[int, int, int]:
        x_cm = float(np.clip(x_cm, 5, 35))
        y_cm = float(np.clip(y_cm, -20, 20))
        z_cm = float(np.clip(z_cm, 0, 25))
        angle_deg = np.degrees(np.arctan2(y_cm, x_cm))
        if config.invert_base:
            base = int(config.ik_base_center - angle_deg)
        else:
            base = int(config.ik_base_center + angle_deg)
        base = int(np.clip(base, config.base_min, config.base_max))
        shoulder = int(config.ik_shoulder_offset + z_cm * config.ik_shoulder_multiplier)
        shoulder = int(np.clip(shoulder, config.shoulder_min, config.shoulder_max))
        wrist = config.home_wrist
        return base, shoulder, wrist

    def move_to_xyz(self, x_cm: float, y_cm: float, z_cm: float) -> None:
        base, shoulder, wrist = self.solve_pose(x_cm, y_cm, z_cm)
        serial_client.pose(base, shoulder, wrist)
        self._set_positions(base=base, shoulder=shoulder, wrist=wrist)

    def home(self) -> None:
        self._set_busy(True)
        try:
            serial_client.home()
            self._set_positions(
                base=config.home_base,
                shoulder=config.home_shoulder,
                wrist=config.home_wrist,
                gripper=config.home_gripper,
                rotgripper=config.home_rotgripper,
            )
        finally:
            self._set_busy(False)

    def open_gripper(self) -> None:
        self._set_busy(True)
        try:
            serial_client.gripper(config.gripper_min)
            self._set_positions(gripper=config.gripper_min)
        finally:
            self._set_busy(False)

    def close_gripper(self, diameter_cm: float = 5.0) -> None:
        angle_range = config.gripper_max - config.gripper_min
        angle = config.gripper_min + int(
            (1 - min(diameter_cm, config.gripper_max_capacity_cm) / config.gripper_max_capacity_cm)
            * angle_range
        )
        angle = int(np.clip(angle, config.gripper_min, config.gripper_max))
        self._set_busy(True)
        try:
            serial_client.gripper(angle)
            self._set_positions(gripper=angle)
        finally:
            self._set_busy(False)

    def twist(self) -> None:
        self._set_busy(True)
        try:
            home_rot = config.home_rotgripper
            for angle in (home_rot + config.twist_angle, home_rot - config.twist_angle):
                serial_client.rotgripper(angle)
                self._set_positions(rotgripper=angle)
                time.sleep(config.twist_delay)
            serial_client.rotgripper(home_rot)
            self._set_positions(rotgripper=home_rot)
        finally:
            self._set_busy(False)

    def pick(self, x_cm: float, y_cm: float, diameter_cm: float) -> bool:
        if diameter_cm > config.gripper_max_capacity_cm:
            log(f"Target too large to grip ({diameter_cm:.1f}cm)", "WARN")
            return False

        with state.lock:
            state.picks_attempted += 1
        self._set_busy(True)
        try:
            log(f"Pick sequence at ({x_cm:.1f}, {y_cm:.1f}) diameter {diameter_cm:.1f}cm", "PICK")
            self.move_to_xyz(x_cm, y_cm, config.grab_height_cm + config.approach_height_cm)
            serial_client.gripper(config.gripper_min)
            self._set_positions(gripper=config.gripper_min)
            self.move_to_xyz(x_cm, y_cm, config.grab_height_cm)

            angle_range = config.gripper_max - config.gripper_min
            grip_angle = config.gripper_min + int(
                (1 - min(diameter_cm, config.gripper_max_capacity_cm) / config.gripper_max_capacity_cm)
                * angle_range
            )
            grip_angle = int(np.clip(grip_angle, config.gripper_min, config.gripper_max))
            serial_client.gripper(grip_angle)
            self._set_positions(gripper=grip_angle)

            home_rot = config.home_rotgripper
            for angle in (home_rot + config.twist_angle, home_rot - config.twist_angle, home_rot):
                serial_client.rotgripper(angle)
                self._set_positions(rotgripper=angle)
                time.sleep(config.twist_delay)

            self.move_to_xyz(x_cm, y_cm, config.grab_height_cm + config.lift_height_cm)
            serial_client.home()
            serial_client.gripper(config.gripper_min)
            self._set_positions(
                base=config.home_base,
                shoulder=config.home_shoulder,
                wrist=config.home_wrist,
                gripper=config.gripper_min,
                rotgripper=config.home_rotgripper,
            )
            with state.lock:
                state.picks_successful += 1
            log("Pick complete", "SUCCESS")
            return True
        except Exception as exc:
            set_fault(f"Arm command failed: {exc}")
            return False
        finally:
            self._set_busy(False)


arm = ArmController()


LIVE_SYSTEM_INSTRUCTION = """
You are AgroPick's live agricultural vision monitor.
Watch the camera feed and return only compact JSON.

Return only this JSON object:
{
  "scene_summary": "short scene summary",
  "selected_target": "label or null",
  "action": "continue_forward|stop_and_confirm|hold",
  "reason": "short reason",
  "candidate_visible": true,
  "candidate": {
    "label": "tomato",
    "box_2d": [ymin, xmin, ymax, xmax],
    "pick_point": [y, x],
    "estimated_diameter_cm": 5.0,
    "is_harvestable": true,
    "reachable": true
  }
}

Rules:
- Detect fruits and vegetables only.
- Coordinates are normalized 0 to 1000.
- candidate may be null if nothing useful is visible.
- If unsure, use action "hold".
- Never return markdown fences.
""".strip()

LIVE_QUERY_PROMPT = "Return the current AgroPick JSON for the latest camera view only."

SNAPSHOT_DECISION_PROMPT = """
Analyze this agricultural camera image and return only JSON:
{
  "scene_summary": "short scene summary",
  "selected_target": "label or null",
  "action": "continue_forward|stop_and_confirm|hold",
  "reason": "short reason",
  "candidate_visible": true,
  "candidate": {
    "label": "tomato",
    "box_2d": [ymin, xmin, ymax, xmax],
    "pick_point": [y, x],
    "estimated_diameter_cm": 5.0,
    "is_harvestable": true,
    "reachable": true
  }
}

Rules:
- Fruits and vegetables only.
- Coordinates normalized 0 to 1000.
- If nothing useful is visible, use candidate_visible false, candidate null, selected_target null, and action continue_forward.
- Never return markdown fences.
""".strip()

CONFIRM_PROMPT_TEMPLATE = """
You are confirming a pick target for AgroPick.
Preferred target: {preferred_label}

Return only this JSON object:
{{
  "detections": [
    {{
      "label": "tomato",
      "box_2d": [ymin, xmin, ymax, xmax],
      "pick_point": [y, x],
      "estimated_diameter_cm": 5.3,
      "is_harvestable": true,
      "reachable": true
    }}
  ],
  "selected_target": "tomato",
  "reason": "closest reachable ripe tomato"
}}

Rules:
- Fruits and vegetables only.
- Coordinates normalized 0 to 1000.
- If the preferred target is not visible, return an empty detections list and selected_target null.
- Never return markdown fences.
""".strip()


def update_detection_state(detections: list[Detection], trace: DecisionTrace, raw_text: str, source: str) -> None:
    with state.lock:
        state.detections = detections
        state.decision_trace = trace
        state.ripe_count = sum(1 for item in detections if item.is_harvestable)
        state.unripe_count = max(0, len(detections) - state.ripe_count)
        state.vision_connected = True
        state.last_decision_at = time.time()
        if source == "live":
            state.last_raw_live = raw_text
        else:
            state.last_raw_confirm = raw_text


def build_decision_from_payload(
    payload: dict[str, Any],
    raw_text: str,
    latency_ms: Optional[int],
    source: str,
) -> tuple[DecisionTrace, list[Detection]]:
    scene_summary = payload.get("scene_summary", "")
    if not isinstance(scene_summary, str):
        scene_summary = ""

    selected_target = payload.get("selected_target")
    if selected_target is not None and not isinstance(selected_target, str):
        selected_target = None

    action = payload.get("action", "hold")
    if action not in LIVE_ACTIONS:
        action = "hold"

    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = ""

    candidate_visible = bool(payload.get("candidate_visible", False))
    detections: list[Detection] = []

    candidate = payload.get("candidate")
    if isinstance(candidate, dict):
        detection = normalize_detection(candidate, source)
        if detection is not None:
            detections.append(detection)

    if not detections:
        for item in payload.get("detections", []):
            detection = normalize_detection(item, source)
            if detection is not None:
                detections.append(detection)

    if detections and selected_target is None:
        selected_target = detections[0].label
        candidate_visible = True

    trace = DecisionTrace(
        scene_summary=scene_summary or "No candidate visible",
        selected_target=selected_target,
        action=action,
        reason=reason or "No explicit reason returned",
        live_latency_ms=latency_ms if source != "confirm" else state.decision_trace.live_latency_ms,
        confirm_latency_ms=latency_ms if source == "confirm" else state.last_confirm_latency_ms,
        stale=False,
        candidate_visible=candidate_visible or bool(detections),
    )
    return trace, detections


class PiCamera:
    def __init__(self) -> None:
        self.camera: Optional[Picamera2] = None

    def start(self) -> bool:
        try:
            self.camera = Picamera2()
            camera_config = self.camera.create_preview_configuration(
                main={
                    "size": (config.camera_width, config.camera_height),
                    "format": config.pi_camera_output_format,
                }
            )
            self.camera.configure(camera_config)
            self.camera.start()
            time.sleep(1.0)
            if controls is not None:
                with contextlib.suppress(Exception):
                    self.camera.set_controls(
                        {
                            "AfMode": controls.AfModeEnum.Continuous,
                            "AfSpeed": controls.AfSpeedEnum.Fast,
                        }
                    )
            log("Pi camera ready", "SUCCESS")
            return True
        except Exception as exc:
            set_fault(f"Camera failed: {exc}")
            return False

    def _apply_color_fix(self, frame: np.ndarray) -> np.ndarray:
        color_order = config.camera_color_order.upper()
        if frame.ndim == 3 and frame.shape[2] == 4:
            if color_order == "RGB":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3 and color_order == "RGB":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if config.camera_swap_red_blue:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _apply_geometry(self, frame: np.ndarray) -> np.ndarray:
        rotation = config.rotation_deg % 360
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if config.flip_vertical:
            frame = cv2.flip(frame, 0)
        return frame

    def capture_frame(self) -> np.ndarray:
        if self.camera is None:
            raise RuntimeError("Camera not started")
        frame = self.camera.capture_array()
        frame = self._apply_color_fix(frame)
        frame = self._apply_geometry(frame)
        return frame

    def frame_to_jpeg(self, frame: np.ndarray, quality: Optional[int] = None) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, int(quality if quality is not None else config.stream_quality)],
        )
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return encoded.tobytes()

    def build_model_input(self, frame: np.ndarray, use_roi: bool) -> tuple[bytes, list[int]]:
        roi_norm = sanitize_roi_norm(config.roi_norm if use_roi else [0, 0, 1000, 1000])
        if use_roi:
            height, width = frame.shape[:2]
            ymin, xmin, ymax, xmax = roi_norm
            x1 = int(xmin / 1000.0 * width)
            y1 = int(ymin / 1000.0 * height)
            x2 = max(x1 + 1, int(xmax / 1000.0 * width))
            y2 = max(y1 + 1, int(ymax / 1000.0 * height))
            model_frame = frame[y1:y2, x1:x2]
        else:
            model_frame = frame

        height, width = model_frame.shape[:2]
        max_dim = max(height, width)
        if max_dim > config.model_input_max_dim:
            scale = config.model_input_max_dim / max_dim
            model_frame = cv2.resize(model_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return self.frame_to_jpeg(model_frame), roi_norm

    def stop(self) -> None:
        if self.camera is not None:
            self.camera.stop()


camera = PiCamera()


class LiveDecisionEngine:
    def __init__(self, apply_payload_callback) -> None:
        self._apply_payload_callback = apply_payload_callback
        self._stop_event = threading.Event()
        self._restart_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        self._latest_frame_at = 0.0
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._last_query_sent_at = 0.0

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._restart_event.set()

    def restart(self) -> None:
        self._restart_event.set()

    def submit_frame(self, frame_bytes: bytes) -> None:
        with self._frame_lock:
            self._latest_frame = frame_bytes
            self._latest_frame_at = time.time()

    def _pop_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
            return frame

    def _run_thread(self) -> None:
        asyncio.run(self._supervisor())

    async def _supervisor(self) -> None:
        if not GOOGLE_API_KEY:
            return

        while not self._stop_event.is_set():
            self._set_state("connecting", False)
            try:
                client = genai.Client(
                    api_key=GOOGLE_API_KEY,
                    http_options=types.HttpOptions(api_version=config.genai_api_version),
                )
                live_config = {
                    "response_modalities": ["TEXT"],
                    "system_instruction": LIVE_SYSTEM_INSTRUCTION,
                }
                async with client.aio.live.connect(model=config.live_model_name, config=live_config) as session:
                    self._set_state("connected", True)
                    receiver = asyncio.create_task(self._receive_loop(session))
                    last_frame_sent = 0.0
                    last_query_sent = 0.0

                    while not self._stop_event.is_set() and not self._restart_event.is_set():
                        frame = self._pop_frame()
                        now = time.time()
                        if frame is not None and now - last_frame_sent >= config.live_frame_interval_seconds:
                            await session.send_realtime_input(
                                video=types.Blob(data=frame, mime_type="image/jpeg")
                            )
                            last_frame_sent = now

                        if now - last_query_sent >= config.live_query_interval_seconds:
                            self._last_query_sent_at = now
                            await session.send_client_content(
                                turns=LIVE_QUERY_PROMPT,
                                turn_complete=True,
                            )
                            last_query_sent = now

                        await asyncio.sleep(0.05)

                    self._restart_event.clear()
                    receiver.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await receiver
            except Exception as exc:
                self._set_state("error", False)
                log(f"Live vision unavailable: {exc}", "WARN")
                await asyncio.sleep(config.live_reconnect_seconds)

    def _set_state(self, session_state: str, connected: bool) -> None:
        with state.lock:
            state.live_session_state = session_state
            if session_state != "fallback_snapshot":
                state.vision_connected = connected

    async def _receive_loop(self, session) -> None:
        buffer = ""
        async for message in session.receive():
            text = self._extract_text(message)
            if text:
                buffer += text

            if self._is_turn_complete(message):
                if buffer.strip():
                    latency_ms = int((time.time() - self._last_query_sent_at) * 1000) if self._last_query_sent_at else None
                    self._apply_payload_callback(buffer, latency_ms, source="live")
                buffer = ""

    def _extract_text(self, message: Any) -> str:
        chunks: list[str] = []
        text = getattr(message, "text", None)
        if isinstance(text, str) and text:
            chunks.append(text)

        server_content = getattr(message, "server_content", None)
        if server_content is not None:
            model_turn = getattr(server_content, "model_turn", None)
            if model_turn is not None:
                for part in getattr(model_turn, "parts", []) or []:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text:
                        chunks.append(part_text)
        return "".join(chunks)

    def _is_turn_complete(self, message: Any) -> bool:
        if bool(getattr(message, "turn_complete", False)):
            return True
        server_content = getattr(message, "server_content", None)
        if server_content is not None:
            if bool(getattr(server_content, "turn_complete", False)):
                return True
            if bool(getattr(server_content, "generation_complete", False)):
                return True
        return False


class VisionCoordinator:
    def __init__(self) -> None:
        self._snapshot_lock = threading.Lock()
        self._confirm_lock = threading.Lock()
        self._sync_client: Optional[genai.Client] = None
        self.live_engine: Optional[LiveDecisionEngine] = None
        if GOOGLE_API_KEY:
            self._sync_client = genai.Client(
                api_key=GOOGLE_API_KEY,
                http_options=types.HttpOptions(api_version=config.genai_api_version),
            )
            self.live_engine = LiveDecisionEngine(self.apply_live_text)

    def start(self) -> None:
        if self.live_engine is not None:
            self.live_engine.start()

    def restart(self) -> None:
        if self.live_engine is not None:
            self.live_engine.restart()

    def stop(self) -> None:
        if self.live_engine is not None:
            self.live_engine.stop()

    def apply_live_text(self, raw_text: str, latency_ms: Optional[int], source: str) -> None:
        try:
            payload = parse_json_safe(raw_text)
            if not isinstance(payload, dict):
                raise ValueError("Live payload was not a JSON object")
            trace, detections = build_decision_from_payload(payload, raw_text, latency_ms, source)
            update_detection_state(detections, trace, raw_text, source)
        except Exception as exc:
            log(f"Vision JSON parse failed: {exc}", "WARN")
            with state.lock:
                if source == "live":
                    state.last_raw_live = raw_text
                else:
                    state.last_raw_confirm = raw_text

    def submit_live_frame(self, frame: np.ndarray) -> None:
        if self.live_engine is None:
            return
        try:
            frame_bytes, _ = camera.build_model_input(frame, use_roi=False)
            self.live_engine.submit_frame(frame_bytes)
        except Exception as exc:
            log(f"Failed to prepare live frame: {exc}", "WARN")

    def snapshot_decision_async(self, frame: np.ndarray, force: bool = False) -> None:
        if self._sync_client is None:
            return
        if not force and not self._snapshot_lock.acquire(blocking=False):
            return

        if force:
            acquired = self._snapshot_lock.acquire(blocking=False)
            if not acquired:
                return

        def worker() -> None:
            try:
                with state.lock:
                    state.live_session_state = "fallback_snapshot"
                start = time.time()
                image_bytes, _ = camera.build_model_input(frame, use_roi=False)
                raw_text, _ = self._call_json_model(
                    [config.snapshot_model_name],
                    SNAPSHOT_DECISION_PROMPT,
                    image_bytes,
                )
                latency_ms = int((time.time() - start) * 1000)
                self.apply_live_text(raw_text, latency_ms, source="live")
                with state.lock:
                    state.live_session_state = "fallback_snapshot"
                    state.vision_connected = True
            except Exception as exc:
                log(f"Snapshot fallback failed: {exc}", "WARN")
            finally:
                self._snapshot_lock.release()

        threading.Thread(target=worker, daemon=True).start()

    def confirm_target(self, frame: np.ndarray, preferred_label: Optional[str]) -> tuple[list[Detection], Optional[Detection], str]:
        if self._sync_client is None:
            raise RuntimeError("Vision client not initialized")
        with self._confirm_lock:
            image_bytes, roi_norm = camera.build_model_input(frame, use_roi=True)
            prompt = CONFIRM_PROMPT_TEMPLATE.format(preferred_label=preferred_label or "any harvestable fruit or vegetable")
            start = time.time()
            raw_text, _ = self._call_json_model(
                [config.confirm_model_name, config.confirm_fallback_model_name],
                prompt,
                image_bytes,
            )
            latency_ms = int((time.time() - start) * 1000)
            with state.lock:
                state.last_confirm_latency_ms = latency_ms
                state.last_raw_confirm = raw_text

            parsed = parse_json_safe(raw_text)
            if not isinstance(parsed, dict):
                raise ValueError("Confirm payload was not a JSON object")

            detections: list[Detection] = []
            for item in parsed.get("detections", []):
                detection = normalize_detection(item, "confirm", roi_norm=roi_norm)
                if detection is not None:
                    detections.append(detection)

            selected_target = parsed.get("selected_target")
            if selected_target is not None and not isinstance(selected_target, str):
                selected_target = None
            reason = parsed.get("reason", "")
            if not isinstance(reason, str):
                reason = ""

            best = choose_best_detection(detections, selected_target or preferred_label)
            return detections, best, reason

    def _call_json_model(self, models: list[str], prompt: str, image_bytes: bytes) -> tuple[str, str]:
        last_error: Optional[Exception] = None
        for model in models:
            try:
                response = self._sync_client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )
                return (response.text or "").strip(), model
            except Exception as exc:
                last_error = exc
                log(f"Model {model} failed: {exc}", "WARN")
        raise RuntimeError(str(last_error) if last_error else "No model available")


vision = VisionCoordinator()


class PickCoordinator:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def start_pick(self, detection: Optional[Detection], resume_forward: bool) -> bool:
        if detection is None:
            return False
        if not self._lock.acquire(blocking=False):
            return False

        def worker() -> None:
            with state.lock:
                state.pick_inflight = True
            try:
                rover.stop()
                time.sleep(0.1)
                frame = camera.capture_frame()
                detections, confirmed, reason = vision.confirm_target(frame, detection.label)
                if confirmed is None or not confirmed.is_harvestable or not confirmed.reachable:
                    log(f"Confirm failed: {reason or 'no reachable target'}", "WARN")
                    update_detection_state(
                        detections,
                        DecisionTrace(
                            scene_summary="Confirm failed",
                            selected_target=None,
                            action="hold",
                            reason=reason or "No reachable target confirmed",
                            live_latency_ms=state.decision_trace.live_latency_ms,
                            confirm_latency_ms=state.last_confirm_latency_ms,
                            stale=False,
                            candidate_visible=False,
                        ),
                        state.last_raw_confirm,
                        "confirm",
                    )
                    if resume_forward:
                        with state.lock:
                            state.awaiting_autonomous_clearance = True
                            state.autonomous_clear_streak = 0
                    return

                arm_x_cm, arm_y_cm = calibration_manager.map_point(
                    confirmed.pick_point[1],
                    confirmed.pick_point[0],
                )
                log(
                    f"Confirmed target {confirmed.label} -> arm ({arm_x_cm:.1f}, {arm_y_cm:.1f})",
                    "PICK",
                )
                success = arm.pick(arm_x_cm, arm_y_cm, confirmed.estimated_diameter_cm or 5.0)
                if success:
                    update_detection_state(
                        [],
                        DecisionTrace(
                            scene_summary="Last pick completed",
                            selected_target=None,
                            action="hold",
                            reason="Pick completed successfully",
                            live_latency_ms=state.decision_trace.live_latency_ms,
                            confirm_latency_ms=state.last_confirm_latency_ms,
                            stale=False,
                            candidate_visible=False,
                        ),
                        state.last_raw_confirm,
                        "confirm",
                    )
                    if resume_forward:
                        with state.lock:
                            auto_allowed = state.system_running and state.mode == "autonomous" and not state.fault_state
                        if auto_allowed:
                            rover.forward(config.auto_forward_speed)
                elif resume_forward:
                    with state.lock:
                        state.awaiting_autonomous_clearance = True
                        state.autonomous_clear_streak = 0
            except Exception as exc:
                set_fault(f"Pick workflow failed: {exc}")
                rover.stop()
            finally:
                with state.lock:
                    state.pick_inflight = False
                self._lock.release()

        threading.Thread(target=worker, daemon=True).start()
        return True


pick_coordinator = PickCoordinator()


def overlay_detections(frame: np.ndarray) -> np.ndarray:
    display = frame.copy()
    height, width = display.shape[:2]
    with state.lock:
        detections = list(state.detections)
        decision = state.decision_trace
        mode = state.mode
        fault_state = state.fault_state
        calibration = calibration_manager.as_dict()

    for detection in detections:
        ymin, xmin, ymax, xmax = [clamp_norm(value) for value in detection.box_2d]
        x1 = int(xmin / 1000.0 * width)
        y1 = int(ymin / 1000.0 * height)
        x2 = int(xmax / 1000.0 * width)
        y2 = int(ymax / 1000.0 * height)
        color = (0, 255, 0) if detection.is_harvestable else (0, 165, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.label} {detection.estimated_diameter_cm:.1f}cm"
        cv2.putText(
            display,
            label[:36],
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
        )
        px = int(detection.pick_point[1] / 1000.0 * width)
        py = int(detection.pick_point[0] / 1000.0 * height)
        cv2.drawMarker(display, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)

    if calibration["active"]:
        for index, point in enumerate(calibration["points"], start=1):
            px = int(point["img_x_norm"] / 1000.0 * width)
            py = int(point["img_y_norm"] / 1000.0 * height)
            cv2.circle(display, (px, py), 6, (255, 200, 0), -1)
            cv2.putText(
                display,
                str(index),
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 200, 0),
                1,
            )

    banner_height = 90 if fault_state else 62
    cv2.rectangle(display, (0, 0), (width, banner_height), (16, 18, 28), cv2.FILLED)
    with state.lock:
        top_line = (
            f"{mode.upper()} | Rover:{state.rover_direction.upper()} | "
            f"Arm:{'BUSY' if state.arm_busy else 'READY'} | "
            f"Live:{state.live_session_state} | Picks:{state.picks_successful}/{state.picks_attempted}"
        )
    cv2.putText(display, top_line, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2)
    cv2.putText(
        display,
        decision.scene_summary[:96],
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (180, 220, 255),
        1,
    )
    if fault_state:
        cv2.putText(
            display,
            fault_state[:96],
            (10, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 90, 255),
            2,
        )
    return display


def update_candidate_tracking() -> Optional[Detection]:
    with state.lock:
        detections = list(state.detections)
        preferred = state.decision_trace.selected_target
        stale = state.decision_trace.stale

    best = choose_best_detection(detections, preferred)
    signature = detection_signature(best) if best and not stale and best.is_harvestable and best.reachable else None

    with state.lock:
        if signature is None:
            state.candidate_signature = None
            state.candidate_streak = 0
        elif signature == state.candidate_signature:
            state.candidate_streak += 1
        else:
            state.candidate_signature = signature
            state.candidate_streak = 1

        if state.awaiting_autonomous_clearance:
            if (
                state.mode == "autonomous"
                and state.system_running
                and not state.decision_trace.stale
                and state.decision_trace.action == "continue_forward"
                and best is None
            ):
                state.autonomous_clear_streak += 1
            else:
                state.autonomous_clear_streak = 0

    return best


def maybe_mark_vision_stale() -> None:
    with state.lock:
        if not state.system_running:
            return
        if state.last_decision_at <= 0:
            return
        if time.time() - state.last_decision_at <= config.live_stale_timeout_seconds:
            return
        state.decision_trace.stale = True
        state.vision_connected = False
        if state.decision_trace.reason == "System idle":
            state.decision_trace.reason = "Vision output timed out"


def handle_mode_logic(best_detection: Optional[Detection]) -> None:
    with state.lock:
        mode = state.mode
        system_running = state.system_running
        fault_state = state.fault_state
        pick_inflight = state.pick_inflight
        arm_busy = state.arm_busy
        trace = state.decision_trace
        candidate_streak = state.candidate_streak
        awaiting_clearance = state.awaiting_autonomous_clearance
        autonomous_clear_streak = state.autonomous_clear_streak

    calibration_ready = calibration_manager.as_dict()["ready"]

    if not system_running:
        return

    if fault_state:
        if mode == "autonomous":
            rover.stop()
        return

    if pick_inflight or arm_busy:
        rover.stop()
        return

    if mode == "manual":
        return

    if mode == "semi-auto":
        if best_detection is None:
            return
        if not calibration_ready:
            return
        if candidate_streak >= config.stable_frames_required and detection_width_norm(best_detection) >= config.pick_box_width_threshold_norm:
            pick_coordinator.start_pick(best_detection, resume_forward=False)
        return

    if mode == "autonomous":
        if not calibration_ready:
            rover.stop()
            return
        if trace.stale:
            rover.stop()
            return
        if awaiting_clearance:
            rover.stop()
            if autonomous_clear_streak >= config.autonomous_clear_frames_required:
                with state.lock:
                    state.awaiting_autonomous_clearance = False
                    state.autonomous_clear_streak = 0
                rover.forward(config.auto_forward_speed)
            return
        if (
            best_detection is not None
            and candidate_streak >= config.stable_frames_required
            and detection_width_norm(best_detection) >= config.pick_box_width_threshold_norm
        ):
            pick_coordinator.start_pick(best_detection, resume_forward=True)
            return
        rover.forward(config.auto_forward_speed)


def control_loop() -> None:
    log("Control loop started", "INFO")
    last_fallback_snapshot = 0.0

    while True:
        if camera.camera is None:
            time.sleep(0.2)
            continue

        try:
            frame = camera.capture_frame()
        except Exception as exc:
            set_fault(f"Camera capture failed: {exc}")
            time.sleep(0.5)
            continue

        with state.lock:
            system_running = state.system_running

        if system_running:
            vision.submit_live_frame(frame)
        maybe_mark_vision_stale()

        with state.lock:
            allow_fallback = (
                state.system_running
                and not state.pick_inflight
                and not state.arm_busy
                and state.live_session_state != "connected"
            )
        if allow_fallback and time.time() - last_fallback_snapshot >= config.snapshot_fallback_interval_seconds:
            last_fallback_snapshot = time.time()
            vision.snapshot_decision_async(frame)

        best_detection = update_candidate_tracking()
        try:
            handle_mode_logic(best_detection)
        except Exception as exc:
            set_fault(f"Control loop action failed: {exc}")
            with contextlib.suppress(Exception):
                rover.stop()

        display = overlay_detections(frame)
        try:
            encoded = camera.frame_to_jpeg(display)
            with state.lock:
                state.last_frame = encoded
        except Exception as exc:
            log(f"Preview encode failed: {exc}", "WARN")

        time.sleep(config.control_loop_delay_seconds)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AgroPick</title>
  <style>
    :root {
      --bg: #09111f;
      --panel: rgba(17, 27, 48, 0.92);
      --line: #20324f;
      --ink: #eff5ff;
      --muted: #8ea4c8;
      --cyan: #39d4ff;
      --green: #56ef8e;
      --amber: #ffbe55;
      --red: #ff617a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(57, 212, 255, 0.12), transparent 30%),
        radial-gradient(circle at bottom right, rgba(86, 239, 142, 0.10), transparent 28%),
        var(--bg);
    }
    .shell {
      max-width: 1550px;
      margin: 0 auto;
      padding: 18px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      margin-bottom: 16px;
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
    .chip, .fault-banner, .safety-banner {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      background: rgba(13, 22, 40, 0.9);
      color: var(--muted);
    }
    .fault-banner, .safety-banner {
      border-radius: 14px;
      margin-bottom: 14px;
    }
    .fault-banner { color: #ffd8df; border-color: rgba(255, 97, 122, 0.5); background: rgba(97, 18, 30, 0.42); }
    .safety-banner { color: #ffe9c0; border-color: rgba(255, 190, 85, 0.45); background: rgba(78, 51, 6, 0.38); display: none; }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 420px;
      gap: 16px;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      padding: 18px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
    }
    .video {
      width: 100%;
      display: block;
      border-radius: 14px;
      background: #000;
      border: 1px solid #1d2a45;
    }
    .grid {
      display: grid;
      gap: 10px;
    }
    .modes { grid-template-columns: repeat(3, 1fr); margin-top: 14px; }
    .actions, .arm-grid { grid-template-columns: repeat(2, 1fr); margin-top: 14px; }
    button, input, select {
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #13213c;
      color: var(--ink);
      padding: 12px 14px;
      font: inherit;
    }
    button { cursor: pointer; font-weight: 600; }
    button:hover { border-color: var(--cyan); }
    button.mode.active { color: var(--cyan); border-color: var(--cyan); }
    button.mode.auto.active { color: var(--green); border-color: var(--green); }
    button.start { background: linear-gradient(135deg, #1d6f4a, #21ae68); }
    button.stop { background: linear-gradient(135deg, #7f2d44, #c34766); }
    button.emergency { background: linear-gradient(135deg, #7a1728, #db233e); }
    .rover-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-top: 14px;
    }
    .rover-grid button {
      min-height: 68px;
      font-size: 24px;
    }
    .empty { border: none; background: transparent; cursor: default; }
    .section-title {
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--cyan);
      font-size: 13px;
      margin-bottom: 10px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin-bottom: 14px;
    }
    .stat {
      border: 1px solid #192743;
      border-radius: 14px;
      background: rgba(10, 18, 34, 0.78);
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
    .trace {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }
    .trace-item {
      border: 1px solid #192743;
      border-radius: 14px;
      background: rgba(10, 18, 34, 0.78);
      padding: 12px;
    }
    .trace-item .label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .trace-item .value {
      margin-top: 4px;
      font-size: 14px;
      line-height: 1.45;
    }
    .logs, pre {
      max-height: 230px;
      overflow: auto;
      background: #07101f;
      border: 1px solid #162540;
      border-radius: 14px;
      padding: 12px;
      font-family: ui-monospace, SFMono-Regular, monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .log-line { padding: 2px 0; }
    .cal-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin-top: 12px;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    details {
      margin-top: 14px;
      border: 1px solid #1a2946;
      border-radius: 14px;
      padding: 12px;
      background: rgba(8, 16, 30, 0.65);
    }
    summary {
      cursor: pointer;
      color: var(--cyan);
      font-weight: 600;
    }
    @media (max-width: 1160px) {
      .layout { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <div class="title">AgroPick</div>
      <div class="chips">
        <div class="chip" id="esp32Chip">ESP32: offline</div>
        <div class="chip" id="visionChip">Vision: disconnected</div>
        <div class="chip" id="modeChip">Mode: manual</div>
        <div class="chip" id="roverChip">Rover: stopped</div>
        <div class="chip" id="armChip">Arm: ready</div>
      </div>
    </div>

    <div class="fault-banner" id="faultBanner" style="display:none;"></div>
    <div class="safety-banner" id="safetyBanner">No obstacle sensing active. Operator supervision required.</div>

    <div class="layout">
      <div class="panel">
        <div class="section-title">Live Feed</div>
        <img class="video" id="videoFeed" src="/video_feed" alt="AgroPick live feed">

        <div class="grid modes">
          <button class="mode active" id="mode-manual" onclick="setMode('manual', this)">Manual</button>
          <button class="mode" id="mode-semi-auto" onclick="setMode('semi-auto', this)">Semi-Auto</button>
          <button class="mode auto" id="mode-autonomous" onclick="setMode('autonomous', this)">Autonomous</button>
        </div>

        <div class="grid actions">
          <button id="systemButton" class="start" onclick="toggleSystem()">Start System</button>
          <button onclick="restartVision()">Restart Vision</button>
          <button onclick="forceSnapshot()">Scan Now</button>
          <button class="emergency" onclick="estop()">E-Stop</button>
        </div>

        <div class="section-title" style="margin-top:18px;">Rover</div>
        <div class="rover-grid">
          <button class="empty"></button>
          <button onmousedown="rover('f')" onmouseup="rover('s')" ontouchstart="rover('f')" ontouchend="rover('s')">▲</button>
          <button class="empty"></button>
          <button onmousedown="rover('l')" onmouseup="rover('s')" ontouchstart="rover('l')" ontouchend="rover('s')">◀</button>
          <button onclick="rover('s')">■</button>
          <button onmousedown="rover('r')" onmouseup="rover('s')" ontouchstart="rover('r')" ontouchend="rover('s')">▶</button>
          <button class="empty"></button>
          <button onmousedown="rover('b')" onmouseup="rover('s')" ontouchstart="rover('b')" ontouchend="rover('s')">▼</button>
          <button class="empty"></button>
        </div>

        <div style="margin-top:14px;">
          <label for="speedSlider">Rover speed: <span id="speedValue">150</span></label>
          <input id="speedSlider" type="range" min="60" max="255" value="150" oninput="updateSpeed(this.value)">
        </div>

        <div class="section-title" style="margin-top:18px;">Calibration</div>
        <div class="grid actions">
          <button onclick="startCalibration()">Start Calibration</button>
          <button onclick="finishCalibration()">Finish Calibration</button>
        </div>
        <div class="cal-grid">
          <input id="armXInput" type="number" step="0.1" placeholder="Arm X (cm)">
          <input id="armYInput" type="number" step="0.1" placeholder="Arm Y (cm)">
        </div>
        <div class="grid actions">
          <button onclick="addCalibrationPoint()">Add Clicked Point</button>
          <button onclick="toggleColorSwap()">Swap Colors</button>
        </div>
        <div class="hint" id="calibrationHint">Click the video while calibration is active, then enter arm X/Y and save the point.</div>
      </div>

      <div class="panel">
        <div class="section-title">Status</div>
        <div class="stats">
          <div class="stat"><div class="value" id="harvestableValue">0</div><div class="label">Harvestable</div></div>
          <div class="stat"><div class="value" id="otherValue">0</div><div class="label">Other Detections</div></div>
          <div class="stat"><div class="value" id="pickSuccessValue">0</div><div class="label">Successful Picks</div></div>
          <div class="stat"><div class="value" id="pickAttemptValue">0</div><div class="label">Pick Attempts</div></div>
        </div>

        <div class="section-title">Decision Trace</div>
        <div class="trace">
          <div class="trace-item"><div class="label">Scene</div><div class="value" id="sceneSummary">Vision idle</div></div>
          <div class="trace-item"><div class="label">Target</div><div class="value" id="selectedTarget">None</div></div>
          <div class="trace-item"><div class="label">Action</div><div class="value" id="actionValue">hold</div></div>
          <div class="trace-item"><div class="label">Reason</div><div class="value" id="reasonValue">System idle</div></div>
          <div class="trace-item"><div class="label">Latency</div><div class="value" id="latencyValue">live: -, confirm: -, serial: -</div></div>
          <div class="trace-item"><div class="label">Calibration</div><div class="value" id="calibrationValue">Not ready</div></div>
        </div>

        <div class="section-title" style="margin-top:18px;">Arm</div>
        <div class="grid arm-grid">
          <button onclick="armAction('home')">Home</button>
          <button onclick="armAction('pick')">Pick Best</button>
          <button onclick="armAction('open')">Open</button>
          <button onclick="armAction('close')">Close</button>
          <button onclick="armAction('twist')">Twist</button>
          <button onclick="saveIK()">Save IK</button>
        </div>

        <div class="cal-grid" style="margin-top:12px;">
          <input id="ikBaseCenter" type="number" placeholder="Base center">
          <input id="ikShoulderOffset" type="number" step="0.1" placeholder="Shoulder offset">
          <input id="ikShoulderMultiplier" type="number" step="0.1" placeholder="Shoulder multiplier">
          <label style="display:flex;align-items:center;gap:8px;color:var(--muted);">
            <input id="ikInvertBase" type="checkbox">Invert base
          </label>
        </div>

        <details>
          <summary>Debug</summary>
          <div class="section-title" style="margin-top:12px;">Raw Live Payload</div>
          <pre id="rawLive"></pre>
          <div class="section-title" style="margin-top:12px;">Raw Confirm Payload</div>
          <pre id="rawConfirm"></pre>
          <div class="section-title" style="margin-top:12px;">Serial</div>
          <pre id="serialDebug"></pre>
          <div class="section-title" style="margin-top:12px;">Calibration Matrix</div>
          <pre id="calibrationMatrix"></pre>
        </details>

        <div class="section-title" style="margin-top:18px;">Logs</div>
        <div class="logs" id="logs"></div>
      </div>
    </div>
  </div>

  <script>
    let running = false;
    let clickedPoint = null;

    function api(endpoint, method = 'GET', data = null) {
      const options = {method};
      if (data !== null) {
        options.headers = {'Content-Type': 'application/json'};
        options.body = JSON.stringify(data);
      }
      return fetch('/api/' + endpoint, options).then((response) => response.json());
    }

    function clearModes() {
      document.querySelectorAll('.mode').forEach((button) => button.classList.remove('active'));
    }

    function setMode(mode, button) {
      api('mode', 'POST', {mode});
      clearModes();
      button.classList.add('active');
    }

    function toggleSystem() {
      running = !running;
      api('system', 'POST', {running});
    }

    function rover(cmd) {
      api('rover/' + cmd);
    }

    function armAction(name) {
      api('arm/' + name);
    }

    function updateSpeed(value) {
      document.getElementById('speedValue').textContent = value;
      api('speed', 'POST', {speed: parseInt(value, 10)});
    }

    function restartVision() {
      api('vision/restart', 'POST');
    }

    function forceSnapshot() {
      api('scan', 'POST', {force: true});
    }

    function estop() {
      api('estop', 'POST');
    }

    function toggleColorSwap() {
      api('camera/color');
    }

    function startCalibration() {
      clickedPoint = null;
      api('calibration/start', 'POST');
    }

    function finishCalibration() {
      api('calibration/finish', 'POST');
    }

    function addCalibrationPoint() {
      if (!clickedPoint) {
        alert('Click the video first.');
        return;
      }
      const armX = parseFloat(document.getElementById('armXInput').value);
      const armY = parseFloat(document.getElementById('armYInput').value);
      if (Number.isNaN(armX) || Number.isNaN(armY)) {
        alert('Enter arm X and Y.');
        return;
      }
      api('calibration/point', 'POST', {
        img_x_norm: clickedPoint.x,
        img_y_norm: clickedPoint.y,
        arm_x_cm: armX,
        arm_y_cm: armY
      }).then(() => {
        document.getElementById('armXInput').value = '';
        document.getElementById('armYInput').value = '';
      });
    }

    function saveIK() {
      api('ik', 'POST', {
        base_center: parseInt(document.getElementById('ikBaseCenter').value, 10),
        shoulder_offset: parseFloat(document.getElementById('ikShoulderOffset').value),
        shoulder_multiplier: parseFloat(document.getElementById('ikShoulderMultiplier').value),
        invert_base: document.getElementById('ikInvertBase').checked
      });
    }

    function updateStatus() {
      api('status').then((data) => {
        running = data.system_running;
        document.getElementById('esp32Chip').textContent = 'ESP32: ' + (data.esp32_connected ? 'online' : 'offline');
        document.getElementById('visionChip').textContent = 'Vision: ' + data.live_session_state;
        document.getElementById('modeChip').textContent = 'Mode: ' + data.mode;
        document.getElementById('roverChip').textContent = 'Rover: ' + data.rover_direction;
        document.getElementById('armChip').textContent = 'Arm: ' + (data.arm_busy || data.pick_inflight ? 'busy' : 'ready');

        const systemButton = document.getElementById('systemButton');
        systemButton.textContent = running ? 'Stop System' : 'Start System';
        systemButton.className = running ? 'stop' : 'start';

        document.getElementById('harvestableValue').textContent = data.ripe_count;
        document.getElementById('otherValue').textContent = data.unripe_count;
        document.getElementById('pickSuccessValue').textContent = data.picks_successful;
        document.getElementById('pickAttemptValue').textContent = data.picks_attempted;
        document.getElementById('speedValue').textContent = data.rover_speed;
        document.getElementById('speedSlider').value = data.rover_speed;

        document.getElementById('sceneSummary').textContent = data.decision_trace.scene_summary;
        document.getElementById('selectedTarget').textContent = data.decision_trace.selected_target || 'None';
        document.getElementById('actionValue').textContent = data.decision_trace.action + (data.decision_trace.stale ? ' (stale)' : '');
        document.getElementById('reasonValue').textContent = data.decision_trace.reason;
        document.getElementById('latencyValue').textContent =
          'live: ' + (data.decision_trace.live_latency_ms ?? '-') + ' ms, ' +
          'confirm: ' + (data.last_confirm_latency_ms ?? '-') + ' ms, ' +
          'serial: ' + (data.serial_rtt_ms ?? '-') + ' ms';
        document.getElementById('calibrationValue').textContent =
          data.calibration.ready ? ('Ready (' + data.calibration.points.length + ' pts)') : 'Not ready';

        document.getElementById('rawLive').textContent = data.last_raw_live || '(none)';
        document.getElementById('rawConfirm').textContent = data.last_raw_confirm || '(none)';
        document.getElementById('serialDebug').textContent = (data.serial_debug_lines || []).join('\\n');
        document.getElementById('calibrationMatrix').textContent =
          JSON.stringify(data.calibration.matrix || [], null, 2);

        const logs = document.getElementById('logs');
        logs.innerHTML = (data.logs || []).map((line) => '<div class="log-line">' + line + '</div>').join('');
        logs.scrollTop = logs.scrollHeight;

        clearModes();
        const active = document.getElementById('mode-' + data.mode);
        if (active) active.classList.add('active');

        const faultBanner = document.getElementById('faultBanner');
        if (data.fault_state) {
          faultBanner.textContent = data.fault_state;
          faultBanner.style.display = 'block';
        } else {
          faultBanner.style.display = 'none';
        }

        const safetyBanner = document.getElementById('safetyBanner');
        safetyBanner.style.display = data.mode === 'autonomous' ? 'block' : 'none';

        const hint = document.getElementById('calibrationHint');
        if (clickedPoint) {
          hint.textContent = 'Clicked point: x=' + clickedPoint.x + ', y=' + clickedPoint.y;
        } else {
          hint.textContent = data.calibration.active
            ? 'Calibration active. Click the video, then enter arm X/Y and save the point.'
            : 'Click the video while calibration is active, then enter arm X/Y and save the point.';
        }
      });
    }

    const video = document.getElementById('videoFeed');
    video.addEventListener('click', (event) => {
      const rect = video.getBoundingClientRect();
      const x = Math.round(((event.clientX - rect.left) / rect.width) * 1000);
      const y = Math.round(((event.clientY - rect.top) / rect.height) * 1000);
      clickedPoint = {x, y};
    });

    api('ik').then((data) => {
      document.getElementById('ikBaseCenter').value = data.base_center;
      document.getElementById('ikShoulderOffset').value = data.shoulder_offset;
      document.getElementById('ikShoulderMultiplier').value = data.shoulder_multiplier;
      document.getElementById('ikInvertBase').checked = data.invert_base;
    });
    updateStatus();
    setInterval(updateStatus, 750);
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
        payload = {
            "mode": state.mode,
            "system_running": state.system_running,
            "esp32_connected": state.esp32_connected,
            "vision_connected": state.vision_connected,
            "live_session_state": state.live_session_state,
            "rover_moving": state.rover_moving,
            "rover_direction": state.rover_direction,
            "rover_speed": config.rover_speed,
            "arm_busy": state.arm_busy,
            "pick_inflight": state.pick_inflight,
            "arm_positions": state.arm_positions,
            "detections": [item.to_dict() for item in state.detections],
            "decision_trace": state.decision_trace.to_dict(),
            "ripe_count": state.ripe_count,
            "unripe_count": state.unripe_count,
            "picks_attempted": state.picks_attempted,
            "picks_successful": state.picks_successful,
            "serial_rtt_ms": state.serial_rtt_ms,
            "last_confirm_latency_ms": state.last_confirm_latency_ms,
            "fault_state": state.fault_state,
            "last_raw_live": state.last_raw_live,
            "last_raw_confirm": state.last_raw_confirm,
            "serial_debug_lines": list(state.serial_debug_lines),
            "logs": state.logs[-70:],
            "calibration": calibration_manager.as_dict(),
        }
    return jsonify(payload)


@app.route("/api/logs")
def api_logs() -> Response:
    with state.lock:
        return jsonify({"logs": state.logs[-70:]})


@app.route("/api/mode", methods=["POST"])
def api_mode() -> Response:
    data = request.get_json(force=True) or {}
    mode = data.get("mode", "manual")
    if mode not in SUPPORTED_MODES:
        return jsonify({"success": False, "error": "invalid mode"}), 400
    with state.lock:
        state.mode = mode
        state.awaiting_autonomous_clearance = False
        state.autonomous_clear_streak = 0
    log(f"Mode -> {mode}", "INFO")
    if mode == "manual":
        with contextlib.suppress(Exception):
            rover.stop()
    return jsonify({"success": True})


@app.route("/api/system", methods=["POST"])
def api_system() -> Response:
    data = request.get_json(force=True) or {}
    running = bool(data.get("running", False))
    with state.lock:
        state.system_running = running
        if not running:
            state.awaiting_autonomous_clearance = False
            state.autonomous_clear_streak = 0
    log(f"System {'STARTED' if running else 'STOPPED'}", "INFO")
    if not running:
        with contextlib.suppress(Exception):
            rover.stop()
    return jsonify({"success": True})


@app.route("/api/speed", methods=["POST"])
def api_speed() -> Response:
    data = request.get_json(force=True) or {}
    speed = int(np.clip(int(data.get("speed", config.rover_speed)), 60, 255))
    config.rover_speed = speed
    config.rover_turn_speed = max(60, speed - 20)
    config.auto_forward_speed = max(70, speed - 20)
    config.save()
    log(f"Rover speed -> {speed}", "INFO")
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
    log("IK settings updated", "INFO")
    return jsonify({"success": True})


@app.route("/api/scan", methods=["POST"])
def api_scan() -> Response:
    try:
        frame = camera.capture_frame()
        vision.snapshot_decision_async(frame, force=True)
        return jsonify({"success": True})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/vision/restart", methods=["POST"])
def api_vision_restart() -> Response:
    vision.restart()
    with state.lock:
        state.live_session_state = "restarting"
    return jsonify({"success": True})


@app.route("/api/estop", methods=["POST"])
def api_estop() -> Response:
    try:
        rover.estop()
        set_fault("Emergency stop active")
        return jsonify({"success": True})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/rover/<cmd>")
def api_rover(cmd: str) -> Response:
    try:
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
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


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
    with state.lock:
        best = choose_best_detection(list(state.detections), state.decision_trace.selected_target)
    if best is None:
        return jsonify({"success": False, "error": "no current target"}), 400
    started = pick_coordinator.start_pick(best, resume_forward=False)
    return jsonify({"success": started})


@app.route("/api/camera/color")
def api_camera_color() -> Response:
    config.camera_swap_red_blue = not config.camera_swap_red_blue
    config.save()
    log(f"Camera red/blue swap -> {config.camera_swap_red_blue}", "INFO")
    return jsonify({"success": True, "swap_rb": config.camera_swap_red_blue})


@app.route("/api/calibration")
def api_calibration() -> Response:
    return jsonify(calibration_manager.as_dict())


@app.route("/api/calibration/start", methods=["POST"])
def api_calibration_start() -> Response:
    return jsonify({"success": True, "calibration": calibration_manager.start()})


@app.route("/api/calibration/point", methods=["POST"])
def api_calibration_point() -> Response:
    data = request.get_json(force=True) or {}
    calibration = calibration_manager.add_point(
        img_x_norm=clamp_norm(data.get("img_x_norm", 0)),
        img_y_norm=clamp_norm(data.get("img_y_norm", 0)),
        arm_x_cm=float(data.get("arm_x_cm", 0.0)),
        arm_y_cm=float(data.get("arm_y_cm", 0.0)),
    )
    return jsonify({"success": True, "calibration": calibration})


@app.route("/api/calibration/finish", methods=["POST"])
def api_calibration_finish() -> Response:
    try:
        calibration = calibration_manager.finish()
        return jsonify({"success": True, "calibration": calibration})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


def main() -> None:
    print(
        """
+-----------------------------------------------------------------------+
| AgroPick Low-Latency Controller                                       |
| Pi: live vision, web UI, autonomy                                     |
| ESP32: pure actuator over serial                                      |
+-----------------------------------------------------------------------+
"""
    )

    if not camera.start():
        return

    serial_client.start()

    if GOOGLE_API_KEY:
        vision.start()
    else:
        log("No GOOGLE_API_KEY configured. Vision features disabled.", "WARN")

    threading.Thread(target=control_loop, daemon=True).start()
    log(f"Open http://0.0.0.0:{config.web_port}", "SUCCESS")
    app.run(host="0.0.0.0", port=config.web_port, threaded=True, debug=False)


if __name__ == "__main__":
    try:
        main()
    finally:
        vision.stop()
        serial_client.stop()
        camera.stop()
