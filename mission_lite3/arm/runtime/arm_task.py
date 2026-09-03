from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 2.0
AUTO_ALIGN_STEPS = 30
AUTO_MAX_JOG_DEG = 3.0
SINGLE_STEP_ALIGN_STEPS = 8
SINGLE_STEP_MAX_JOG_DEG = 1.0
SMALL_NEGATIVE_E_LIMIT_DEG = 5.0
TASK_IDLE_STABILITY_TOLERANCE_DEGREES = 1.0
TASK_IDLE_STABLE_SAMPLES = 3
TASK_IDLE_WAIT_TIMEOUT_SECONDS = 8.0
GRIPPER_MOTION_WAIT_TIMEOUT_SECONDS = 25.0
GRIPPER_TARGET_TOLERANCE_DEGREES = 5.0
GRIPPER_STABILITY_TOLERANCE_DEGREES = 0.5
GRIPPER_STABLE_SAMPLES = 3
GRIPPER_MIN_CONTACT_MOVEMENT_DEGREES = 2.0
GRIPPER_HOLD_ANGLE_CHANGE_TOLERANCE_DEGREES = 3.0
JOINT_TARGET_TIMEOUT_MARGIN_SECONDS = 8.0
JOINT_TARGET_TIMEOUT_MIN_SPEED_DEG_PER_SEC = 1.0
STATUS_QUERY_RETRIES = 3
STATUS_QUERY_RETRY_DELAY_SECONDS = 0.2
STATUS_QUERY_RETRY_TIMEOUT_STEP_SECONDS = 0.25
DEFAULT_RUN_LOG_ROOT = MODULE_DIR / "grasp_runs"
DEFAULT_GRASP_CONFIG_PATH = MODULE_DIR / "strip_detector_grasp_config.json"
DEFAULT_GRASP_REFERENCE_PATH = MODULE_DIR / "grasp_reference_square_face.json"
CARDBOARD_HSV_LOWER = (8, 18, 70)
CARDBOARD_HSV_UPPER = (38, 140, 245)
CARDBOARD_LAB_B_RANGE = (132, 180)


class StatusIncompleteError(ValueError):
    def __init__(self, missing_fields: Iterable[str], status: Mapping[str, Any]):
        self.missing_fields = list(missing_fields)
        self.status = dict(status)
        super().__init__(
            "status missing required field(s): " + ", ".join(self.missing_fields)
        )


class TerminalAbortWatcher:
    def __init__(
        self,
        *,
        enabled: bool,
        read_key: Optional[Callable[[float], Optional[str]]] = None,
    ):
        self.enabled = bool(enabled)
        self._read_key = read_key
        self._abort_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._terminal_fd: Optional[int] = None
        self._terminal_attributes: Optional[Any] = None

    def _read_posix_key(self, timeout: float) -> Optional[str]:
        import select

        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        if not readable:
            return None
        return sys.stdin.read(1)

    def _watch(self) -> None:
        while not self._stop_event.is_set():
            try:
                key = self._read_key(0.1) if self._read_key is not None else None
            except (OSError, ValueError):
                return
            if key in ("q", "Q", "\x1b"):
                self._abort_event.set()
                return
            if key is None:
                self._stop_event.wait(0.01)

    def __enter__(self):
        if not self.enabled:
            return self
        if self._read_key is None:
            if os.name != "posix" or not sys.stdin.isatty():
                return self
            import termios
            import tty

            self._terminal_fd = sys.stdin.fileno()
            self._terminal_attributes = termios.tcgetattr(self._terminal_fd)
            tty.setcbreak(self._terminal_fd)
            self._read_key = self._read_posix_key
        self._thread = threading.Thread(
            target=self._watch,
            name="arm-task-terminal-abort",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.3)
        if self._terminal_fd is not None and self._terminal_attributes is not None:
            import termios

            termios.tcsetattr(
                self._terminal_fd,
                termios.TCSADRAIN,
                self._terminal_attributes,
            )

    def request_abort(self) -> None:
        self._abort_event.set()

    def abort_requested(self) -> bool:
        return self._abort_event.is_set()

    def wait_for_abort(self, timeout: float) -> bool:
        return self._abort_event.wait(timeout)


def _load_local_module(name: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


place_controller = _load_local_module("place_controller")
arm_grasp = _load_local_module("arm_grasp")
DEFAULT_GRASP_READY_POSE = dict(arm_grasp.DEFAULT_GRASP_READY_POSE_DEG)
DEFAULT_GRASP_READY_GRIPPER_H = arm_grasp.DEFAULT_GRASP_READY_GRIPPER_H
DEFAULT_GRASP_READY_TOLERANCE_DEG = arm_grasp.DEFAULT_GRASP_READY_TOLERANCE_DEG
GRASP_READY_NEGATIVE_E_COMPENSATION_DEG = arm_grasp.GRASP_READY_NEGATIVE_E_COMPENSATION_DEG
DEFAULT_TRANSPORT_POSE = DEFAULT_GRASP_READY_POSE
DEFAULT_TRANSPORT_GRIPPER_H = DEFAULT_GRASP_READY_GRIPPER_H
DEFAULT_HOLD_POSE = dict(DEFAULT_GRASP_READY_POSE)


def _load_arm_test_module():
    return _load_local_module("test")


def _mask_shape(mask: Any) -> tuple[int, int]:
    shape = getattr(mask, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    height = len(mask)
    width = len(mask[0]) if height else 0
    return int(height), int(width)


def _region_nonzero_ratio(mask: Any, x0: int, y0: int, x1: int, y1: int) -> float:
    width = max(0, int(x1) - int(x0))
    height = max(0, int(y1) - int(y0))
    total = width * height
    if total <= 0:
        return 0.0
    try:
        numpy = __import__("numpy")
        region = mask[y0:y1, x0:x1]
        return float(numpy.count_nonzero(region)) / float(total)
    except Exception:
        count = 0
        for row_index in range(y0, y1):
            row = mask[row_index]
            for column_index in range(x0, x1):
                if row[column_index]:
                    count += 1
        return float(count) / float(total)


def cardboard_ratios_from_mask(mask: Any) -> Dict[str, float]:
    height, width = _mask_shape(mask)
    if height <= 0 or width <= 0:
        return {"full_ratio": 0.0, "center_ratio": 0.0, "lower_center_ratio": 0.0}
    center_x0 = width // 4
    center_x1 = (width * 3) // 4
    center_y0 = height // 4
    center_y1 = (height * 3) // 4
    lower_y0 = height // 2
    lower_y1 = height
    return {
        "full_ratio": _region_nonzero_ratio(mask, 0, 0, width, height),
        "center_ratio": _region_nonzero_ratio(
            mask,
            center_x0,
            center_y0,
            center_x1,
            center_y1,
        ),
        "lower_center_ratio": _region_nonzero_ratio(
            mask,
            center_x0,
            lower_y0,
            center_x1,
            lower_y1,
        ),
    }


def cardboard_mask_from_frame(frame_bgr: Any, *, cv2_module: Optional[Any] = None) -> Any:
    cv2_module = cv2_module or __import__("cv2")
    numpy = __import__("numpy")
    hsv = cv2_module.cvtColor(frame_bgr, cv2_module.COLOR_BGR2HSV)
    lab = cv2_module.cvtColor(frame_bgr, cv2_module.COLOR_BGR2LAB)
    hsv_mask = cv2_module.inRange(
        hsv,
        numpy.array(CARDBOARD_HSV_LOWER, dtype=numpy.uint8),
        numpy.array(CARDBOARD_HSV_UPPER, dtype=numpy.uint8),
    )
    lab_b = lab[:, :, 2]
    lab_mask = cv2_module.inRange(
        lab_b,
        CARDBOARD_LAB_B_RANGE[0],
        CARDBOARD_LAB_B_RANGE[1],
    )
    return cv2_module.bitwise_and(hsv_mask, lab_mask)


class CardboardColorVision:
    def __init__(
        self,
        *,
        device: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        frames_per_sample: int = 3,
        cv2_module: Optional[Any] = None,
    ):
        self.cv2 = cv2_module or __import__("cv2")
        self.device = str(device)
        self.width = width
        self.height = height
        self.fps = fps
        self.frames_per_sample = max(1, int(frames_per_sample))
        self.capture = self._open_capture()
        self.run_log_directory: Optional[Path] = None
        self.sample_index = 0
        self.log_errors: list[str] = []
        self.abort_checker: Optional[Callable[[], bool]] = None

    def _open_capture(self) -> Any:
        backends = []
        if hasattr(self.cv2, "CAP_V4L2"):
            backends.append(self.cv2.CAP_V4L2)
        backends.append(None)
        last_capture = None
        for backend in backends:
            capture = (
                self.cv2.VideoCapture(self.device, backend)
                if backend is not None
                else self.cv2.VideoCapture(self.device)
            )
            last_capture = capture
            if self.width is not None:
                capture.set(self.cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
            if self.height is not None:
                capture.set(self.cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
            if self.fps is not None:
                capture.set(self.cv2.CAP_PROP_FPS, int(self.fps))
            if capture.isOpened():
                ok, frame = capture.read()
                if ok and frame is not None:
                    self._pending_frame = frame
                    return capture
            capture.release()
        raise RuntimeError(f"failed to open cardboard camera: {self.device}")

    def set_abort_checker(self, checker: Optional[Callable[[], bool]]) -> None:
        self.abort_checker = checker

    def set_run_log_directory(self, directory: Path) -> None:
        self.run_log_directory = Path(directory)
        self.run_log_directory.mkdir(parents=True, exist_ok=True)

    def _raise_if_aborted(self) -> None:
        if self.abort_checker is not None and bool(self.abort_checker()):
            raise KeyboardInterrupt

    def _read_frame(self) -> Any:
        frame = getattr(self, "_pending_frame", None)
        self._pending_frame = None
        for _ in range(self.frames_per_sample):
            self._raise_if_aborted()
            ok, candidate = self.capture.read()
            if ok and candidate is not None:
                frame = candidate
        if frame is None:
            raise RuntimeError("failed to read frame from cardboard camera")
        return frame

    def _write_sample_log(self, frame_bgr: Any, mask: Any, sample: Mapping[str, Any]) -> None:
        if self.run_log_directory is None:
            return
        prefix = f"place_cardboard_{self.sample_index:03d}"
        raw_path = self.run_log_directory / f"{prefix}.jpg"
        mask_path = self.run_log_directory / f"{prefix}_mask.png"
        json_path = self.run_log_directory / f"{prefix}.json"
        try:
            self.cv2.imwrite(str(raw_path), frame_bgr)
            self.cv2.imwrite(str(mask_path), mask)
            _write_json_file(json_path, sample)
        except Exception as exc:
            self.log_errors.append(str(exc))

    def detect_cardboard(self) -> Dict[str, Any]:
        self.sample_index += 1
        frame = self._read_frame()
        mask = cardboard_mask_from_frame(frame, cv2_module=self.cv2)
        ratios = cardboard_ratios_from_mask(mask)
        sample: Dict[str, Any] = {
            **ratios,
            "hsv_lower": list(CARDBOARD_HSV_LOWER),
            "hsv_upper": list(CARDBOARD_HSV_UPPER),
            "lab_b_range": list(CARDBOARD_LAB_B_RANGE),
        }
        self._write_sample_log(frame, mask, sample)
        return sample

    def close(self) -> None:
        self.capture.release()


class DryRunMotion:
    def __init__(self):
        self.calls = []
        self.event_log = []

    def jog(self, joint: str, delta_deg: float, *, spd: float, acc: float) -> None:
        self.calls.append(("jog", joint, float(delta_deg), spd, acc))

    def move_joints(
        self,
        joints: Mapping[str, float],
        *,
        spd: float,
        acc: float,
        tolerance_degrees: Optional[float] = None,
    ) -> None:
        self.calls.append(("move_joints", dict(joints), spd, acc))

    def move_joints_with_expected_deltas(
        self,
        command_deltas: Mapping[str, float],
        expected_deltas: Mapping[str, float],
        *,
        spd: float,
        acc: float,
        tolerance_degrees: Optional[float] = None,
    ) -> None:
        self.calls.append(
            (
                "move_joints_with_expected_deltas",
                dict(command_deltas),
                dict(expected_deltas),
                spd,
                acc,
                tolerance_degrees,
            )
        )

    def open_gripper(self, *, angle: float, spd: float, acc: float) -> None:
        self.calls.append(("open_gripper", float(angle), spd, acc))

    def close_gripper(self, *, angle: float, spd: float, acc: float) -> None:
        self.calls.append(("close_gripper", float(angle), spd, acc))

    def close_gripper_at_pose(
        self,
        *,
        angle: float,
        joints: Mapping[str, float],
        spd: float,
        acc: float,
    ) -> None:
        self.calls.append(("close_gripper_at_pose", float(angle), dict(joints), spd, acc))

    def home(self) -> None:
        self.calls.append(("home",))

    def abort(self) -> None:
        self.calls.append(("abort",))


class ArmTestSerialMotion:
    """Small adapter around arm/test.py so task code reuses its safety checks."""

    def __init__(
        self,
        *,
        port: str,
        arm_module: Optional[Any] = None,
        baudrate: int = DEFAULT_BAUD,
        timeout: float = DEFAULT_TIMEOUT,
        abort_checker: Optional[Callable[[], bool]] = None,
    ):
        if not port:
            raise ValueError("real motion requires --port")
        self.port = port
        self.arm = arm_module or _load_arm_test_module()
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.abort_checker = abort_checker
        self._last_command_pose: Optional[Dict[str, float]] = None
        self._last_gripper_close_result: Optional[Dict[str, Any]] = None
        self.event_log: list[Dict[str, Any]] = []

    def set_abort_checker(self, checker: Optional[Callable[[], bool]]) -> None:
        self.abort_checker = checker

    def _raise_if_aborted(self) -> None:
        if self.abort_checker is not None and bool(self.abort_checker()):
            raise KeyboardInterrupt

    @staticmethod
    def _is_retryable_status_error(exc: BaseException) -> bool:
        message = str(exc)
        if isinstance(exc, StatusIncompleteError):
            return True
        return "没有解析到JSON返回" in message or "JSON" in message

    def _require_complete_status(self, status: Mapping[str, Any]) -> None:
        required_fields = ("move", *self.arm.JOINT_KEYS, "t")
        missing_fields = [field for field in required_fields if field not in status]
        if missing_fields:
            raise StatusIncompleteError(missing_fields, status)

    @staticmethod
    def _raw_status_response_record(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, bytes):
            data = raw
        elif isinstance(raw, bytearray):
            data = bytes(raw)
        else:
            data = str(raw).encode("utf-8", errors="replace")
        return {
            "length": len(data),
            "hex": data.hex(),
            "text": data.decode("utf-8", errors="replace"),
        }

    def _log_status_parse_error(
        self,
        exc: BaseException,
        *,
        attempt: int,
        timeout: float,
    ) -> None:
        raw_responses = list(getattr(exc, "raw_responses", []) or [])
        raw_response = getattr(exc, "raw_response", None)
        if raw_response is not None:
            raw_responses.append(raw_response)
        event = {
            "type": "status_parse_error",
            "timestamp_unix": time.time(),
            "attempt": attempt,
            "timeout_seconds": timeout,
            "error": str(exc),
            "raw_responses": [
                self._raw_status_response_record(raw) for raw in raw_responses
            ],
        }
        if isinstance(exc, StatusIncompleteError):
            event["missing_fields"] = list(exc.missing_fields)
            event["status"] = dict(exc.status)
        self.event_log.append(event)

    def _query_status_with_retry(self, timeout: float) -> Dict[str, Any]:
        last_error: Optional[BaseException] = None
        for attempt in range(STATUS_QUERY_RETRIES):
            self._raise_if_aborted()
            attempt_timeout = timeout + attempt * STATUS_QUERY_RETRY_TIMEOUT_STEP_SECONDS
            try:
                status = self.arm.query_status(
                    self.port,
                    self.baudrate,
                    attempt_timeout,
                )
                self._require_complete_status(status)
                self.event_log.append(
                    {
                        "type": "status",
                        "timestamp_unix": time.time(),
                        "status": dict(status),
                    }
                )
                return status
            except (RuntimeError, ValueError) as exc:
                if not self._is_retryable_status_error(exc):
                    raise
                last_error = exc
                self._log_status_parse_error(
                    exc,
                    attempt=attempt + 1,
                    timeout=attempt_timeout,
                )
                if attempt >= STATUS_QUERY_RETRIES - 1:
                    break
                time.sleep(STATUS_QUERY_RETRY_DELAY_SECONDS)
        assert last_error is not None
        raise last_error

    def _query_status(self) -> Dict[str, Any]:
        return self._query_status_with_retry(self.timeout)

    def _query_fast_status(self) -> Dict[str, Any]:
        return self._query_status_with_retry(min(self.timeout, 0.5))

    def _wait_ready(
        self,
        initial_status: Dict[str, Any],
        *,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "query_status_fn": self._query_fast_status,
            "initial_status": initial_status,
            "tolerance_degrees": TASK_IDLE_STABILITY_TOLERANCE_DEGREES,
            "stable_samples": TASK_IDLE_STABLE_SAMPLES,
            "timeout_seconds": TASK_IDLE_WAIT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds,
        }
        return self.arm.wait_for_motion_idle(**kwargs)

    def status(self) -> Dict[str, Any]:
        return self._query_status()

    def current_pose_degrees(self) -> Dict[str, float]:
        status = self._wait_ready(self._query_status())
        return {
            joint: float(value)
            for joint, value in self.arm.status_to_command_degrees(status).items()
        }

    def _send(self, command: Dict[str, Any]) -> None:
        self.event_log.append(
            {
                "type": "command",
                "timestamp_unix": time.time(),
                "command": dict(command),
            }
        )
        self.arm.send_serial_command(
            self.port,
            command,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )

    def _pose_from_status(self, status: Mapping[str, Any]) -> Dict[str, float]:
        pose = {
            key: float(value)
            for key, value in self.arm.status_to_command_degrees(status).items()
        }
        pose["h"] = float(self.arm.status_to_gripper_degrees(status))
        return pose

    def _status_from_pose(self, pose: Mapping[str, float]) -> Dict[str, Any]:
        status: Dict[str, Any] = {"move": 0}
        for joint in self.arm.JOINT_KEYS:
            status[joint] = math.radians(
                self.arm.command_to_feedback_degrees(joint, float(pose[joint]))
            )
        status["t"] = math.radians(float(pose["h"]))
        return status

    def _remember_command_pose(self, command: Mapping[str, Any]) -> None:
        required = [*self.arm.JOINT_KEYS, "h"]
        if all(key in command for key in required):
            self._last_command_pose = {
                key: float(command[key])
                for key in required
            }

    def jog(self, joint: str, delta_deg: float, *, spd: float, acc: float) -> None:
        delta_deg = float(delta_deg)
        if joint == "e" and delta_deg < 0 and abs(delta_deg) < SMALL_NEGATIVE_E_LIMIT_DEG:
            raise ValueError("small negative e jog is disabled for automatic grasp")
        status = self._wait_ready(self._query_status())
        command_status = (
            self._status_from_pose(self._last_command_pose)
            if self._last_command_pose is not None
            else status
        )
        command = self.arm.build_jog_command(
            joint,
            delta_deg,
            command_status,
            spd=spd,
            acc=acc,
        )
        self._send(command)
        self._remember_command_pose(command)
        self._wait_ready(self._query_fast_status())

    def move_joints(
        self,
        joints: Mapping[str, float],
        *,
        spd: float,
        acc: float,
        tolerance_degrees: Optional[float] = None,
    ) -> None:
        status = self._wait_ready(self._query_status())
        command = {"T": 122}
        command.update(self.arm.status_to_command_degrees(status))
        command.update({joint: float(value) for joint, value in joints.items()})
        command.update(
            {
                "h": self.arm.status_to_gripper_degrees(status),
                "spd": spd,
                "acc": acc,
            }
        )
        self._send(command)
        self._remember_command_pose(command)
        for joint in joints:
            if joint in ("b", "s", "e", "w"):
                if self._joint_already_within_target(status, joint, command[joint], tolerance_degrees):
                    continue
                self.arm.wait_for_joint_target(
                    query_status_fn=self._query_fast_status,
                    joint=joint,
                    target_degrees=command[joint],
                    initial_status=status,
                    tolerance_degrees=tolerance_degrees
                    if tolerance_degrees is not None
                    else getattr(self.arm, "TARGET_TOLERANCE_DEGREES", 1.0),
                    timeout_seconds=self._joint_target_timeout_seconds(
                        status,
                        joint,
                        command[joint],
                        spd,
                    ),
                )

    def move_joints_with_expected_deltas(
        self,
        command_deltas: Mapping[str, float],
        expected_deltas: Mapping[str, float],
        *,
        spd: float,
        acc: float,
        tolerance_degrees: Optional[float] = None,
    ) -> None:
        command_delta_map = {joint: float(delta) for joint, delta in command_deltas.items()}
        expected_delta_map = {joint: float(delta) for joint, delta in expected_deltas.items()}
        if set(command_delta_map) != set(expected_delta_map):
            raise ValueError("command and expected compensated joints must match")

        status = self._wait_ready(self._query_status())
        current_pose = self.arm.status_to_command_degrees(status)
        command_targets = {
            joint: float(current_pose[joint]) + command_delta_map[joint]
            for joint in command_delta_map
        }
        expected_targets = {
            joint: float(current_pose[joint]) + expected_delta_map[joint]
            for joint in expected_delta_map
        }
        command = {"T": 122}
        command.update(current_pose)
        command.update(command_targets)
        command.update(
            {
                "h": self.arm.status_to_gripper_degrees(status),
                "spd": spd,
                "acc": acc,
            }
        )
        self._send(command)

        self._remember_command_pose(command)

        for joint, expected_target in expected_targets.items():
            if joint in ("b", "s", "e", "w"):
                if self._joint_already_within_target(status, joint, expected_target, tolerance_degrees):
                    continue
                self.arm.wait_for_joint_target(
                    query_status_fn=self._query_fast_status,
                    joint=joint,
                    target_degrees=expected_target,
                    initial_status=status,
                    tolerance_degrees=tolerance_degrees
                    if tolerance_degrees is not None
                    else getattr(self.arm, "TARGET_TOLERANCE_DEGREES", 1.0),
                    timeout_seconds=self._joint_target_timeout_seconds(
                        status,
                        joint,
                        expected_target,
                        spd,
                    ),
                )

    def move_joints_with_expected_targets(
        self,
        command_targets: Mapping[str, float],
        expected_targets: Mapping[str, float],
        *,
        spd: float,
        acc: float,
        tolerance_degrees: Optional[float] = None,
    ) -> None:
        command_target_map = {
            joint: float(target)
            for joint, target in command_targets.items()
        }
        expected_target_map = {
            joint: float(target)
            for joint, target in expected_targets.items()
        }
        if set(command_target_map) != set(expected_target_map):
            raise ValueError("command and expected compensated joints must match")

        status = self._wait_ready(self._query_status())
        command = {"T": 122}
        command.update(self.arm.status_to_command_degrees(status))
        command.update(command_target_map)
        command.update(
            {
                "h": self.arm.status_to_gripper_degrees(status),
                "spd": spd,
                "acc": acc,
            }
        )
        self._send(command)

        self._remember_command_pose(command)

        for joint, expected_target in expected_target_map.items():
            if joint not in ("b", "s", "e", "w"):
                continue
            if self._joint_already_within_target(
                status,
                joint,
                expected_target,
                tolerance_degrees,
            ):
                continue
            self.arm.wait_for_joint_target(
                query_status_fn=self._query_fast_status,
                joint=joint,
                target_degrees=expected_target,
                initial_status=status,
                tolerance_degrees=tolerance_degrees
                if tolerance_degrees is not None
                else getattr(self.arm, "TARGET_TOLERANCE_DEGREES", 1.0),
                timeout_seconds=self._joint_target_timeout_seconds(
                    status,
                    joint,
                    expected_target,
                    spd,
                ),
            )

    def _joint_target_timeout_seconds(
        self,
        status: Mapping[str, Any],
        joint: str,
        target_degrees: float,
        spd: float,
    ) -> float:
        default_timeout = float(
            getattr(self.arm, "MOTION_WAIT_TIMEOUT_SECONDS", 15.0)
        )
        try:
            joints = self.arm.status_to_joint_degrees(status)
            target_feedback = self.arm.command_to_feedback_degrees(
                joint,
                target_degrees,
            )
            travel_degrees = abs(float(joints[joint]) - float(target_feedback))
            speed = max(
                abs(float(spd)),
                JOINT_TARGET_TIMEOUT_MIN_SPEED_DEG_PER_SEC,
            )
        except (KeyError, TypeError, ValueError):
            return default_timeout
        if not math.isfinite(travel_degrees) or not math.isfinite(speed):
            return default_timeout
        return max(
            default_timeout,
            travel_degrees / speed + JOINT_TARGET_TIMEOUT_MARGIN_SECONDS,
        )

    def _joint_already_within_target(
        self,
        status: Mapping[str, Any],
        joint: str,
        target_degrees: float,
        tolerance_degrees: Optional[float] = None,
    ) -> bool:
        joints = self.arm.status_to_joint_degrees(status)
        target_feedback = self.arm.command_to_feedback_degrees(joint, target_degrees)
        tolerance = (
            float(tolerance_degrees)
            if tolerance_degrees is not None
            else float(getattr(self.arm, "TARGET_TOLERANCE_DEGREES", 1.0))
        )
        return abs(float(joints[joint]) - float(target_feedback)) <= tolerance

    def open_gripper(self, *, angle: float, spd: float, acc: float) -> Dict[str, Any]:
        status = self._wait_ready(self._query_status())
        initial_gripper_angle = float(self.arm.status_to_gripper_degrees(status))
        if abs(initial_gripper_angle - float(angle)) <= GRIPPER_TARGET_TOLERANCE_DEGREES:
            self._last_gripper_close_result = None
            return {
                "status": dict(status),
                "target_angle_deg": float(angle),
                "close_angle_deg": initial_gripper_angle,
                "target_reached": True,
                "contact_detected": False,
                "skipped": True,
            }
        result = self._set_gripper_angle(angle, spd=spd, acc=acc, initial_status=status)
        self._last_gripper_close_result = None
        return result

    def close_gripper(self, *, angle: float, spd: float, acc: float) -> Dict[str, Any]:
        result = self._set_gripper_angle(angle, spd=spd, acc=acc)
        self._last_gripper_close_result = dict(result)
        return result

    def close_gripper_at_pose(
        self,
        *,
        angle: float,
        joints: Mapping[str, float],
        spd: float,
        acc: float,
    ) -> Dict[str, Any]:
        result = self._set_gripper_angle(angle, spd=spd, acc=acc, joints=joints)
        self._last_gripper_close_result = dict(result)
        return result

    def _command_status_for_gripper(
        self,
        status: Mapping[str, Any],
        joints: Optional[Mapping[str, float]],
    ) -> Mapping[str, Any]:
        pose_override: Optional[Dict[str, float]] = None
        if joints is not None:
            pose_override = self._pose_from_status(status)
            for joint, value in joints.items():
                if joint not in self.arm.JOINT_KEYS:
                    raise ValueError(f"unknown joint for gripper pose: {joint}")
                pose_override[joint] = float(value)
        elif self._last_command_pose is not None:
            pose_override = dict(self._last_command_pose)

        if pose_override is None:
            return status
        pose_override["h"] = float(self.arm.status_to_gripper_degrees(status))
        return self._status_from_pose(pose_override)

    def _set_gripper_angle(
        self,
        angle: float,
        *,
        spd: float,
        acc: float,
        joints: Optional[Mapping[str, float]] = None,
        initial_status: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        status = dict(initial_status) if initial_status is not None else self._wait_ready(self._query_status())
        initial_gripper_angle = float(self.arm.status_to_gripper_degrees(status))
        command_status = self._command_status_for_gripper(status, joints)
        command = self.arm.build_gripper_angle_command(
            angle,
            command_status,
            spd=spd,
            acc=acc,
        )
        self._send(command)
        self._remember_command_pose(command)
        return self._wait_gripper_angle(angle, initial_angle=initial_gripper_angle)

    def _wait_gripper_angle(self, target_angle: float, *, initial_angle: float) -> Dict[str, Any]:
        deadline = time.monotonic() + GRIPPER_MOTION_WAIT_TIMEOUT_SECONDS
        previous_angle: Optional[float] = None
        stable_count = 0
        last_status: Dict[str, Any] = self._query_fast_status()
        poll_seconds = float(getattr(self.arm, "MOTION_POLL_SECONDS", 0.05))

        while True:
            current_angle = float(self.arm.status_to_gripper_degrees(last_status))
            target_error = abs(current_angle - float(target_angle))
            movement_from_initial = abs(current_angle - float(initial_angle))
            contact_limited_close = (
                float(target_angle) > float(initial_angle)
                and movement_from_initial >= GRIPPER_MIN_CONTACT_MOVEMENT_DEGREES
            )
            if previous_angle is None:
                stable_count = 1
            elif abs(current_angle - previous_angle) <= GRIPPER_STABILITY_TOLERANCE_DEGREES:
                stable_count += 1
            else:
                stable_count = 1

            if (
                (
                    target_error <= GRIPPER_TARGET_TOLERANCE_DEGREES
                    or contact_limited_close
                )
                and stable_count >= GRIPPER_STABLE_SAMPLES
            ):
                target_reached = target_error <= GRIPPER_TARGET_TOLERANCE_DEGREES
                return {
                    "status": dict(last_status),
                    "target_angle_deg": float(target_angle),
                    "close_angle_deg": current_angle,
                    "target_reached": target_reached,
                    "contact_detected": bool(contact_limited_close and not target_reached),
                }

            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "夹爪未稳定到目标角度："
                    f"目标={float(target_angle):.2f}度，"
                    f"当前={current_angle:.2f}度，"
                    f"误差={target_error:.2f}度，"
                    f"move={last_status.get('move', '缺失')}"
                )

            previous_angle = current_angle
            time.sleep(poll_seconds)
            last_status = self._query_fast_status()

    def home(self) -> None:
        status = self._wait_ready(self._query_status())
        self._send(self.arm.build_home_command(status))

    def abort(self) -> None:
        # The current TLS test tool has no verified emergency-stop command.
        # Keep this as a task-level state transition until a real stop command is validated.
        return None


class ArmTask:
    def __init__(
        self,
        *,
        motion: Optional[Any] = None,
        vision: Optional[Any] = None,
        place_vision: Optional[Any] = None,
        grasp_reference: Optional[Mapping[str, Any]] = None,
        place_reference: Optional[Mapping[str, Any]] = None,
        dry_run: bool = False,
        single_step: bool = False,
        max_align_steps: int = AUTO_ALIGN_STEPS,
        max_jog_deg: float = AUTO_MAX_JOG_DEG,
        spd: float = arm_grasp.DEFAULT_SPEED,
        acc: float = arm_grasp.DEFAULT_ACCELERATION,
        final_spd: Optional[float] = None,
        final_acc: Optional[float] = None,
        skip_grasp_ready: bool = False,
        stop_after_final_pose: bool = False,
        abort_checker: Optional[Callable[[], bool]] = None,
    ):
        self.motion = motion or DryRunMotion()
        self.vision = vision or arm_grasp.StaticVision()
        self.place_vision = place_vision
        self.grasp_reference = grasp_reference or arm_grasp.default_grasp_reference()
        self.place_reference = place_reference or place_controller.default_place_reference()
        self.dry_run = bool(dry_run)
        self.single_step = bool(single_step)
        self.max_align_steps = int(max_align_steps)
        self.max_jog_deg = float(max_jog_deg)
        self.spd = float(spd)
        self.acc = float(acc)
        self.final_spd = final_spd
        self.final_acc = final_acc
        self.skip_grasp_ready = bool(skip_grasp_ready)
        self.stop_after_final_pose = bool(stop_after_final_pose)
        self.abort_checker = abort_checker
        set_motion_abort_checker = getattr(self.motion, "set_abort_checker", None)
        if callable(set_motion_abort_checker):
            set_motion_abort_checker(self.abort_checker)
        self.stage = "IDLE"
        self.object_held = False

    def _finish(self, payload: Dict[str, Any], started_at: float) -> Dict[str, Any]:
        payload.setdefault("elapsed_seconds", round(time.monotonic() - started_at, 3))
        payload.setdefault("object_held", self.object_held)
        return payload

    def status(self) -> Dict[str, Any]:
        payload = {"ok": True, "stage": self.stage, "object_held": self.object_held}
        if hasattr(self.motion, "status"):
            payload["arm_status"] = self.motion.status()
        return payload

    def grasp(self) -> Dict[str, Any]:
        started_at = time.monotonic()
        machine = arm_grasp.ArmGraspStateMachine(
            vision=self.vision,
            motion=self.motion,
            reference=self.grasp_reference,
            dry_run=True if self.dry_run else False,
            single_step=self.single_step,
            max_align_steps=self.max_align_steps,
            max_jog_deg=self.max_jog_deg,
            spd=self.spd,
            acc=self.acc,
            final_spd=self.final_spd,
            final_acc=self.final_acc,
            run_grasp_ready=not self.skip_grasp_ready,
            stop_after_final_pose=self.stop_after_final_pose,
            abort_checker=self.abort_checker,
        )
        result = machine.run().to_dict()
        self.stage = "HOLDING" if result["ok"] and result.get("object_held") else result["stage"]
        self.object_held = bool(result.get("object_held"))
        return self._finish(result, started_at)

    def hold(self) -> Dict[str, Any]:
        started_at = time.monotonic()
        plan = [{"stage": "MOVE_TO_HOLD_POSE", "joints_deg": dict(DEFAULT_HOLD_POSE)}]
        if not self.dry_run:
            self.motion.move_joints(DEFAULT_HOLD_POSE, spd=self.spd, acc=self.acc)
        self.object_held = True
        self.stage = "HOLDING"
        return self._finish(
            {"ok": True, "stage": "HOLDING", "reason": "", "plan": plan},
            started_at,
        )

    def grasp_ready(self) -> Dict[str, Any]:
        started_at = time.monotonic()
        ready_step = {
            "stage": "MOVE_TO_GRASP_READY_POSE",
            "joints_deg": dict(DEFAULT_GRASP_READY_POSE),
        }
        plan = [
            {"stage": "OPEN_GRIPPER", "open_gripper_h": DEFAULT_GRASP_READY_GRIPPER_H},
            ready_step,
        ]
        if not self.dry_run:
            self.motion.open_gripper(angle=DEFAULT_GRASP_READY_GRIPPER_H, spd=self.spd, acc=self.acc)
            current_pose_fn = getattr(self.motion, "current_pose_degrees", None)
            compensated_move_fn = getattr(
                self.motion,
                "move_joints_with_expected_targets",
                None,
            )
            compensated = False
            if callable(current_pose_fn) and callable(compensated_move_fn):
                current_pose = current_pose_fn()
                e_delta = (
                    DEFAULT_GRASP_READY_POSE["e"]
                    - float(current_pose["e"])
                )
                if e_delta < -DEFAULT_GRASP_READY_TOLERANCE_DEG:
                    command_targets = dict(DEFAULT_GRASP_READY_POSE)
                    command_targets["e"] -= GRASP_READY_NEGATIVE_E_COMPENSATION_DEG
                    ready_step.update(
                        {
                            "action": "move_joints_with_expected_targets",
                            "command_joints_deg": dict(command_targets),
                            "expected_joints_deg": dict(DEFAULT_GRASP_READY_POSE),
                            "e_negative_compensation_deg": (
                                GRASP_READY_NEGATIVE_E_COMPENSATION_DEG
                            ),
                        }
                    )
                    compensated_move_fn(
                        command_targets,
                        DEFAULT_GRASP_READY_POSE,
                        spd=self.spd,
                        acc=self.acc,
                        tolerance_degrees=DEFAULT_GRASP_READY_TOLERANCE_DEG,
                    )
                    compensated = True
            if not compensated:
                ready_step["action"] = "move_joints"
                self.motion.move_joints(
                    DEFAULT_GRASP_READY_POSE,
                    spd=self.spd,
                    acc=self.acc,
                    tolerance_degrees=DEFAULT_GRASP_READY_TOLERANCE_DEG,
                )
        self.object_held = False
        self.stage = "GRASP_READY"
        return self._finish(
            {"ok": True, "stage": "GRASP_READY", "reason": "", "plan": plan},
            started_at,
        )

    def transport(self) -> Dict[str, Any]:
        return self.grasp_ready()

    def _reference_close_angle(self, angle: Optional[float]) -> float:
        if angle is not None:
            return float(angle)
        terminal = self.grasp_reference.get("terminal_sequence", {})
        return float(terminal.get("close_gripper_h", 20.0))

    def _reference_final_grasp_pose(self) -> Optional[Dict[str, float]]:
        approach = self.grasp_reference.get("approach_sequence", {})
        if bool(approach.get("requires_reteach", False)):
            return None
        pose = approach.get("grasp_pose_deg", approach.get("final_grasp_pose_deg"))
        if not pose:
            return None
        return {joint: float(value) for joint, value in dict(pose).items()}

    def close(self, angle: Optional[float] = None) -> Dict[str, Any]:
        started_at = time.monotonic()
        close_angle = self._reference_close_angle(angle)
        final_pose = self._reference_final_grasp_pose()
        close_step: Dict[str, Any] = {
            "stage": "CLOSE_GRIPPER",
            "close_gripper_h": close_angle,
            "pose_source": "current_status",
        }
        if final_pose:
            close_step["joints_deg"] = dict(final_pose)
            close_step["pose_source"] = "reference_final_grasp_pose"
        plan = [close_step]

        if not self.dry_run:
            if final_pose and hasattr(self.motion, "close_gripper_at_pose"):
                self.motion.close_gripper_at_pose(
                    angle=close_angle,
                    joints=final_pose,
                    spd=self.spd,
                    acc=self.acc,
                )
            else:
                self.motion.close_gripper(angle=close_angle, spd=self.spd, acc=self.acc)

        self.object_held = True
        self.stage = "CLOSED"
        return self._finish(
            {"ok": True, "stage": "CLOSED", "reason": "", "plan": plan},
            started_at,
        )

    def place(self, slot: str) -> Dict[str, Any]:
        started_at = time.monotonic()
        controller = place_controller.PlaceController(
            self.place_reference,
            self.motion,
            place_vision=self.place_vision,
            spd=place_controller.DEFAULT_SPEED,
            acc=place_controller.DEFAULT_ACCELERATION,
        )
        result = controller.place(slot, object_held=self.object_held, dry_run=self.dry_run)
        if (result.ok or result.released) and not self.dry_run:
            self.object_held = result.object_held
        if result.ok and self.dry_run:
            self.stage = "PLACE_DRY_RUN"
        elif result.ok:
            self.stage = "PLACED"
        else:
            self.stage = result.stage
        payload = result.to_dict()
        payload["object_held"] = self.object_held if self.dry_run else result.object_held
        return self._finish(payload, started_at)

    def home(self) -> Dict[str, Any]:
        started_at = time.monotonic()
        if not self.dry_run:
            self.motion.home()
        self.stage = "IDLE"
        return self._finish({"ok": True, "stage": "IDLE", "reason": ""}, started_at)

    def abort(self) -> Dict[str, Any]:
        started_at = time.monotonic()
        if self.dry_run:
            self.stage = "ABORTED"
            return self._finish(
                {"ok": True, "stage": "ABORTED", "reason": ""},
                started_at,
            )
        self.stage = "MANUAL_RECOVERY_REQUIRED"
        return self._finish(
            {
                "ok": False,
                "stage": "MANUAL_RECOVERY_REQUIRED",
                "reason": (
                    "no verified hardware emergency-stop command; "
                    "manual recovery required"
                ),
                "object_held": self.object_held,
            },
            started_at,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arm task entry for grasp, hold and place")
    parser.add_argument("--port", help="serial device for real motion, for example /dev/ttyUSB0")
    parser.add_argument("--device", help="reserved camera device argument")
    parser.add_argument("--calibration", help="reserved camera calibration argument")
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_GRASP_REFERENCE_PATH,
        help=f"grasp reference file, default {DEFAULT_GRASP_REFERENCE_PATH.name}",
    )
    parser.add_argument("--dry-run", action="store_true", help="print plan only, do not move the arm")
    parser.add_argument("--single-step", action="store_true", help="run at most one visual alignment step")
    parser.add_argument(
        "--stop-after-final-pose",
        action="store_true",
        help="stop after MOVE_TO_FINAL_GRASP_POSE, before final recheck and gripper close",
    )
    parser.add_argument(
        "--skip-grasp-ready",
        action="store_true",
        help="skip grasp-ready pose setup; use after the arm is already in grasp-ready pose",
    )
    parser.add_argument(
        "--show-vision",
        action="store_true",
        help="show the live grasp detection window while reading camera frames",
    )
    parser.add_argument(
        "--hold-vision",
        action="store_true",
        help="keep the vision window open after the task finishes; requires --show-vision",
    )
    parser.add_argument(
        "--require-preflight",
        action="store_true",
        help="also require preflight for dry-run; real grasp always runs preflight",
    )
    parser.add_argument("--json-result", action="store_true", help="print JSON result")
    parser.add_argument("--result-file", type=Path, help="write JSON result to this file")
    parser.add_argument(
        "--run-log-dir",
        type=Path,
        help="write request, result, motion events and detection frames to this directory",
    )
    parser.add_argument("--place-reference", type=Path, help="A/B/C/D place reference file")
    parser.add_argument("--config", type=Path, default=DEFAULT_GRASP_CONFIG_PATH)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--frames-per-detect", type=int, default=5)
    parser.add_argument("--max-align-steps", type=int)
    parser.add_argument("--max-jog-deg", type=float)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--spd", type=float, default=arm_grasp.DEFAULT_SPEED)
    parser.add_argument("--acc", type=float, default=arm_grasp.DEFAULT_ACCELERATION)
    parser.add_argument("--final-spd", type=float)
    parser.add_argument("--final-acc", type=float)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("grasp")
    subparsers.add_parser("grasp-ready")
    subparsers.add_parser("transport")
    close_parser = subparsers.add_parser("close")
    close_parser.add_argument("--angle", type=float, help="override close gripper angle")
    subparsers.add_parser("hold")
    place_parser = subparsers.add_parser("place")
    place_parser.add_argument("--slot", default="A")
    place_parser.add_argument(
        "--object-held",
        action="store_true",
        help="declare that the arm is already holding the red strip before place",
    )
    subparsers.add_parser("home")
    subparsers.add_parser("status")
    subparsers.add_parser("abort")
    subparsers.add_parser("preflight")
    diagnose_parser = subparsers.add_parser(
        "diagnose-run",
        help="read a grasp run log directory and summarize failure cause",
    )
    diagnose_parser.add_argument("directory", type=Path, nargs="?")
    validate_parser = subparsers.add_parser(
        "validate-run",
        help="read a grasp run log directory and check whether it satisfies the fixed-base acceptance gate",
    )
    validate_parser.add_argument("directory", type=Path, nargs="?")
    return parser


def _resolve_alignment_defaults(args: argparse.Namespace) -> None:
    if args.max_align_steps is None:
        args.max_align_steps = SINGLE_STEP_ALIGN_STEPS if args.single_step else AUTO_ALIGN_STEPS
    if args.max_jog_deg is None:
        args.max_jog_deg = SINGLE_STEP_MAX_JOG_DEG if args.single_step else AUTO_MAX_JOG_DEG


def _load_grasp_reference(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        if DEFAULT_GRASP_REFERENCE_PATH.exists():
            return arm_grasp.load_grasp_reference(DEFAULT_GRASP_REFERENCE_PATH)
        return arm_grasp.default_grasp_reference()
    return arm_grasp.load_grasp_reference(path)


def _build_grasp_vision(args: argparse.Namespace):
    if args.command != "grasp" or not args.device:
        return arm_grasp.StaticVision(), None
    vision = arm_grasp.open_strip_camera_vision(
        device=args.device,
        config_path=args.config,
        calibration_path=args.calibration,
        width=args.width,
        height=args.height,
        fps=args.fps,
        frames_per_detect=args.frames_per_detect,
        show_window=args.show_vision,
    )
    return vision, vision


def _build_place_vision(args: argparse.Namespace):
    return None, None


def _load_place_reference(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        default_path = MODULE_DIR / "place_reference.json"
        if default_path.exists():
            return place_controller.load_place_reference(default_path)
        return place_controller.default_place_reference()
    return place_controller.load_place_reference(path)


def _failure_result(stage: str, reason: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "stage": stage,
        "reason": reason,
        "feedback": "arm_control_failed",
        "object_held": False,
        "elapsed_seconds": 0.0,
    }


def _aborted_result(reason: str = "user interrupted") -> Dict[str, Any]:
    return {
        "ok": False,
        "stage": "ABORTED",
        "reason": reason,
        "feedback": "user_aborted",
        "object_held": False,
        "elapsed_seconds": 0.0,
    }


def _write_and_print_result(args: argparse.Namespace, result: Dict[str, Any]) -> None:
    if args.result_file:
        args.result_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.json_result:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    elif not args.result_file:
        print(result)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _resolve_run_log_directory(args: argparse.Namespace) -> Optional[Path]:
    if args.run_log_dir is not None:
        return Path(args.run_log_dir)
    if args.command != "grasp" or args.dry_run:
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = f"{time.time_ns() % 1_000_000_000:09d}"
    return DEFAULT_RUN_LOG_ROOT / f"{timestamp}_{suffix}"


def _write_json_file(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_run_log(
    directory: Path,
    *,
    args: argparse.Namespace,
    result: Dict[str, Any],
    motion: Any,
    vision: Any,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    result["run_log_dir"] = str(directory)
    vision_log_errors = list(getattr(vision, "log_errors", []))
    if vision_log_errors:
        result["vision_log_errors"] = vision_log_errors
    _write_json_file(directory / "request.json", vars(args))
    _write_json_file(directory / "result.json", result)
    _write_json_file(
        directory / "motion_events.json",
        list(getattr(motion, "event_log", [])),
    )
    _write_json_file(directory / "diagnosis.json", diagnose_run_log(directory))
    _write_json_file(directory / "validation.json", validate_run_log(directory))


def _read_run_log_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


GRASP_SCHEME_NODES = [
    {
        "index": 1,
        "name": "进入抓取准备姿态",
        "stages": ["GRASP_READY", "OPEN_GRIPPER", "MOVE_TO_GRASP_READY_POSE"],
    },
    {
        "index": 2,
        "name": "检测红色长条端面",
        "stages": ["DETECT_RED_STRIP"],
    },
    {
        "index": 3,
        "name": "正方形端面粗对准",
        "stages": ["VISUAL_ALIGN"],
    },
    {
        "index": 4,
        "name": "计算最终抓取姿态补偿",
        "stages": ["CALCULATE_FINAL_GRASP_COMPENSATION"],
    },
    {
        "index": 5,
        "name": "移动到补偿后的最终抓取姿态",
        "stages": ["MOVE_TO_FINAL_GRASP_POSE"],
    },
    {
        "index": 6,
        "name": "最终画面复检",
        "stages": ["FINAL_VIEW_RECHECK"],
    },
    {
        "index": 7,
        "name": "闭合夹爪",
        "stages": ["CLOSE_GRIPPER", "CLOSED"],
    },
    {
        "index": 8,
        "name": "抬升与货运姿态",
        "stages": ["LIFT", "HOLD_POSE", "MOVE_TO_CARGO_POSE", "HOLDING"],
    },
]

GRASP_SCHEME_STAGE_TO_NODE = {
    stage: node
    for node in GRASP_SCHEME_NODES
    for stage in node["stages"]
}


def _public_scheme_node(node: Mapping[str, Any], stage: str) -> Dict[str, Any]:
    return {
        "index": int(node["index"]),
        "name": str(node["name"]),
        "stage": stage,
    }


def _scheme_node_for_stage(stage: str) -> Dict[str, Any]:
    if stage == "PREFLIGHT":
        return {"index": 0, "name": "预检", "stage": stage}
    if stage == "DONE":
        return {"index": 9, "name": "抓取流程完成", "stage": stage}
    node = GRASP_SCHEME_STAGE_TO_NODE.get(stage)
    if node is None:
        return {"index": None, "name": "未知节点", "stage": stage}
    return _public_scheme_node(node, stage)


def _scheme_progress_for_stage(stage: str, task_ok: bool) -> Dict[str, list[Dict[str, Any]]]:
    nodes = [
        {"index": int(node["index"]), "name": str(node["name"])}
        for node in GRASP_SCHEME_NODES
    ]
    if task_ok or stage == "DONE":
        return {"completed": nodes, "remaining": []}

    scheme_node = _scheme_node_for_stage(stage)
    index = scheme_node.get("index")
    if not isinstance(index, int) or index < 1:
        return {"completed": [], "remaining": nodes}
    return {
        "completed": [node for node in nodes if node["index"] < index],
        "remaining": [node for node in nodes if node["index"] >= index],
    }

def _diagnosis_next_action(feedback: str, stage: str) -> str:
    if feedback == "preflight_failed" or stage == "PREFLIGHT":
        return "先修复 failed_checks 中列出的预检失败项，再重新运行 preflight 或 grasp。"
    if stage == "FINAL_VIEW_RECHECK":
        return (
            "查看最后一张 annotated 图像，并对照 reason 中的 center_error_px、"
            "size_ratio 和 angle_error_deg；最终画面未通过前不要闭合夹爪。"
        )
    if feedback in {"target_left", "target_right", "target_too_far", "target_too_near", "target_lost"}:
        return "查看最新 annotated 图像，确认红色长条相对参考图的位置、大小和稳定性。"
    if feedback == "object_not_held" or stage == "VERIFY_HOLD":
        return "检查夹爪闭合角、闭合后 t 角变化和抬升后的持物状态。"
    if feedback == "arm_control_failed":
        return "查看 motion_events.json 中最后的 command/status，确认串口、关节目标和反馈角误差。"
    if feedback == "user_aborted":
        return "任务由人工中断；查看中断前最后一张 annotated 图像和最后一条运动事件。"
    return "结合 result.json、motion_events.json 和 annotated 图像定位失败原因。"


def _latest_motion_event_of_type(
    motion_events: list[Any],
    event_type: str,
) -> Optional[Dict[str, Any]]:
    for event in reversed(motion_events):
        if isinstance(event, dict) and event.get("type") == event_type:
            return event
    return None


def diagnose_run_log(directory: Path) -> Dict[str, Any]:
    run_log_dir = Path(directory)
    if not run_log_dir.exists():
        return {
            "ok": False,
            "stage": "RUN_DIAGNOSIS",
            "reason": f"run log directory does not exist: {run_log_dir}",
            "feedback": "run_log_missing",
        }

    result_path = run_log_dir / "result.json"
    if not result_path.exists():
        return {
            "ok": False,
            "stage": "RUN_DIAGNOSIS",
            "reason": f"missing result.json in run log directory: {run_log_dir}",
            "feedback": "run_log_incomplete",
        }

    try:
        task_result = _read_run_log_json(result_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "stage": "RUN_DIAGNOSIS",
            "reason": f"cannot read result.json: {exc}",
            "feedback": "run_log_invalid",
        }

    motion_events_path = run_log_dir / "motion_events.json"
    motion_events: list[Any] = []
    motion_events_error = ""
    if motion_events_path.exists():
        try:
            loaded_events = _read_run_log_json(motion_events_path)
            if isinstance(loaded_events, list):
                motion_events = loaded_events
            else:
                motion_events_error = "motion_events.json is not a list"
        except (OSError, json.JSONDecodeError) as exc:
            motion_events_error = str(exc)
    else:
        motion_events_error = "motion_events.json is missing"

    task_ok = bool(task_result.get("ok"))
    task_stage = str(task_result.get("stage", "UNKNOWN"))
    feedback = str(task_result.get("feedback", ""))
    reason = str(task_result.get("reason", ""))
    annotated_images = sorted(str(path) for path in run_log_dir.glob("*_annotated.jpg"))
    raw_images = sorted(str(path) for path in run_log_dir.glob("*_raw.jpg"))
    latest_motion_event = motion_events[-1] if motion_events else None
    latest_status_event = _latest_motion_event_of_type(motion_events, "status")
    latest_command_event = _latest_motion_event_of_type(motion_events, "command")
    scheme_node = _scheme_node_for_stage(task_stage)
    scheme_progress = _scheme_progress_for_stage(task_stage, task_ok)
    plan = task_result.get("plan")
    failed_plan_step = None
    if not task_ok and isinstance(plan, list):
        failed_plan_step = next(
            (
                dict(step)
                for step in reversed(plan)
                if isinstance(step, dict) and str(step.get("stage", "")) == task_stage
            ),
            None,
        )

    if task_ok:
        summary = f"任务成功，最终节点 {task_stage}，方案节点已全部完成。"
    elif isinstance(scheme_node.get("index"), int) and int(scheme_node["index"]) > 0:
        summary = (
            f"失败节点 {task_stage}（方案第{scheme_node['index']}步：{scheme_node['name']}），"
            f"反馈 {feedback or 'unknown'}，原因：{reason or '未记录'}"
        )
    else:
        summary = f"失败节点 {task_stage}，反馈 {feedback or 'unknown'}，原因：{reason or '未记录'}"


    diagnosis: Dict[str, Any] = {
        "ok": True,
        "stage": "RUN_DIAGNOSIS",
        "run_log_dir": str(run_log_dir),
        "task_ok": task_ok,
        "task_stage": task_stage,
        "feedback": feedback,
        "reason": reason,
        "object_held": bool(task_result.get("object_held", False)),
        "summary": summary,
        "next_action": _diagnosis_next_action(feedback, task_stage),
        "scheme_node": scheme_node,
        "completed_scheme_nodes": scheme_progress["completed"],
        "remaining_scheme_nodes": scheme_progress["remaining"],
        "latest_motion_event": latest_motion_event,
        "latest_status_event": latest_status_event,
        "latest_command_event": latest_command_event,
        "motion_event_count": len(motion_events),
        "annotated_images": annotated_images,
        "raw_images": raw_images,
        "failed_plan_step": failed_plan_step,
    }
    checks = task_result.get("checks")
    if isinstance(checks, list):
        diagnosis["failed_checks"] = [
            check for check in checks
            if isinstance(check, dict) and not check.get("ok")
        ]
    if motion_events_error:
        diagnosis["motion_events_error"] = motion_events_error
    return diagnosis


def _validation_requirement(
    name: str,
    ok: bool,
    *,
    reason: str = "",
    count: Optional[int] = None,
) -> Dict[str, Any]:
    requirement: Dict[str, Any] = {"name": name, "ok": bool(ok)}
    if reason:
        requirement["reason"] = reason
    if count is not None:
        requirement["count"] = count
    return requirement


def _plan_contains_stage_sequence(plan: Any, required_stages: list[str]) -> tuple[bool, str, list[str]]:
    if not isinstance(plan, list):
        return False, "result plan is missing or not a list", []
    stages = [
        str(step.get("stage", ""))
        for step in plan
        if isinstance(step, dict)
    ]
    search_index = 0
    for required_stage in required_stages:
        try:
            found_index = stages.index(required_stage, search_index)
        except ValueError:
            return False, f"missing required plan stage sequence: {' -> '.join(required_stages)}", stages
        search_index = found_index + 1
    return True, "", stages


def _plan_close_gripper_angle(plan: Any) -> Optional[float]:
    if not isinstance(plan, list):
        return None
    for step in plan:
        if not isinstance(step, dict):
            continue
        if step.get("stage") != "CLOSE_GRIPPER":
            continue
        if "close_gripper_h" not in step:
            return None
        try:
            return float(step["close_gripper_h"])
        except (TypeError, ValueError):
            return None
    return None


def _plan_step(plan: Any, stage: str) -> Optional[Dict[str, Any]]:
    if not isinstance(plan, list):
        return None
    for step in plan:
        if isinstance(step, dict) and step.get("stage") == stage:
            return step
    return None


def _plan_has_final_recheck_target(plan: Any) -> tuple[bool, str]:
    step = _plan_step(plan, "FINAL_VIEW_RECHECK")
    if step is None:
        return False, "result plan is missing FINAL_VIEW_RECHECK"
    target = step.get("target")
    if not isinstance(target, dict):
        return False, "FINAL_VIEW_RECHECK has no target object"
    return True, ""


def _plan_has_final_pose_visual_compensation(plan: Any) -> tuple[bool, str]:
    step = _plan_step(plan, "MOVE_TO_FINAL_GRASP_POSE")
    if step is None:
        return False, "result plan is missing MOVE_TO_FINAL_GRASP_POSE"
    visual_error = step.get("visual_error")
    if not isinstance(visual_error, dict):
        return False, "MOVE_TO_FINAL_GRASP_POSE has no visual_error object"
    if "center_px" not in visual_error or "size_ratio" not in visual_error:
        return False, "MOVE_TO_FINAL_GRASP_POSE visual_error lacks center_px or size_ratio"
    if not isinstance(step.get("visual_adjustment_deg"), dict):
        return False, "MOVE_TO_FINAL_GRASP_POSE has no visual_adjustment_deg object"
    return True, ""


def _motion_events_include_arm_status(
    motion_events: list[Any],
) -> tuple[bool, str, int]:
    required_fields = {"b", "s", "e", "w", "t", "x", "y", "z", "move"}
    status_event_count = 0
    last_missing_fields: set[str] = set(required_fields)
    for event in motion_events:
        if not isinstance(event, dict) or event.get("type") != "status":
            continue
        status = event.get("status")
        if not isinstance(status, dict):
            continue
        status_event_count += 1
        last_missing_fields = required_fields.difference(status)
        if not last_missing_fields:
            return True, "", status_event_count
    if status_event_count == 0:
        return False, "motion_events.json has no arm status events", 0
    return (
        False,
        "arm status events lack required fields: "
        + ", ".join(sorted(last_missing_fields)),
        status_event_count,
    )


def _motion_events_include_gripper_close_command(
    motion_events: list[Any],
    close_gripper_h: Optional[float],
    *,
    tolerance_deg: float = 0.25,
) -> tuple[bool, str]:
    if close_gripper_h is None:
        return False, "result plan CLOSE_GRIPPER is missing close_gripper_h"
    for event in motion_events:
        if not isinstance(event, dict) or event.get("type") != "command":
            continue
        command = event.get("command")
        if not isinstance(command, dict):
            continue
        try:
            command_type = int(command.get("T"))
            command_h = float(command["h"])
        except (KeyError, TypeError, ValueError):
            continue
        if command_type == 122 and abs(command_h - close_gripper_h) <= tolerance_deg:
            return True, ""
    return False, f"no T=122 close command with h near {close_gripper_h:g}"


def validate_run_log(directory: Path) -> Dict[str, Any]:
    run_log_dir = Path(directory)
    if not run_log_dir.exists():
        return {
            "ok": False,
            "stage": "RUN_VALIDATION",
            "accepted": False,
            "reason": f"run log directory does not exist: {run_log_dir}",
            "feedback": "run_log_missing",
            "failed_requirements": ["result"],
        }

    result_path = run_log_dir / "result.json"
    if not result_path.exists():
        return {
            "ok": False,
            "stage": "RUN_VALIDATION",
            "accepted": False,
            "reason": f"missing result.json in run log directory: {run_log_dir}",
            "feedback": "run_log_incomplete",
            "failed_requirements": ["result"],
        }

    try:
        task_result = _read_run_log_json(result_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "stage": "RUN_VALIDATION",
            "accepted": False,
            "reason": f"cannot read result.json: {exc}",
            "feedback": "run_log_invalid",
            "failed_requirements": ["result"],
        }

    motion_events_path = run_log_dir / "motion_events.json"
    motion_events: list[Any] = []
    motion_events_error = ""
    if motion_events_path.exists():
        try:
            loaded_events = _read_run_log_json(motion_events_path)
            if isinstance(loaded_events, list):
                motion_events = loaded_events
            else:
                motion_events_error = "motion_events.json is not a list"
        except (OSError, json.JSONDecodeError) as exc:
            motion_events_error = str(exc)
    else:
        motion_events_error = "motion_events.json is missing"

    task_ok = bool(task_result.get("ok"))
    task_stage = str(task_result.get("stage", "UNKNOWN"))
    feedback = str(task_result.get("feedback", ""))
    reason = str(task_result.get("reason", ""))
    object_held = bool(task_result.get("object_held", False))
    raw_images = sorted(str(path) for path in run_log_dir.glob("*_raw.jpg"))
    annotated_images = sorted(str(path) for path in run_log_dir.glob("*_annotated.jpg"))
    request_path = run_log_dir / "request.json"
    request: Dict[str, Any] = {}
    request_error = ""
    if request_path.exists():
        try:
            loaded_request = _read_run_log_json(request_path)
            if isinstance(loaded_request, dict):
                request = loaded_request
            else:
                request_error = "request.json is not an object"
        except (OSError, json.JSONDecodeError) as exc:
            request_error = str(exc)
    else:
        request_error = "request.json is missing"
    real_grasp_request = (
        request.get("command") == "grasp"
        and request.get("dry_run") is False
    )
    plan_sequence_ok, plan_sequence_reason, plan_stages = _plan_contains_stage_sequence(
        task_result.get("plan"),
        ["FINAL_VIEW_RECHECK", "CLOSE_GRIPPER", "LIFT", "MOVE_TO_CARGO_POSE"],
    )
    arm_status_feedback_ok, arm_status_feedback_reason, status_event_count = (
        _motion_events_include_arm_status(motion_events)
    )
    close_gripper_h = _plan_close_gripper_angle(task_result.get("plan"))
    close_gripper_command_ok, close_gripper_command_reason = (
        _motion_events_include_gripper_close_command(motion_events, close_gripper_h)
    )
    final_recheck_target_ok, final_recheck_target_reason = _plan_has_final_recheck_target(
        task_result.get("plan")
    )
    final_pose_visual_ok, final_pose_visual_reason = _plan_has_final_pose_visual_compensation(
        task_result.get("plan")
    )

    requirements = [
        _validation_requirement(
            "task_ok",
            task_ok,
            reason="" if task_ok else f"task result ok is {task_result.get('ok')!r}",
        ),
        _validation_requirement(
            "stage_done",
            task_stage == "DONE",
            reason="" if task_stage == "DONE" else f"task stage is {task_stage}",
        ),
        _validation_requirement(
            "object_held",
            object_held,
            reason="" if object_held else "object_held is false",
        ),
        _validation_requirement(
            "motion_events",
            bool(motion_events),
            reason=motion_events_error or ("motion_events.json has no events" if not motion_events else ""),
            count=len(motion_events),
        ),
        _validation_requirement(
            "arm_status_feedback",
            arm_status_feedback_ok,
            reason=arm_status_feedback_reason,
            count=status_event_count,
        ),
        _validation_requirement(
            "raw_images",
            len(raw_images) >= 2,
            reason="" if len(raw_images) >= 2 else (
                "fewer than 2 raw detection images recorded; "
                "initial detection and final view recheck are both required"
            ),
            count=len(raw_images),
        ),
        _validation_requirement(
            "close_gripper_command",
            close_gripper_command_ok,
            reason=close_gripper_command_reason,
        ),
        _validation_requirement(
            "final_pose_visual_compensation",
            final_pose_visual_ok,
            reason=final_pose_visual_reason,
        ),
        _validation_requirement(
            "final_recheck_target",
            final_recheck_target_ok,
            reason=final_recheck_target_reason,
        ),
        _validation_requirement(
            "annotated_images",
            len(annotated_images) >= 2,
            reason="" if len(annotated_images) >= 2 else (
                "fewer than 2 annotated detection images recorded; "
                "initial detection and final view recheck are both required"
            ),
            count=len(annotated_images),
        ),
        _validation_requirement(
            "plan_sequence",
            plan_sequence_ok,
            reason=plan_sequence_reason,
            count=len(plan_stages),
        ),
        _validation_requirement(
            "real_grasp_request",
            real_grasp_request,
            reason="" if real_grasp_request else (
                request_error or (
                    f"request command={request.get('command')!r}, "
                    f"dry_run={request.get('dry_run')!r}"
                )
            ),
        ),
    ]
    failed_requirements = [
        requirement["name"] for requirement in requirements
        if not requirement["ok"]
    ]
    accepted = not failed_requirements
    scheme_node = _scheme_node_for_stage(task_stage)
    scheme_progress = _scheme_progress_for_stage(task_stage, accepted)
    failed_text = ", ".join(failed_requirements) if failed_requirements else "none"
    if accepted:
        summary = "运行记录通过验收：真实抓取、最终 DONE、闭合抬升、货运姿态、运动反馈和图像证据完整。"
        next_action = "可以把这次 run_log_dir 作为一次有效抓取记录保存。"
    elif isinstance(scheme_node.get("index"), int) and int(scheme_node["index"]) > 0:
        summary = (
            f"运行记录未通过验收；任务停在 {task_stage}"
            f"（方案第{scheme_node['index']}步：{scheme_node['name']}），"
            f"未满足验收项：{failed_text}。"
        )
        next_action = "先查看 diagnose-run 的失败节点、最新图像和 motion_events，再按 failed_requirements 补证据或修复流程。"
    else:
        summary = (
            f"运行记录未通过验收；任务停在 {task_stage}，"
            f"未满足验收项：{failed_text}。"
        )
        next_action = "先查看 diagnose-run 和 requirements 中每项 reason，确认是任务失败还是日志证据不完整。"
    return {
        "ok": True,
        "stage": "RUN_VALIDATION",
        "run_log_dir": str(run_log_dir),
        "accepted": accepted,
        "summary": summary,
        "next_action": next_action,
        "requirements": requirements,
        "failed_requirements": failed_requirements,
        "task_stage": task_stage,
        "feedback": feedback,
        "reason": reason,
        "object_held": object_held,
        "scheme_node": scheme_node,
        "completed_scheme_nodes": scheme_progress["completed"],
        "remaining_scheme_nodes": scheme_progress["remaining"],
        "motion_event_count": len(motion_events),
        "arm_status_feedback_found": arm_status_feedback_ok,
        "status_event_count": status_event_count,
        "plan_stages": plan_stages,
        "close_gripper_h": close_gripper_h,
        "close_gripper_command_found": close_gripper_command_ok,
        "final_pose_visual_compensation_recorded": final_pose_visual_ok,
        "final_recheck_target_recorded": final_recheck_target_ok,
        "request_command": request.get("command"),
        "request_dry_run": request.get("dry_run"),
        "raw_images": raw_images,
        "annotated_images": annotated_images,
    }


def find_latest_run_log_directory(root: Path = DEFAULT_RUN_LOG_ROOT) -> Optional[Path]:
    root = Path(root)
    if not root.exists():
        return None
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and (path / "result.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "result.json").stat().st_mtime)


def _preflight_item(
    name: str,
    ok: bool,
    *,
    reason: str = "",
    path: Optional[Any] = None,
    required: bool = True,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "name": name,
        "ok": bool(ok),
        "required": bool(required),
    }
    if reason:
        item["reason"] = reason
    if path is not None:
        item["path"] = str(path)
    return item


def _path_exists_check(name: str, path: Optional[Any], *, required: bool = True) -> Dict[str, Any]:
    if path is None:
        return _preflight_item(name, not required, reason="not provided", required=required)
    path_obj = Path(path)
    if path_obj.exists():
        return _preflight_item(name, True, path=path_obj, required=required)
    return _preflight_item(name, False, reason="path does not exist", path=path_obj, required=required)


def _read_calibration_image_size(path: Path) -> Optional[tuple[int, int]]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        calibration = json.loads(text)
    except json.JSONDecodeError:
        calibration = None
    if isinstance(calibration, Mapping):
        image_size = calibration.get("image_size")
        if (
            isinstance(image_size, (list, tuple))
            and len(image_size) == 2
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in image_size
            )
        ):
            return int(image_size[0]), int(image_size[1])
    width_match = re.search(r"(?m)^\s*image_width\s*:\s*(\d+)\s*$", text)
    height_match = re.search(r"(?m)^\s*image_height\s*:\s*(\d+)\s*$", text)
    if width_match and height_match:
        return int(width_match.group(1)), int(height_match.group(1))
    size_match = re.search(
        r"(?m)^\s*image_size\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*$",
        text,
    )
    if size_match:
        return int(size_match.group(1)), int(size_match.group(2))
    return None


def _read_config_camera_size(path: Optional[Any]) -> Optional[tuple[int, int]]:
    if path is None:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    try:
        config = json.loads(path_obj.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    camera = config.get("camera") if isinstance(config, dict) else None
    if not isinstance(camera, dict):
        return None
    width = camera.get("width")
    height = camera.get("height")
    if width is None or height is None:
        return None
    return int(width), int(height)


def _effective_camera_size(args: argparse.Namespace) -> Optional[tuple[int, int]]:
    if args.width is not None and args.height is not None:
        return int(args.width), int(args.height)
    configured_size = _read_config_camera_size(args.config)
    if configured_size is None:
        return None
    width, height = configured_size
    if args.width is not None:
        width = int(args.width)
    if args.height is not None:
        height = int(args.height)
    return width, height


def _calibration_resolution_check(args: argparse.Namespace) -> Dict[str, Any]:
    if args.calibration is None:
        return _preflight_item(
            "calibration_resolution",
            False,
            reason="calibration file not provided",
        )
    calibration_path = Path(args.calibration)
    if not calibration_path.exists():
        return _preflight_item(
            "calibration_resolution",
            False,
            reason="calibration file does not exist",
            path=calibration_path,
        )
    try:
        calibrated_size = _read_calibration_image_size(calibration_path)
    except Exception as exc:
        return _preflight_item(
            "calibration_resolution",
            False,
            reason=f"cannot read calibration size: {exc}",
            path=calibration_path,
        )
    if calibrated_size is None:
        return _preflight_item(
            "calibration_resolution",
            False,
            reason="calibration file does not include image_width/image_height",
            path=calibration_path,
        )
    camera_size = _effective_camera_size(args)
    if camera_size is None:
        return _preflight_item(
            "calibration_resolution",
            True,
            reason=f"calibrated {calibrated_size[0]}x{calibrated_size[1]}; camera size not available in args/config",
            path=calibration_path,
        )
    if camera_size != calibrated_size:
        return _preflight_item(
            "calibration_resolution",
            False,
            reason=(
                f"calibrated {calibrated_size[0]}x{calibrated_size[1]}, "
                f"requested {camera_size[0]}x{camera_size[1]}"
            ),
            path=calibration_path,
        )
    return _preflight_item(
        "calibration_resolution",
        True,
        reason=f"calibrated {calibrated_size[0]}x{calibrated_size[1]} matches requested size",
        path=calibration_path,
    )


def preflight_check(args: argparse.Namespace, *, motion: Optional[Any] = None) -> Dict[str, Any]:
    checks: list[Dict[str, Any]] = []
    checks.append(
        _preflight_item(
            "port",
            bool(args.port),
            reason="" if args.port else "--port not provided",
            path=args.port,
        )
    )
    checks.append(_path_exists_check("device", args.device))
    checks.append(_path_exists_check("calibration", args.calibration))
    checks.append(_calibration_resolution_check(args))
    checks.append(_path_exists_check("config", args.config))

    try:
        _load_grasp_reference(args.reference)
    except Exception as exc:
        checks.append(
            _preflight_item(
                "reference",
                False,
                reason=str(exc),
                path=args.reference,
            )
        )
    else:
        checks.append(_preflight_item("reference", True, path=args.reference))

    arm_status: Optional[Dict[str, Any]] = None
    if args.dry_run:
        checks.append(
            _preflight_item(
                "serial_status",
                True,
                reason="skipped in dry-run",
                required=False,
            )
        )
    elif motion is None:
        checks.append(
            _preflight_item(
                "serial_status",
                False,
                reason="not checked",
                required=True,
            )
        )
    else:
        try:
            arm_status = dict(motion.status())
        except Exception as exc:
            checks.append(_preflight_item("serial_status", False, reason=str(exc)))
        else:
            checks.append(_preflight_item("serial_status", True))

    required_failures = [
        check for check in checks
        if check.get("required", True) and not check.get("ok")
    ]
    ok = not required_failures
    result: Dict[str, Any] = {
        "ok": ok,
        "stage": "PREFLIGHT",
        "feedback": "preflight_ok" if ok else "preflight_failed",
        "reason": "" if ok else "; ".join(
            f"{check['name']}: {check.get('reason', 'failed')}" for check in required_failures
        ),
        "checks": checks,
    }
    if arm_status is not None:
        result["arm_status"] = arm_status
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_alignment_defaults(args)
    if args.command == "diagnose-run":
        directory = args.directory
        if directory is None:
            directory = find_latest_run_log_directory(DEFAULT_RUN_LOG_ROOT)
        if directory is None:
            result = {
                "ok": False,
                "stage": "RUN_DIAGNOSIS",
                "reason": f"no run logs with result.json found under: {DEFAULT_RUN_LOG_ROOT}",
                "feedback": "run_log_missing",
            }
        else:
            result = diagnose_run_log(directory)
        _write_and_print_result(args, result)
        return 0 if result.get("ok") else 1
    if args.command == "validate-run":
        directory = args.directory
        if directory is None:
            directory = find_latest_run_log_directory(DEFAULT_RUN_LOG_ROOT)
        if directory is None:
            result = {
                "ok": False,
                "stage": "RUN_VALIDATION",
                "accepted": False,
                "reason": f"no run logs with result.json found under: {DEFAULT_RUN_LOG_ROOT}",
                "feedback": "run_log_missing",
                "failed_requirements": ["result"],
            }
        else:
            result = validate_run_log(directory)
        _write_and_print_result(args, result)
        return 0 if result.get("accepted") else 1
    if args.command == "preflight":
        motion = None
        if args.port and not args.dry_run:
            try:
                motion = ArmTestSerialMotion(
                    port=args.port,
                    baudrate=args.baud,
                    timeout=args.timeout,
                )
            except Exception as exc:
                reason = f"serial initialization failed: {exc}"
                serial_check = _preflight_item(
                    "serial_status",
                    False,
                    reason=reason,
                    path=args.port,
                )
                result = {
                    "ok": False,
                    "stage": "PREFLIGHT",
                    "feedback": "preflight_failed",
                    "reason": reason,
                    "checks": [serial_check],
                    "failed_checks": [serial_check],
                    "object_held": False,
                }
                _write_and_print_result(args, result)
                return 1
        result = preflight_check(args, motion=motion)
        _write_and_print_result(args, result)
        return 0 if result.get("ok") else 1

    run_log_directory = _resolve_run_log_directory(args)
    if not args.dry_run and not args.port:
        reason = "real motion requires --port; use --dry-run for planning only"
        if args.command == "grasp":
            port_check = _preflight_item(
                "port",
                False,
                reason="--port not provided",
            )
            result = {
                "ok": False,
                "stage": "PREFLIGHT",
                "feedback": "preflight_failed",
                "reason": reason,
                "checks": [port_check],
                "failed_checks": [port_check],
                "object_held": False,
            }
        else:
            result = _failure_result("CONFIG", reason)
        if run_log_directory is not None:
            _write_run_log(
                run_log_directory,
                args=args,
                result=result,
                motion=DryRunMotion(),
                vision=arm_grasp.StaticVision(),
            )
        _write_and_print_result(args, result)
        return 1 if args.command == "grasp" else 2

    try:
        motion = DryRunMotion() if args.dry_run else ArmTestSerialMotion(
            port=args.port,
            baudrate=args.baud,
            timeout=args.timeout,
        )
    except Exception as exc:
        reason = f"serial initialization failed: {exc}"
        if args.command == "grasp":
            serial_check = _preflight_item(
                "serial_status",
                False,
                reason=reason,
                path=args.port,
            )
            result = {
                "ok": False,
                "stage": "PREFLIGHT",
                "feedback": "preflight_failed",
                "reason": reason,
                "checks": [serial_check],
                "failed_checks": [serial_check],
                "object_held": False,
            }
        else:
            result = _failure_result("ARM_CONTROL", reason)
        if run_log_directory is not None:
            _write_run_log(
                run_log_directory,
                args=args,
                result=result,
                motion=DryRunMotion(),
                vision=arm_grasp.StaticVision(),
            )
        _write_and_print_result(args, result)
        return 1
    if args.command == "grasp" and (args.require_preflight or not args.dry_run):
        result = preflight_check(args, motion=motion)
        if not result.get("ok"):
            if run_log_directory is not None:
                _write_run_log(
                    run_log_directory,
                    args=args,
                    result=result,
                    motion=motion,
                    vision=arm_grasp.StaticVision(),
                )
            _write_and_print_result(args, result)
            return 1
    try:
        vision, closeable_vision = _build_grasp_vision(args)
        place_vision, closeable_place_vision = _build_place_vision(args)
    except KeyboardInterrupt:
        result = _aborted_result()
        vision = arm_grasp.StaticVision()
        place_vision = None
        closeable_place_vision = None
        if run_log_directory is not None:
            _write_run_log(
                run_log_directory,
                args=args,
                result=result,
                motion=motion,
                vision=vision,
            )
        _write_and_print_result(args, result)
        return 1
    except Exception as exc:
        result = _failure_result("ARM_CONTROL", str(exc))
        vision = arm_grasp.StaticVision()
        place_vision = None
        closeable_place_vision = None
        if run_log_directory is not None:
            _write_run_log(
                run_log_directory,
                args=args,
                result=result,
                motion=motion,
                vision=vision,
            )
        _write_and_print_result(args, result)
        return 1
    abort_watcher = TerminalAbortWatcher(
        enabled=args.show_vision and args.command == "grasp",
    )
    result: Dict[str, Any] = _failure_result(
        "ARM_CONTROL",
        "task did not start",
    )
    try:
        with abort_watcher:
            set_abort_checker = getattr(vision, "set_abort_checker", None)
            if callable(set_abort_checker):
                set_abort_checker(abort_watcher.abort_requested)
            set_place_abort_checker = getattr(place_vision, "set_abort_checker", None)
            if callable(set_place_abort_checker):
                set_place_abort_checker(abort_watcher.abort_requested)
            set_run_log_directory = getattr(vision, "set_run_log_directory", None)
            if run_log_directory is not None and callable(set_run_log_directory):
                set_run_log_directory(run_log_directory)
            set_place_run_log_directory = getattr(
                place_vision,
                "set_run_log_directory",
                None,
            )
            if run_log_directory is not None and callable(set_place_run_log_directory):
                set_place_run_log_directory(run_log_directory)
            task = ArmTask(
                motion=motion,
                vision=vision,
                place_vision=place_vision,
                grasp_reference=_load_grasp_reference(args.reference),
                dry_run=args.dry_run,
                single_step=args.single_step,
                max_align_steps=args.max_align_steps,
                max_jog_deg=args.max_jog_deg,
                place_reference=_load_place_reference(args.place_reference),
                spd=args.spd,
                acc=args.acc,
                final_spd=args.final_spd,
                final_acc=args.final_acc,
                skip_grasp_ready=args.skip_grasp_ready,
                stop_after_final_pose=args.stop_after_final_pose,
                abort_checker=abort_watcher.abort_requested,
            )
            try:
                if args.command == "grasp":
                    result = task.grasp()
                elif args.command in ("grasp-ready", "transport"):
                    result = task.grasp_ready()
                elif args.command == "close":
                    result = task.close(args.angle)
                elif args.command == "hold":
                    result = task.hold()
                elif args.command == "place":
                    if args.dry_run or args.object_held:
                        task.object_held = True
                    result = task.place(args.slot)
                elif args.command == "home":
                    result = task.home()
                elif args.command == "status":
                    result = task.status()
                elif args.command == "abort":
                    result = task.abort()
                else:
                    raise RuntimeError(f"unknown command: {args.command}")
            except KeyboardInterrupt:
                abort_watcher.request_abort()
                result = _aborted_result()
            except Exception as exc:
                result = _failure_result("ARM_CONTROL", str(exc))

            should_hold_window = (
                args.show_vision
                and args.hold_vision
                and result.get("stage") != "ABORTED"
                and not abort_watcher.abort_requested()
                and hasattr(closeable_vision, "hold_window_until_closed")
            )
            if should_hold_window:
                try:
                    closeable_vision.hold_window_until_closed()
                except KeyboardInterrupt:
                    abort_watcher.request_abort()
                    result = _aborted_result()
    except KeyboardInterrupt:
        abort_watcher.request_abort()
        result = _aborted_result()
    except Exception as exc:
        result = _failure_result("ARM_CONTROL", str(exc))
    finally:
        if closeable_vision is not None:
            closeable_vision.close()
        if closeable_place_vision is not None:
            closeable_place_vision.close()

    if run_log_directory is not None:
        _write_run_log(
            run_log_directory,
            args=args,
            result=result,
            motion=motion,
            vision=place_vision if args.command == "place" and place_vision is not None else vision,
        )
    _write_and_print_result(args, result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
