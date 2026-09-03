#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TLS机械臂串口安全测试工具。

示例：
  python3 test.py --port /dev/ttyUSB0 status
  python3 test.py --port /dev/ttyUSB0 light 255
  python3 test.py --port /dev/ttyUSB0 gripper open
  python3 test.py --port /dev/ttyUSB0 gripper close
  python3 test.py --port /dev/ttyUSB0 jog b 2
  python3 test.py --port /dev/ttyUSB0 jog b 2 --execute
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError:  # pyserial is only needed when talking to real hardware.
    serial = None


class StatusParseError(RuntimeError):
    def __init__(self, message: str, raw_response: bytes):
        super().__init__(message)
        self.raw_response = raw_response


class StatusQueryError(RuntimeError):
    def __init__(self, message: str, raw_responses: Iterable[bytes]):
        super().__init__(message)
        self.raw_responses = list(raw_responses)


DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 2.0
STATUS_QUERY_RETRY_DELAY_SECONDS = 0.2
STATUS_QUERY_RETRY_TIMEOUT_STEP_SECONDS = 0.25
JOINT_KEYS = ("b", "s", "e", "w")
JOINT_MAX_JOG_DEGREES = {
    "b": 20.0,
    "s": 20.0,
    "e": 20.0,
    "w": 20.0,
}
SAFE_JOG_SPEED = 3.0
SAFE_JOG_ACCELERATION = 3.0
SHOULDER_REFERENCE_DEGREES = -1.67
# 反馈坐标系下记录的整体复位姿态（度），由实测status换算得到。
HOME_REFERENCE_DEGREES = {
    "b": -0.70,
    "s": -0.44,
    "e": 4.13,
    "w": -0.62,
}
POSE_SCHEMA_VERSION = 1
MOVING_POSE_NAME = "moving_pose"
MOVING_POSE_FILE = Path(__file__).with_name("moving_pose.json")
MOVING_POSE_SPEED = 5.0
MOVING_POSE_ACCELERATION = 5.0
# 复位默认速度高于jog：肘关节向上抬需对抗前臂重力，spd=3驱动力不足。
HOME_SPEED = 10.0
HOME_ACCELERATION = 10.0
MOTION_WAIT_TIMEOUT_SECONDS = 15.0
MOTION_START_TIMEOUT_SECONDS = 2.0
MOTION_POLL_SECONDS = 0.3
MINIMUM_JOINT_TARGET_TOLERANCE_DEGREES = 2.0
TARGET_TOLERANCE_DEGREES = MINIMUM_JOINT_TARGET_TOLERANCE_DEGREES
JOINT_STABILITY_TOLERANCE_DEGREES = 0.25
JOINT_STABLE_SAMPLES = 3
PRE_COMMAND_STABILITY_TIMEOUT_SECONDS = 3.0
PRE_COMMAND_IDLE_TIMEOUT_SECONDS = 3.0
PRE_COMMAND_IDLE_SAMPLES = 2
STOPPED_STABLE_SAMPLES = 2
MAX_CARTESIAN_JOG_UNITS = 5.0
SAFE_CARTESIAN_SPEED = 0.1
CARTESIAN_TARGET_TOLERANCE = 2.0
CARTESIAN_EXECUTION_DISABLED_REASON = (
    "T104已停用：实测相同z增量会产生相反运动方向，"
    "且T105反馈坐标不能直接作为T104目标"
)
GRIPPER_TARGETS = {
    "open": -45,
    "mid": 0,
    "close": 45,
}


def compact_json(command: Dict[str, Any]) -> str:
    return json.dumps(command, separators=(",", ":"), ensure_ascii=False)


def build_light_command(led: int) -> Dict[str, int]:
    if led < 0 or led > 255:
        raise ValueError("灯光亮度必须在0到255之间")
    return {"T": 114, "led": int(led)}


def build_status_command() -> Dict[str, int]:
    return {"T": 105}


def status_to_joint_degrees(status: Dict[str, Any]) -> Dict[str, float]:
    joints: Dict[str, float] = {}
    for key in JOINT_KEYS:
        if key not in status:
            raise ValueError(f"状态返回中缺少关节字段: {key}")
        value = float(status[key])
        if not math.isfinite(value):
            raise ValueError(f"状态返回中的{key}不是有效数字")
        joints[key] = math.degrees(value)
    return joints


def status_to_command_degrees(status: Dict[str, Any]) -> Dict[str, float]:
    joints = status_to_joint_degrees(status)
    joints["s"] = -joints["s"]
    return joints


def status_to_gripper_degrees(status: Dict[str, Any]) -> float:
    if "t" not in status:
        raise ValueError("状态返回中缺少夹爪字段: t")
    value = float(status["t"])
    if not math.isfinite(value):
        raise ValueError("状态返回中的t不是有效数字")
    return math.degrees(value)


def command_to_feedback_degrees(joint: str, command_degrees: float) -> float:
    if joint not in JOINT_KEYS:
        raise ValueError(f"未知关节: {joint}")
    return -command_degrees if joint == "s" else command_degrees


def max_joint_change_degrees(
    previous_status: Dict[str, Any],
    current_status: Dict[str, Any],
) -> float:
    previous = status_to_joint_degrees(previous_status)
    current = status_to_joint_degrees(current_status)
    changes = [abs(current[key] - previous[key]) for key in JOINT_KEYS]
    if "t" in previous_status or "t" in current_status:
        changes.append(
            abs(
                status_to_gripper_degrees(current_status)
                - status_to_gripper_degrees(previous_status)
            )
        )
    return max(changes)


def wait_for_motion_idle(
    query_status_fn: Callable[[], Dict[str, Any]],
    initial_status: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = PRE_COMMAND_IDLE_TIMEOUT_SECONDS,
    poll_seconds: float = MOTION_POLL_SECONDS,
    idle_samples: int = PRE_COMMAND_IDLE_SAMPLES,
    tolerance_degrees: float = JOINT_STABILITY_TOLERANCE_DEGREES,
    stable_samples: int = JOINT_STABLE_SAMPLES,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    if idle_samples < 1:
        raise ValueError("空闲采样数必须至少为1")

    if stable_samples < 2:
        raise ValueError("stable_samples must be at least 2")

    deadline = now_fn() + timeout_seconds
    current_status = initial_status if initial_status is not None else query_status_fn()
    idle_count = 0
    stable_count = 0
    last_move: Any = None
    last_change = math.inf
    previous_status: Optional[Dict[str, Any]] = None
    require_gripper = "t" in current_status

    while True:
        if "move" not in current_status:
            raise ValueError("状态返回中缺少运动字段: move")
        try:
            status_to_joint_degrees(current_status)
            if require_gripper:
                status_to_gripper_degrees(current_status)
        except (TypeError, ValueError) as exc:
            if previous_status is None:
                raise
            if now_fn() >= deadline:
                raise RuntimeError("mechanical arm status stayed incomplete") from exc
            sleep_fn(poll_seconds)
            current_status = query_status_fn()
            continue
        if "t" in current_status:
            require_gripper = True
        last_move = current_status["move"]
        idle_count = idle_count + 1 if last_move == 0 else 0
        if previous_status is None:
            status_to_joint_degrees(current_status)
            stable_count = 1
        else:
            last_change = max_joint_change_degrees(previous_status, current_status)
            stable_count = stable_count + 1 if last_change <= tolerance_degrees else 1
        if idle_count >= idle_samples or stable_count >= stable_samples:
            return current_status
        if now_fn() >= deadline:
            raise RuntimeError(f"机械臂未进入空闲状态，move仍为{last_move}")
        previous_status = current_status
        sleep_fn(poll_seconds)
        current_status = query_status_fn()


def wait_for_joint_stability(
    query_status_fn: Callable[[], Dict[str, Any]],
    initial_status: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = PRE_COMMAND_STABILITY_TIMEOUT_SECONDS,
    poll_seconds: float = MOTION_POLL_SECONDS,
    tolerance_degrees: float = JOINT_STABILITY_TOLERANCE_DEGREES,
    stable_samples: int = JOINT_STABLE_SAMPLES,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    if stable_samples < 2:
        raise ValueError("稳定采样数必须至少为2")

    deadline = now_fn() + timeout_seconds
    previous_status = initial_status if initial_status is not None else query_status_fn()
    status_to_joint_degrees(previous_status)
    stable_count = 1
    last_change = math.inf

    while stable_count < stable_samples:
        if now_fn() >= deadline:
            raise RuntimeError(
                f"机械臂关节反馈仍在变化，最大变化={last_change:.2f}度"
            )
        sleep_fn(poll_seconds)
        current_status = query_status_fn()
        last_change = max_joint_change_degrees(previous_status, current_status)
        stable_count = stable_count + 1 if last_change <= tolerance_degrees else 1
        previous_status = current_status

    return previous_status


def require_cartesian_execution_enabled() -> None:
    raise RuntimeError(CARTESIAN_EXECUTION_DISABLED_REASON)


def wait_for_joint_target(
    query_status_fn: Callable[[], Dict[str, Any]],
    joint: str,
    target_degrees: float,
    initial_status: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = MOTION_WAIT_TIMEOUT_SECONDS,
    motion_start_timeout_seconds: float = MOTION_START_TIMEOUT_SECONDS,
    poll_seconds: float = MOTION_POLL_SECONDS,
    tolerance_degrees: float = TARGET_TOLERANCE_DEGREES,
    stable_samples: int = JOINT_STABLE_SAMPLES,
    stability_tolerance_degrees: float = JOINT_STABILITY_TOLERANCE_DEGREES,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    if joint not in JOINT_KEYS:
        raise ValueError(f"未知关节: {joint}")

    target_feedback_degrees = command_to_feedback_degrees(joint, target_degrees)
    initial_error: Optional[float] = None
    effective_tolerance = max(
        float(tolerance_degrees),
        MINIMUM_JOINT_TARGET_TOLERANCE_DEGREES,
    )
    if initial_status is not None:
        initial_joints = status_to_joint_degrees(initial_status)
        initial_error = abs(initial_joints[joint] - target_feedback_degrees)

    start_time = now_fn()
    deadline = start_time + timeout_seconds
    stable_count = 0
    last_status: Optional[Dict[str, Any]] = None
    previous_status = initial_status
    movement_observed = False
    last_error = math.inf
    last_change = math.inf

    while True:
        last_status = query_status_fn()
        joints = status_to_joint_degrees(last_status)
        last_error = abs(joints[joint] - target_feedback_degrees)
        if previous_status is None:
            stable_count = 1
        else:
            last_change = max_joint_change_degrees(previous_status, last_status)
            stable_count = (
                stable_count + 1
                if stable_count > 0
                and last_change <= stability_tolerance_degrees
                else 1
            )
        previous_status = last_status

        if initial_status is not None:
            movement_observed = movement_observed or (
                max_joint_change_degrees(initial_status, last_status)
                > stability_tolerance_degrees
            )

        made_progress = (
            last_error < effective_tolerance
            if initial_error is None
            else last_error < initial_error
        )
        if stable_count >= stable_samples:
            if last_error <= effective_tolerance and made_progress:
                return last_status
            if movement_observed and initial_error is not None:
                if last_error >= initial_error:
                    raise RuntimeError(
                        f"{joint}关节已停止但未接近目标："
                        f"初始误差={initial_error:.2f}度，最终误差={last_error:.2f}度"
                    )
                raise RuntimeError(
                    f"{joint}关节已停止但未到达目标："
                    f"允许误差={effective_tolerance:.2f}度，"
                    f"最终误差={last_error:.2f}度"
                )
            if (
                initial_status is not None
                and not movement_observed
                and now_fn() - start_time >= motion_start_timeout_seconds
            ):
                raise RuntimeError(
                    f"{joint}关节未开始运动：目标={target_feedback_degrees:.2f}度，"
                    f"当前误差={last_error:.2f}度"
                )

        if now_fn() >= deadline:
            if last_status is not None and last_error <= effective_tolerance and made_progress:
                return last_status
            move = last_status.get("move", "缺失")
            raise RuntimeError(
                f"{joint}关节未到达目标：move={move}，"
                f"命令目标={target_degrees:.2f}度，"
                f"反馈目标={target_feedback_degrees:.2f}度，误差={last_error:.2f}度，"
                f"允许误差={effective_tolerance:.2f}度，"
                f"最大变化={last_change:.2f}度"
            )
        sleep_fn(poll_seconds)


def status_to_cartesian(status: Dict[str, Any]) -> Dict[str, float]:
    coordinates: Dict[str, float] = {}
    for axis in ("x", "y", "z"):
        if axis not in status:
            raise ValueError(f"状态返回中缺少坐标字段: {axis}")
        value = float(status[axis])
        if not math.isfinite(value):
            raise ValueError(f"状态返回中的{axis}不是有效数字")
        coordinates[axis] = value
    return coordinates


def build_cartesian_jog_command(
    axis: str,
    delta: float,
    status: Dict[str, Any],
    spd: float = SAFE_CARTESIAN_SPEED,
) -> Dict[str, Any]:
    if axis != "z":
        raise ValueError("当前安全测试只允许z轴")
    if not math.isfinite(delta) or delta == 0:
        raise ValueError("坐标增量必须是非零有效数字")
    if abs(delta) > MAX_CARTESIAN_JOG_UNITS:
        raise ValueError(f"单次坐标微动不得超过{MAX_CARTESIAN_JOG_UNITS:g}个单位")
    if not math.isfinite(spd) or spd <= 0:
        raise ValueError("坐标运动速度必须大于0")

    coordinates = status_to_cartesian(status)
    coordinates[axis] += delta
    return {
        "T": 104,
        **coordinates,
        "spd": spd,
    }


def wait_for_cartesian_target(
    query_status_fn: Callable[[], Dict[str, Any]],
    axis: str,
    target_value: float,
    timeout_seconds: float = MOTION_WAIT_TIMEOUT_SECONDS,
    poll_seconds: float = MOTION_POLL_SECONDS,
    tolerance: float = CARTESIAN_TARGET_TOLERANCE,
    stable_samples: int = STOPPED_STABLE_SAMPLES,
    now_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    if axis != "z":
        raise ValueError("当前安全测试只允许z轴")

    deadline = now_fn() + timeout_seconds
    stopped_samples = 0
    last_status: Optional[Dict[str, Any]] = None
    last_error = math.inf

    while True:
        last_status = query_status_fn()
        coordinates = status_to_cartesian(last_status)
        last_error = abs(coordinates[axis] - target_value)
        if int(last_status.get("move", -1)) == 0 and last_error <= tolerance:
            stopped_samples += 1
            if stopped_samples >= stable_samples:
                return last_status
        else:
            stopped_samples = 0

        if now_fn() >= deadline:
            move = last_status.get("move", "缺失")
            raise RuntimeError(
                f"{axis}轴未到达目标：move={move}，"
                f"目标={target_value:.2f}，误差={last_error:.2f}"
            )
        sleep_fn(poll_seconds)


def build_gripper_command(
    action: str,
    status: Dict[str, Any],
    spd: float = 10,
    acc: float = 10,
) -> Dict[str, Any]:
    if action not in GRIPPER_TARGETS:
        raise ValueError(f"未知夹爪动作: {action}")
    return build_gripper_angle_command(
        GRIPPER_TARGETS[action],
        status,
        spd=spd,
        acc=acc,
    )


def build_gripper_angle_command(
    angle: float,
    status: Dict[str, Any],
    spd: float = 10,
    acc: float = 10,
) -> Dict[str, Any]:
    if not math.isfinite(float(angle)):
        raise ValueError("夹爪角度必须是有效数字")
    min_angle = min(GRIPPER_TARGETS.values())
    max_angle = max(GRIPPER_TARGETS.values())
    if not min_angle <= float(angle) <= max_angle:
        raise ValueError(f"夹爪角度必须在{min_angle}到{max_angle}度之间")
    command: Dict[str, Any] = {"T": 122}
    command.update(status_to_command_degrees(status))
    command.update({"h": float(angle), "spd": spd, "acc": acc})
    return command


def build_jog_command(
    joint: str,
    delta: float,
    status: Dict[str, Any],
    spd: float = SAFE_JOG_SPEED,
    acc: float = SAFE_JOG_ACCELERATION,
    unsafe: bool = False,
) -> Dict[str, Any]:
    if joint not in JOINT_KEYS:
        raise ValueError(f"未知关节: {joint}")
    if not math.isfinite(delta) or delta == 0:
        raise ValueError("单步角度必须是非零有效数字")
    max_jog = JOINT_MAX_JOG_DEGREES[joint]
    if not unsafe and abs(delta) > max_jog:
        raise ValueError(f"{joint}关节单次微动不得超过{max_jog:g}度")

    joints = status_to_command_degrees(status)
    joints[joint] += delta
    command: Dict[str, Any] = {"T": 122}
    command.update(joints)
    command.update(
        {
            "h": status_to_gripper_degrees(status),
            "spd": spd,
            "acc": acc,
        }
    )
    return command


def build_restore_shoulder_command(
    status: Dict[str, Any],
    spd: float = SAFE_JOG_SPEED,
    acc: float = SAFE_JOG_ACCELERATION,
) -> Dict[str, Any]:
    joints = status_to_command_degrees(status)
    joints["s"] = -SHOULDER_REFERENCE_DEGREES
    command: Dict[str, Any] = {"T": 122}
    command.update(joints)
    command.update(
        {
            "h": status_to_gripper_degrees(status),
            "spd": spd,
            "acc": acc,
        }
    )
    return command


def build_home_command(
    status: Dict[str, Any],
    spd: float = HOME_SPEED,
    acc: float = HOME_ACCELERATION,
) -> Dict[str, Any]:
    joints = {
        joint: command_to_feedback_degrees(joint, HOME_REFERENCE_DEGREES[joint])
        for joint in JOINT_KEYS
    }
    command: Dict[str, Any] = {"T": 122}
    command.update(joints)
    command.update(
        {
            "h": status_to_gripper_degrees(status),
            "spd": spd,
            "acc": acc,
        }
    )
    return command


def pose_record_from_status(
    status: Dict[str, Any],
    name: str = MOVING_POSE_NAME,
) -> Dict[str, Any]:
    return {
        "schema_version": POSE_SCHEMA_VERSION,
        "name": name,
        "joints_feedback_deg": status_to_joint_degrees(status),
        "gripper_deg": status_to_gripper_degrees(status),
    }


def validate_pose_record(record: Dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise ValueError("姿态记录必须是JSON对象")
    if int(record.get("schema_version", -1)) != POSE_SCHEMA_VERSION:
        raise ValueError(f"姿态记录版本必须是{POSE_SCHEMA_VERSION}")
    joints = record.get("joints_feedback_deg")
    if not isinstance(joints, dict):
        raise ValueError("姿态记录缺少joints_feedback_deg")
    for joint in JOINT_KEYS:
        if joint not in joints:
            raise ValueError(f"姿态记录缺少{joint}关节")
        value = float(joints[joint])
        if not math.isfinite(value):
            raise ValueError(f"姿态记录中的{joint}关节不是有效数字")
    gripper = float(record.get("gripper_deg"))
    if not math.isfinite(gripper):
        raise ValueError("姿态记录中的夹爪角度不是有效数字")


def feedback_pose_to_command_degrees(joints_feedback: Dict[str, Any]) -> Dict[str, float]:
    joints: Dict[str, float] = {}
    for joint in JOINT_KEYS:
        feedback_degrees = float(joints_feedback[joint])
        joints[joint] = -feedback_degrees if joint == "s" else feedback_degrees
    return joints


def build_pose_command(
    record: Dict[str, Any],
    spd: float = MOVING_POSE_SPEED,
    acc: float = MOVING_POSE_ACCELERATION,
) -> Dict[str, Any]:
    validate_pose_record(record)
    if spd <= 0 or acc <= 0:
        raise ValueError("姿态运动速度和加速度必须大于0")
    command: Dict[str, Any] = {"T": 122}
    command.update(feedback_pose_to_command_degrees(record["joints_feedback_deg"]))
    command.update({"h": float(record["gripper_deg"]), "spd": spd, "acc": acc})
    return command


def write_pose_record(path: Path, record: Dict[str, Any]) -> None:
    validate_pose_record(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_pose_record(path: Path) -> Dict[str, Any]:
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"未找到移动姿态文件: {path}，请先把机械臂摆到照片姿态后运行 teach-moving-pose"
        ) from exc
    validate_pose_record(record)
    return record


def default_port_name() -> str:
    env_port = os.environ.get("ARM_PORT")
    if env_port:
        return env_port
    if platform.system().lower().startswith("win"):
        return "COM7"
    return "/dev/ttyUSB0"


def require_pyserial() -> None:
    if serial is None:
        raise SystemExit("缺少pyserial，请先执行: pip3 install pyserial")


def list_ports() -> None:
    require_pyserial()
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("未检测到串口设备")
        return
    for port in ports:
        description = port.description or ""
        hwid = port.hwid or ""
        print(f"{port.device}\t{description}\t{hwid}")


def read_response(
    ser: Any,
    wait_seconds: float,
    *,
    stop_on_json: bool = False,
    poll_seconds: float = 0.05,
) -> bytes:
    deadline = time.time() + wait_seconds
    chunks = []
    while time.time() < deadline:
        waiting = getattr(ser, "in_waiting", 0)
        if waiting:
            chunks.append(ser.read(waiting))
            data = b"".join(chunks)
            if stop_on_json and (
                response_has_json_line(data) or data.endswith((b"\n", b"\r"))
            ):
                return data
        if poll_seconds > 0:
            time.sleep(poll_seconds)
    waiting = getattr(ser, "in_waiting", 0)
    if waiting:
        chunks.append(ser.read(waiting))
    return b"".join(chunks)


def parse_first_json_line(data: bytes) -> Dict[str, Any]:
    text = data.decode("utf-8", errors="replace")
    decoder = json.JSONDecoder()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            for index, char in enumerate(line):
                if char != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(line[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    return value
            continue
        if isinstance(value, dict):
            return value
    raise StatusParseError(f"没有解析到JSON返回: {text!r}", data)


def response_has_json_line(data: bytes) -> bool:
    try:
        parse_first_json_line(data)
    except RuntimeError:
        return False
    return True


def send_serial_command(
    port_name: str,
    command: Dict[str, Any],
    baudrate: int = DEFAULT_BAUD,
    timeout: float = DEFAULT_TIMEOUT,
    read_seconds: float = 0.0,
    stop_on_json: bool = False,
) -> bytes:
    require_pyserial()
    line = compact_json(command)
    with serial.Serial(port_name, baudrate=baudrate, timeout=timeout) as ser:
        ser.reset_input_buffer()
        ser.write((line + "\n").encode("utf-8"))
        ser.flush()
        print(f"已发送命令: {line}")
        if read_seconds > 0:
            return read_response(ser, read_seconds, stop_on_json=stop_on_json)
    return b""


def query_status(port_name: str, baudrate: int, timeout: float) -> Dict[str, Any]:
    last_error: Optional[RuntimeError] = None
    raw_responses = []
    for attempt in range(3):
        attempt_timeout = timeout + attempt * STATUS_QUERY_RETRY_TIMEOUT_STEP_SECONDS
        data = send_serial_command(
            port_name,
            build_status_command(),
            baudrate=baudrate,
            timeout=attempt_timeout,
            read_seconds=attempt_timeout,
            stop_on_json=True,
        )
        raw_responses.append(data)
        if data:
            print("原始返回:", data.decode("utf-8", errors="replace").strip())
        try:
            return parse_first_json_line(data)
        except RuntimeError as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(STATUS_QUERY_RETRY_DELAY_SECONDS)
    raise StatusQueryError(
        "机械臂状态连续3次无法解析JSON返回",
        raw_responses,
    ) from last_error


def print_json(value: Dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TLS机械臂串口安全测试工具")
    parser.add_argument("--port", default=default_port_name(), help="串口名，Ubuntu常见为/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="波特率，默认115200")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="串口读取超时秒数")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="列出当前系统识别到的串口")
    subparsers.add_parser("status", help="查询机械臂状态")

    light_parser = subparsers.add_parser("light", help="设置灯光亮度，范围0-255")
    light_parser.add_argument("led", type=int)

    gripper_parser = subparsers.add_parser("gripper", help="夹爪开合，默认先查询当前姿态并保持其他关节不变")
    gripper_parser.add_argument("action", choices=sorted(GRIPPER_TARGETS))
    gripper_parser.add_argument(
        "--angle",
        type=float,
        help="指定夹爪目标角度，覆盖action对应的默认角度",
    )
    gripper_parser.add_argument("--spd", type=float, default=10)
    gripper_parser.add_argument("--acc", type=float, default=10)
    gripper_parser.add_argument(
        "--zero-pose",
        action="store_true",
        help="不查询当前姿态，直接使用b/s/e/w=0；只有确认空间安全时再用",
    )

    jog_parser = subparsers.add_parser(
        "jog",
        help="单关节小角度微动；默认只显示命令，添加--execute才实际运动",
    )
    jog_parser.add_argument("joint", choices=JOINT_KEYS, help="要微动的关节")
    jog_parser.add_argument(
        "delta",
        type=float,
        help="命令角度增量：b/w最多2度，s/e最多5度，且不能为0",
    )
    jog_parser.add_argument(
        "--execute",
        action="store_true",
        help="实际发送运动命令；不添加时只预览",
    )
    jog_parser.add_argument(
        "--unsafe",
        action="store_true",
        help="允许单次微动超过默认5度上限；仍需添加--execute才实际运动",
    )
    jog_parser.add_argument(
        "--spd",
        type=float,
        default=SAFE_JOG_SPEED,
        help=f"关节速度，单位度/秒，默认{SAFE_JOG_SPEED:g}",
    )
    jog_parser.add_argument(
        "--acc",
        type=float,
        default=SAFE_JOG_ACCELERATION,
        help=f"关节加速度，单位度/秒²，默认{SAFE_JOG_ACCELERATION:g}",
    )

    xyz_jog_parser = subparsers.add_parser(
        "xyz-jog",
        help="末端坐标小步运动；当前只开放z轴，默认只预览",
    )
    xyz_jog_parser.add_argument("axis", choices=("z",), help="当前仅允许z轴")
    xyz_jog_parser.add_argument(
        "delta",
        type=float,
        help=f"相对坐标增量，最大正负{MAX_CARTESIAN_JOG_UNITS:g}个控制器单位",
    )
    xyz_jog_parser.add_argument(
        "--spd",
        type=float,
        default=SAFE_CARTESIAN_SPEED,
        help=f"坐标运动速度，默认{SAFE_CARTESIAN_SPEED:g}",
    )
    xyz_jog_parser.add_argument(
        "--execute",
        action="store_true",
        help="实际发送运动命令；不添加时只预览",
    )

    restore_shoulder_parser = subparsers.add_parser(
        "restore-shoulder",
        help=f"将肩关节恢复到记录基准{SHOULDER_REFERENCE_DEGREES:.2f}度；默认只预览",
    )
    restore_shoulder_parser.add_argument(
        "--execute",
        action="store_true",
        help="实际发送复位命令；不添加时只预览",
    )

    home_parser = subparsers.add_parser(
        "home",
        help="将b/s/e/w四个关节整体复位到记录的复位姿态；默认只预览",
    )
    home_parser.add_argument(
        "--spd",
        type=float,
        default=HOME_SPEED,
        help=f"关节速度，单位度/秒，默认{HOME_SPEED:g}",
    )
    home_parser.add_argument(
        "--acc",
        type=float,
        default=HOME_ACCELERATION,
        help=f"关节加速度，单位度/秒²，默认{HOME_ACCELERATION:g}",
    )
    home_parser.add_argument(
        "--execute",
        action="store_true",
        help="实际发送复位命令；不添加时只预览",
    )

    teach_moving_pose_parser = subparsers.add_parser(
        "teach-moving-pose",
        help="把当前机械臂状态保存为移动姿态；请先人工摆到照片中的安全移动姿态",
    )
    teach_moving_pose_parser.add_argument(
        "--file",
        default=str(MOVING_POSE_FILE),
        help=f"移动姿态保存路径，默认{MOVING_POSE_FILE}",
    )

    moving_pose_parser = subparsers.add_parser(
        "moving-pose",
        aliases=("move-pose",),
        help="移动到已保存的移动姿态；默认只预览，添加--execute才实际运动",
    )
    moving_pose_parser.add_argument(
        "--file",
        default=str(MOVING_POSE_FILE),
        help=f"移动姿态文件路径，默认{MOVING_POSE_FILE}",
    )
    moving_pose_parser.add_argument(
        "--spd",
        type=float,
        default=MOVING_POSE_SPEED,
        help=f"关节速度，单位度/秒，默认{MOVING_POSE_SPEED:g}",
    )
    moving_pose_parser.add_argument(
        "--acc",
        type=float,
        default=MOVING_POSE_ACCELERATION,
        help=f"关节加速度，单位度/秒²，默认{MOVING_POSE_ACCELERATION:g}",
    )
    moving_pose_parser.add_argument(
        "--execute",
        action="store_true",
        help="实际发送移动姿态命令；不添加时只预览",
    )

    raw_parser = subparsers.add_parser("raw", help="发送原始JSON命令")
    raw_parser.add_argument("json_command")
    raw_parser.add_argument("--read", type=float, default=0.5, help="发送后读取返回的秒数")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == "list":
        list_ports()
        return 0

    if args.command == "status":
        print_json(query_status(args.port, args.baud, args.timeout))
        return 0

    if args.command == "light":
        send_serial_command(args.port, build_light_command(args.led), args.baud, args.timeout)
        return 0

    if args.command == "gripper":
        if args.zero_pose:
            status = {key: 0.0 for key in JOINT_KEYS}
        else:
            status = query_status(args.port, args.baud, args.timeout)
            status = wait_for_motion_idle(
                query_status_fn=lambda: query_status(
                    args.port,
                    args.baud,
                    min(args.timeout, 0.5),
                ),
                initial_status=status,
            )
        if args.angle is None:
            command = build_gripper_command(args.action, status, spd=args.spd, acc=args.acc)
        else:
            command = build_gripper_angle_command(
                args.angle,
                status,
                spd=args.spd,
                acc=args.acc,
            )
        send_serial_command(args.port, command, args.baud, args.timeout)
        return 0

    if args.command == "jog":
        if args.spd <= 0 or args.acc <= 0:
            raise SystemExit("--spd和--acc必须大于0")
        status = query_status(args.port, args.baud, args.timeout)
        if args.execute:
            status = wait_for_motion_idle(
                query_status_fn=lambda: query_status(
                    args.port,
                    args.baud,
                    min(args.timeout, 0.5),
                ),
                initial_status=status,
            )
        current = status_to_command_degrees(status)
        command = build_jog_command(
            args.joint,
            args.delta,
            status,
            spd=args.spd,
            acc=args.acc,
            unsafe=args.unsafe,
        )
        if args.unsafe:
            print("警告：已启用--unsafe，单次微动角度上限已绕过")
        print(f"当前{args.joint}关节角度: {current[args.joint]:.2f}度")
        print(f"目标{args.joint}关节角度: {command[args.joint]:.2f}度")
        print(f"完整目标命令: {compact_json(command)}")
        if not args.execute:
            print("预览模式：未发送运动命令。确认安全后添加 --execute")
            return 0
        send_serial_command(args.port, command, args.baud, args.timeout)
        final_status = wait_for_joint_target(
            query_status_fn=lambda: query_status(
                args.port,
                args.baud,
                min(args.timeout, 0.5),
            ),
            joint=args.joint,
            target_degrees=command[args.joint],
            initial_status=status,
        )
        final_angle = status_to_joint_degrees(final_status)[args.joint]
        print(
            f"运动完成：{args.joint}反馈={final_angle:.2f}度，"
            f"命令坐标={command[args.joint]:.2f}度，"
            f"move={final_status.get('move', '缺失')}"
        )
        return 0

    if args.command == "xyz-jog":
        status = query_status(args.port, args.baud, args.timeout)
        current = status_to_cartesian(status)
        command = build_cartesian_jog_command(
            args.axis,
            args.delta,
            status,
            spd=args.spd,
        )
        print(f"当前{args.axis}坐标: {current[args.axis]:.2f}")
        print(f"目标{args.axis}坐标: {command[args.axis]:.2f}")
        print(f"完整目标命令: {compact_json(command)}")
        if not args.execute:
            print("预览模式：未发送坐标运动命令。确认安全后添加 --execute")
            return 0
        require_cartesian_execution_enabled()
        send_serial_command(args.port, command, args.baud, args.timeout)
        final_status = wait_for_cartesian_target(
            query_status_fn=lambda: query_status(
                args.port,
                args.baud,
                min(args.timeout, 0.5),
            ),
            axis=args.axis,
            target_value=command[args.axis],
        )
        final_value = status_to_cartesian(final_status)[args.axis]
        print(f"坐标运动完成：{args.axis}={final_value:.2f}，move=0")
        return 0

    if args.command == "restore-shoulder":
        status = query_status(args.port, args.baud, args.timeout)
        if args.execute:
            status = wait_for_motion_idle(
                query_status_fn=lambda: query_status(
                    args.port,
                    args.baud,
                    min(args.timeout, 0.5),
                ),
                initial_status=status,
            )
        current = status_to_joint_degrees(status)
        command = build_restore_shoulder_command(status)
        movement = command["s"] - current["s"]
        print(f"当前s关节角度: {current['s']:.2f}度")
        print(f"基准s关节角度: {command['s']:.2f}度")
        print(f"预计运动量: {movement:+.2f}度")
        print(f"完整目标命令: {compact_json(command)}")
        if not args.execute:
            print("预览模式：未发送复位命令。确认路径安全后添加 --execute")
            return 0
        send_serial_command(args.port, command, args.baud, args.timeout)
        final_status = wait_for_joint_target(
            query_status_fn=lambda: query_status(
                args.port,
                args.baud,
                min(args.timeout, 0.5),
            ),
            joint="s",
            target_degrees=command["s"],
            initial_status=status,
        )
        final_angle = status_to_joint_degrees(final_status)["s"]
        print(
            f"复位完成：s={final_angle:.2f}度，"
            f"move={final_status.get('move', '缺失')}"
        )
        return 0

    if args.command == "home":
        if args.spd <= 0 or args.acc <= 0:
            raise SystemExit("--spd和--acc必须大于0")
        status = query_status(args.port, args.baud, args.timeout)
        if args.execute:
            status = wait_for_motion_idle(
                query_status_fn=lambda: query_status(
                    args.port,
                    args.baud,
                    min(args.timeout, 0.5),
                ),
                initial_status=status,
            )
        current = status_to_joint_degrees(status)
        command = build_home_command(status, spd=args.spd, acc=args.acc)
        for joint in JOINT_KEYS:
            target_feedback = HOME_REFERENCE_DEGREES[joint]
            movement = target_feedback - current[joint]
            print(
                f"{joint}关节: 当前={current[joint]:.2f}度 "
                f"目标={target_feedback:.2f}度 运动量={movement:+.2f}度"
            )
        print(f"完整目标命令: {compact_json(command)}")
        if not args.execute:
            print("预览模式：未发送复位命令。确认路径安全后添加 --execute")
            return 0
        send_serial_command(args.port, command, args.baud, args.timeout)
        final_status = wait_for_joint_stability(
            query_status_fn=lambda: query_status(
                args.port,
                args.baud,
                min(args.timeout, 0.5),
            ),
            initial_status=status,
            timeout_seconds=MOTION_WAIT_TIMEOUT_SECONDS,
        )
        final = status_to_joint_degrees(final_status)
        for joint in JOINT_KEYS:
            error = abs(final[joint] - HOME_REFERENCE_DEGREES[joint])
            print(
                f"复位结果 {joint}: 反馈={final[joint]:.2f}度 "
                f"目标={HOME_REFERENCE_DEGREES[joint]:.2f}度 误差={error:.2f}度"
            )
        print(f"move={final_status.get('move', '缺失')}")
        return 0

    if args.command == "raw":
        try:
            command = json.loads(args.json_command)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"原始命令不是合法JSON: {exc}") from exc
        if not isinstance(command, dict):
            raise SystemExit("原始命令必须是JSON对象")
        try:
            raw_command_type = float(command.get("T"))
        except (TypeError, ValueError):
            raw_command_type = math.nan
        if math.isfinite(raw_command_type) and raw_command_type == 104.0:
            require_cartesian_execution_enabled()
        data = send_serial_command(args.port, command, args.baud, args.timeout, read_seconds=args.read)
        if data:
            print("原始返回:", data.decode("utf-8", errors="replace").strip())
        return 0

    raise SystemExit(f"未知命令: {args.command}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
