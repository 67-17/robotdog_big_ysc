#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/common.sh
source "$(dirname "$0")/common.sh"

ARM_STARTUP_POSE_FILE="${ARM_STARTUP_POSE_FILE:-moving_pose.json}"
ARM_STARTUP_SPD="${ARM_STARTUP_SPD:-15}"
ARM_STARTUP_ACC="${ARM_STARTUP_ACC:-15}"
export ARM_STARTUP_POSE_FILE ARM_STARTUP_SPD ARM_STARTUP_ACC

python3 - <<'PY'
import importlib.util
import json
import os
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("arm_test_runtime", "test.py")
arm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arm)

pose_path = Path(os.environ["ARM_STARTUP_POSE_FILE"])
record = arm.read_pose_record(pose_path)
command = arm.build_pose_command(
    record,
    spd=float(os.environ["ARM_STARTUP_SPD"]),
    acc=float(os.environ["ARM_STARTUP_ACC"]),
)

port = os.environ["ARM_PORT"]
baud = int(os.environ["ARM_BAUD"])
timeout = float(os.environ["ARM_TIMEOUT"])

print("移动姿态命令:", json.dumps(command, ensure_ascii=False, separators=(",", ":")))
arm.send_serial_command(port, command, baud, timeout, read_seconds=0.5)
time.sleep(1.0)

try:
    status = arm.query_status(port, baud, max(timeout, 5.0))
except Exception as exc:
    print(f"移动姿态命令已发送，但状态读取失败: {exc}")
else:
    print("当前关节反馈:", json.dumps(arm.status_to_joint_degrees(status), ensure_ascii=False))
    print(f"当前夹爪反馈: {arm.status_to_gripper_degrees(status):.2f}度")
    print(f"move={status.get('move')}")
PY
