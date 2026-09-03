from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, Mapping, Optional

from mission_lite3.arm.runtime import test as arm_test
from mission_lite3.arm.runtime.arm_task import ArmTestSerialMotion


DEFAULT_PORT = "/dev/ttyUSB0"
DIRECTION_DELTAS = {"left": 10.0, "right": -10.0}
DIRECTION_NAMES = {"left": "向左", "right": "向右"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="b关节向左或向右转动10度，并记录实测误差")
    parser.add_argument("direction", choices=tuple(DIRECTION_DELTAS), help="left向左，right向右")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--spd", type=float, default=arm_test.SAFE_JOG_SPEED)
    parser.add_argument("--acc", type=float, default=arm_test.SAFE_JOG_ACCELERATION)
    parser.add_argument("--serial-timeout", type=float, default=arm_test.DEFAULT_TIMEOUT)
    parser.add_argument("--motion-timeout", type=float, default=arm_test.MOTION_WAIT_TIMEOUT_SECONDS)
    parser.add_argument("--execute", action="store_true", help="实际执行；不添加时只预览")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("spd", "acc", "serial_timeout", "motion_timeout"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}必须是大于0的有效数字")


def build_result(
    *,
    direction: str,
    initial_status: Mapping[str, Any],
    command: Mapping[str, Any],
    executed: bool,
    final_status: Optional[Mapping[str, Any]] = None,
    error: Optional[BaseException] = None,
) -> Dict[str, Any]:
    requested_delta = DIRECTION_DELTAS[direction]
    initial_angle = arm_test.status_to_joint_degrees(dict(initial_status))["b"]
    target_angle = float(command["b"])
    result: Dict[str, Any] = {
        "成功": error is None,
        "已执行": executed,
        "关节": "b",
        "方向": DIRECTION_NAMES[direction],
        "要求转动角度_度": 10.0,
        "命令增量_度": requested_delta,
        "初始角度_度": initial_angle,
        "目标角度_度": target_angle,
        "完整命令": dict(command),
    }
    if final_status is not None:
        final_angle = arm_test.status_to_joint_degrees(dict(final_status))["b"]
        actual_delta = final_angle - initial_angle
        signed_error = actual_delta - requested_delta
        target_error = final_angle - target_angle
        result.update(
            {
                "最终角度_度": final_angle,
                "实际转动角度_度": abs(actual_delta),
                "实际带方向增量_度": actual_delta,
                "相对10度误差_度": signed_error if direction == "left" else -signed_error,
                "目标角度误差_度": target_error,
                "绝对误差_度": abs(target_error),
            }
        )
    if error is not None:
        result["错误"] = f"{type(error).__name__}: {error}"
    return result


def print_result(result: Mapping[str, Any]) -> None:
    print(json.dumps(dict(result), ensure_ascii=False, indent=2))
    if result.get("已执行") and "实际转动角度_度" in result:
        print(
            "[b关节] "
            f"方向={result['方向']}，"
            f"要求角度=10.000度，"
            f"实际角度={float(result['实际转动角度_度']):.3f}度，"
            f"误差={float(result['相对10度误差_度']):+.3f}度，"
            f"绝对误差={float(result['绝对误差_度']):.3f}度"
        )


def run(args: argparse.Namespace) -> int:
    try:
        validate_args(args)
        motion = ArmTestSerialMotion(port=args.port, arm_module=arm_test, timeout=args.serial_timeout)
        initial_status = motion._wait_ready(motion._query_status())
        command = arm_test.build_jog_command(
            "b",
            DIRECTION_DELTAS[args.direction],
            initial_status,
            spd=args.spd,
            acc=args.acc,
        )
    except (KeyboardInterrupt, Exception) as exc:
        print(json.dumps({"成功": False, "已执行": False, "错误": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2))
        return 130 if isinstance(exc, KeyboardInterrupt) else 1

    if not args.execute:
        print_result(build_result(direction=args.direction, initial_status=initial_status, command=command, executed=False))
        return 0

    final_status: Optional[Dict[str, Any]] = None
    motion_error: Optional[BaseException] = None
    try:
        motion._send(command)
        final_status = arm_test.wait_for_joint_target(
            query_status_fn=motion._query_fast_status,
            joint="b",
            target_degrees=float(command["b"]),
            initial_status=initial_status,
            timeout_seconds=args.motion_timeout,
        )
    except (KeyboardInterrupt, Exception) as exc:
        motion_error = exc
        try:
            final_status = motion._query_status()
        except Exception:
            pass

    print_result(build_result(direction=args.direction, initial_status=initial_status, command=command, executed=True, final_status=final_status, error=motion_error))
    if isinstance(motion_error, KeyboardInterrupt):
        return 130
    return 0 if motion_error is None else 1


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
