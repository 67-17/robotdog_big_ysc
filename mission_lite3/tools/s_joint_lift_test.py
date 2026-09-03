from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, Mapping, Optional

from mission_lite3.arm.runtime import test as arm_test
from mission_lite3.arm.runtime.arm_task import ArmTestSerialMotion


DEFAULT_PORT = "/dev/ttyUSB0"
DIRECTION_DELTAS = {"up": 10.0, "down": -10.0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move the s joint up or down by 10 degrees and report the measured error."
    )
    parser.add_argument("direction", choices=tuple(DIRECTION_DELTAS))
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--spd", type=float, default=arm_test.SAFE_JOG_SPEED)
    parser.add_argument("--acc", type=float, default=arm_test.SAFE_JOG_ACCELERATION)
    parser.add_argument("--serial-timeout", type=float, default=arm_test.DEFAULT_TIMEOUT)
    parser.add_argument(
        "--motion-timeout",
        type=float,
        default=arm_test.MOTION_WAIT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send the command. Without this flag, only preview it.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("spd", "acc", "serial_timeout", "motion_timeout"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be a positive finite number")


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
    initial_feedback = arm_test.status_to_joint_degrees(dict(initial_status))["s"]
    initial_command = arm_test.status_to_command_degrees(dict(initial_status))["s"]
    target_command = float(command["s"])
    target_feedback = arm_test.command_to_feedback_degrees("s", target_command)
    result: Dict[str, Any] = {
        "ok": error is None,
        "executed": executed,
        "joint": "s",
        "direction": direction,
        "requested_motion_deg": 10.0,
        "requested_command_delta_deg": requested_delta,
        "initial_command_deg": initial_command,
        "initial_feedback_deg": initial_feedback,
        "target_command_deg": target_command,
        "target_feedback_deg": target_feedback,
        "command": dict(command),
    }
    if final_status is not None:
        final_feedback = arm_test.status_to_joint_degrees(dict(final_status))["s"]
        final_command = arm_test.status_to_command_degrees(dict(final_status))["s"]
        actual_command_delta = final_command - initial_command
        target_error = final_command - target_command
        result.update(
            {
                "final_command_deg": final_command,
                "final_feedback_deg": final_feedback,
                "actual_command_delta_deg": actual_command_delta,
                "raw_feedback_delta_deg": final_feedback - initial_feedback,
                "delta_error_deg": actual_command_delta - requested_delta,
                "target_error_deg": target_error,
                "absolute_target_error_deg": abs(target_error),
            }
        )
    if error is not None:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def print_result(result: Mapping[str, Any]) -> None:
    print(json.dumps(dict(result), ensure_ascii=False, indent=2))
    if result.get("executed") and "actual_command_delta_deg" in result:
        print(
            "[s-joint] "
            f"direction={result['direction']} "
            f"requested=10.000deg "
            f"actual={abs(float(result['actual_command_delta_deg'])):.3f}deg "
            f"signed_error={float(result['delta_error_deg']):+.3f}deg "
            f"absolute_error={float(result['absolute_target_error_deg']):.3f}deg"
        )


def run(args: argparse.Namespace) -> int:
    try:
        validate_args(args)
        delta = DIRECTION_DELTAS[args.direction]
        motion = ArmTestSerialMotion(
            port=args.port,
            arm_module=arm_test,
            timeout=args.serial_timeout,
        )
        initial_status = motion._wait_ready(motion._query_status())
        command = arm_test.build_jog_command(
            "s",
            delta,
            initial_status,
            spd=args.spd,
            acc=args.acc,
        )
    except (KeyboardInterrupt, Exception) as exc:
        print(json.dumps({"ok": False, "executed": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2))
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
            joint="s",
            target_degrees=float(command["s"]),
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
