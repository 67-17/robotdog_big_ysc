from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, Mapping, Optional

from mission_lite3.arm.runtime import test as arm_test
from mission_lite3.arm.runtime.arm_task import ArmTestSerialMotion


DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_DELTA_DEGREES = -10.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read the current e joint, optionally lift it upward by 10 degrees "
            "(this arm uses e -10 degrees for upward motion), and report error."
        )
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--delta-deg", type=float, default=DEFAULT_DELTA_DEGREES)
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
        help="Actually send the motion command. Without this flag, only preview it.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    values = {
        "delta-deg": args.delta_deg,
        "spd": args.spd,
        "acc": args.acc,
        "serial-timeout": args.serial_timeout,
        "motion-timeout": args.motion_timeout,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    max_delta = float(arm_test.JOINT_MAX_JOG_DEGREES["e"])
    if args.delta_deg == 0 or abs(args.delta_deg) > max_delta:
        raise ValueError(f"delta-deg must be non-zero and within +/-{max_delta:g} degrees")
    if args.spd <= 0 or args.acc <= 0:
        raise ValueError("spd and acc must be positive")
    if args.serial_timeout <= 0 or args.motion_timeout <= 0:
        raise ValueError("timeouts must be positive")


def build_result(
    *,
    initial_status: Mapping[str, Any],
    command: Mapping[str, Any],
    requested_delta_deg: float,
    executed: bool,
    final_status: Optional[Mapping[str, Any]] = None,
    error: Optional[BaseException] = None,
) -> Dict[str, Any]:
    initial_feedback = arm_test.status_to_joint_degrees(dict(initial_status))["e"]
    initial_command = arm_test.status_to_command_degrees(dict(initial_status))["e"]
    target_command = float(command["e"])
    target_feedback = arm_test.command_to_feedback_degrees("e", target_command)
    result: Dict[str, Any] = {
        "ok": error is None,
        "executed": executed,
        "joint": "e",
        "requested_delta_deg": float(requested_delta_deg),
        "initial_feedback_deg": initial_feedback,
        "initial_command_deg": initial_command,
        "target_command_deg": target_command,
        "target_feedback_deg": target_feedback,
        "command": dict(command),
    }
    if final_status is not None:
        final_feedback = arm_test.status_to_joint_degrees(dict(final_status))["e"]
        actual_delta = final_feedback - initial_feedback
        target_error = final_feedback - target_feedback
        result.update(
            {
                "final_feedback_deg": final_feedback,
                "actual_delta_deg": actual_delta,
                "delta_error_deg": actual_delta - float(requested_delta_deg),
                "target_error_deg": target_error,
                "absolute_target_error_deg": abs(target_error),
            }
        )
    if error is not None:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def run(args: argparse.Namespace) -> int:
    try:
        validate_args(args)
        motion = ArmTestSerialMotion(
            port=args.port,
            arm_module=arm_test,
            timeout=args.serial_timeout,
        )
        initial_status = motion._wait_ready(motion._query_status())
        command = arm_test.build_jog_command(
            "e",
            args.delta_deg,
            initial_status,
            spd=args.spd,
            acc=args.acc,
        )
    except (KeyboardInterrupt, Exception) as exc:
        print(json.dumps({"ok": False, "executed": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2))
        return 130 if isinstance(exc, KeyboardInterrupt) else 1

    if not args.execute:
        print(
            json.dumps(
                build_result(
                    initial_status=initial_status,
                    command=command,
                    requested_delta_deg=args.delta_deg,
                    executed=False,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    final_status: Optional[Dict[str, Any]] = None
    motion_error: Optional[BaseException] = None
    try:
        motion._send(command)
        final_status = arm_test.wait_for_joint_target(
            query_status_fn=motion._query_fast_status,
            joint="e",
            target_degrees=float(command["e"]),
            initial_status=initial_status,
            timeout_seconds=args.motion_timeout,
        )
    except (KeyboardInterrupt, Exception) as exc:
        motion_error = exc
        try:
            final_status = motion._query_status()
        except Exception:
            pass

    print(
        json.dumps(
            build_result(
                initial_status=initial_status,
                command=command,
                requested_delta_deg=args.delta_deg,
                executed=True,
                final_status=final_status,
                error=motion_error,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    if isinstance(motion_error, KeyboardInterrupt):
        return 130
    return 0 if motion_error is None else 1


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
