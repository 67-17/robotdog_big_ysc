#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mission_lite3.arm import run_arm_task  # noqa: E402


REAL_MOTION_COMMANDS = {
    "abort",
    "grasp",
    "grasp-ready",
    "home",
    "place",
    "transport",
}


@dataclass(frozen=True)
class Step:
    command: str
    object_held: bool = False


SCENARIOS: dict[str, tuple[Step, ...]] = {
    "dry-run": (
        Step("status"),
        Step("transport"),
        Step("grasp"),
        Step("place", object_held=True),
    ),
    "preflight": (Step("preflight"),),
    "status": (Step("status"),),
    "transport": (Step("transport"),),
    "grasp-ready": (Step("grasp-ready"),),
    "grasp": (Step("grasp"),),
    "place": (Step("place", object_held=True),),
    "full": (
        Step("preflight"),
        Step("transport"),
        Step("grasp"),
        Step("place", object_held=True),
        Step("transport"),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Lite3 arm test entry. Defaults to dry-run and never starts "
            "the quadruped mission state machine."
        )
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="dry-run",
        help="test sequence to run; default dry-run",
    )
    parser.add_argument("--real", action="store_true", help="run against real arm hardware")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="required with --real for commands that can move the arm",
    )
    parser.add_argument("--port", help="override arm serial device, for example /dev/ttyUSB1")
    parser.add_argument("--camera", help="override arm camera device, for example /dev/video2")
    parser.add_argument("--slot", default="A", choices=("A", "B", "C", "D"))
    parser.add_argument("--result-dir", type=Path, help="write last_*_result.json here")
    parser.add_argument("--show-vision", action="store_true", help="show grasp/preflight vision window")
    parser.add_argument("--single-step", action="store_true", help="limit grasp visual alignment to one step")
    parser.add_argument(
        "--skip-grasp-ready",
        action="store_true",
        help="skip grasp-ready setup for grasp when the arm is already prepared",
    )
    parser.add_argument(
        "--stop-after-final-pose",
        action="store_true",
        help="stop grasp after final pose, before final recheck and gripper close",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="run remaining scenario steps after a failure",
    )
    return parser


def _step_argv(
    args: argparse.Namespace,
    step: Step,
    *,
    object_held: bool | None = None,
) -> list[str]:
    argv: list[str] = []
    if not args.real:
        argv.append("--dry-run")
    if args.port:
        argv.extend(["--port", args.port])
    if args.camera:
        argv.extend(["--camera", args.camera])
    if args.show_vision:
        argv.append("--show-vision")
    if args.single_step:
        argv.append("--single-step")
    if args.skip_grasp_ready:
        argv.append("--skip-grasp-ready")
    if args.stop_after_final_pose:
        argv.append("--stop-after-final-pose")

    argv.append(step.command)
    if step.command == "place":
        argv.extend(["--slot", args.slot])
        if step.object_held if object_held is None else object_held:
            argv.append("--object-held")
    return argv


def _last_result_path(result_dir: Path, command: str) -> Path:
    return result_dir / f"last_{command.replace('-', '_')}_result.json"


def _read_last_result(result_dir: Path, command: str) -> dict:
    path = _last_result_path(result_dir, command)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _requires_confirmation(args: argparse.Namespace, steps: Iterable[Step]) -> bool:
    if not args.real or args.yes:
        return False
    return any(step.command in REAL_MOTION_COMMANDS for step in steps)


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    steps = SCENARIOS[args.scenario]
    if _requires_confirmation(args, steps):
        print(
            "[standalone-arm] refusing real motion without --yes; "
            "rerun with --real --yes after the workspace is clear",
            file=sys.stderr,
        )
        return 2

    previous_result_dir = os.environ.get("ARM_RESULT_DIR")
    result_dir_changed = args.result_dir is not None
    if result_dir_changed:
        args.result_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ARM_RESULT_DIR"] = str(args.result_dir)

    try:
        print(
            "[standalone-arm] "
            f"scenario={args.scenario} mode={'real' if args.real else 'dry-run'} "
            f"steps={','.join(step.command for step in steps)}"
        )
        worst_exit_code = 0
        result_dir = args.result_dir or Path(
            os.environ.get("ARM_RESULT_DIR", PROJECT_ROOT / "logs")
        )
        chained_object_held = False
        chained_scenario = len(steps) > 1
        for index, step in enumerate(steps, start=1):
            if step.command == "place" and chained_scenario and not chained_object_held:
                print(
                    "[standalone-arm] skipping place: previous grasp did not "
                    "report verified object_held=true"
                )
                continue
            declared_object_held = (
                chained_object_held
                if step.command == "place" and chained_scenario
                else None
            )
            step_argv = _step_argv(
                args,
                step,
                object_held=declared_object_held,
            )
            started_at = time.monotonic()
            print(f"[standalone-arm] step {index}/{len(steps)} argv={step_argv}")
            exit_code = run_arm_task.main(step_argv)
            elapsed = time.monotonic() - started_at
            print(
                f"[standalone-arm] step {index}/{len(steps)} "
                f"command={step.command} exit={exit_code} elapsed={elapsed:.2f}s"
            )
            worst_exit_code = max(worst_exit_code, int(exit_code))
            payload = _read_last_result(result_dir, step.command)
            if step.command == "grasp":
                chained_object_held = bool(
                    payload.get("ok") and payload.get("object_held")
                )
            elif (
                step.command == "place"
                and payload.get("ok")
                and not payload.get("object_held", True)
            ):
                chained_object_held = False
            if exit_code != 0 and not args.continue_on_error:
                break
        return worst_exit_code
    finally:
        if result_dir_changed:
            if previous_result_dir is None:
                os.environ.pop("ARM_RESULT_DIR", None)
            else:
                os.environ["ARM_RESULT_DIR"] = previous_result_dir


if __name__ == "__main__":
    raise SystemExit(main())
