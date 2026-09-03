from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

from mission_lite3.config_loader import load_config

from .lite_arm import LiteArmController
from .runtime import arm_task


ARM_COMMANDS = (
    "grasp",
    "grasp-ready",
    "moving-pose",
    "transport",
    "close",
    "hold",
    "place",
    "home",
    "status",
    "abort",
    "preflight",
    "diagnose-run",
    "validate-run",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the packaged Lite3 arm runtime with mission device paths",
    )
    parser.add_argument("command", choices=ARM_COMMANDS)
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--port", help="override arm serial device")
    parser.add_argument("--camera", help="override arm camera device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--slot", default="A")
    parser.add_argument("--object-held", action="store_true")
    parser.add_argument("--show-vision", action="store_true")
    parser.add_argument("--hold-vision", action="store_true")
    parser.add_argument("--single-step", action="store_true")
    parser.add_argument("--skip-grasp-ready", action="store_true")
    parser.add_argument("--stop-after-final-pose", action="store_true")
    parser.add_argument("--require-preflight", action="store_true")
    parser.add_argument("--angle", type=float)
    parser.add_argument("--max-align-steps", type=int)
    parser.add_argument("--max-jog-deg", type=float)
    parser.add_argument("--spd", type=float)
    parser.add_argument("--acc", type=float)
    parser.add_argument("--final-spd", type=float)
    parser.add_argument("--final-acc", type=float)
    return parser


def _set_runtime_option(argv: list[str], option: str, value: object) -> None:
    try:
        index = argv.index(option)
    except ValueError:
        argv.extend([option, str(value)])
    else:
        argv[index + 1] = str(value)


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config_dir)
    arm_cfg = dict(config.get("arm", {}))
    if args.port:
        arm_cfg["port"] = args.port
    if args.camera:
        arm_cfg["camera_device"] = args.camera
    config["arm"] = arm_cfg

    controller = LiteArmController(config, dry_run=args.dry_run, skip_arm=False)
    if args.command == "moving-pose":
        started_at = time.monotonic()
        result = controller.moving_pose()
        print(
            f"[arm-cli] command={args.command} "
            f"elapsed={time.monotonic() - started_at:.2f}s "
            f"ok={result.ok} stage={result.stage} reason={result.reason}"
        )
        return 0 if result.ok else 1

    result_file = controller._result_file(  # noqa: SLF001
        args.command.replace("-", "_")
    )
    include_camera = args.command == "preflight" or (
        args.command == "grasp" and not args.dry_run
    )
    runtime_argv = controller._runtime_base_argv(  # noqa: SLF001
        result_file,
        include_camera=include_camera,
    )

    for enabled, option in (
        (args.dry_run, "--dry-run"),
        (args.show_vision, "--show-vision"),
        (args.hold_vision, "--hold-vision"),
        (args.single_step, "--single-step"),
        (args.skip_grasp_ready, "--skip-grasp-ready"),
        (args.stop_after_final_pose, "--stop-after-final-pose"),
        (args.require_preflight, "--require-preflight"),
    ):
        if enabled:
            runtime_argv.append(option)

    for option, value in (
        ("--max-align-steps", args.max_align_steps),
        ("--max-jog-deg", args.max_jog_deg),
        ("--spd", args.spd),
        ("--acc", args.acc),
        ("--final-spd", args.final_spd),
        ("--final-acc", args.final_acc),
    ):
        if value is not None:
            _set_runtime_option(runtime_argv, option, value)

    if args.command == "grasp":
        runtime_argv.extend(
            ["--run-log-dir", str(controller._new_run_log_dir())]  # noqa: SLF001
        )

    runtime_argv.append(args.command)
    if args.command == "close" and args.angle is not None:
        runtime_argv.extend(["--angle", str(args.angle)])
    if args.command == "place":
        runtime_argv.extend(["--slot", args.slot])
        if args.object_held:
            runtime_argv.append("--object-held")
    if args.command in {"diagnose-run", "validate-run"} and args.directory:
        runtime_argv.append(str(args.directory))

    started_at = time.monotonic()
    exit_code = arm_task.main(runtime_argv)
    print(
        f"[arm-cli] command={args.command} "
        f"elapsed={time.monotonic() - started_at:.2f}s result={result_file}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
