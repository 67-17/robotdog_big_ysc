from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .config_loader import ConfigError, load_config
from .mission import LargeQuadrupedMission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the camera-guided quadruped strafe used before grasping. "
            "Hardware motion requires --robot."
        )
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument(
        "--robot",
        action="store_true",
        help="enable real robot output; without this flag the command is a dry run",
    )
    parser.add_argument(
        "--max-pulses",
        type=int,
        default=None,
        help="override the configured pulse limit; use 1 for step-by-step testing",
    )
    parser.add_argument(
        "--udp-fallback",
        action="store_true",
        help="use direct UDP motion output instead of ROS2 cmd_vel",
    )
    parser.add_argument(
        "--axis-fallback",
        action="store_true",
        help="with --udp-fallback, send the official UDP axis command",
    )
    parser.add_argument(
        "--skip-wide-parallel",
        action="store_true",
        help=(
            "skip wide-camera yaw alignment and run only the arm-camera "
            "quadruped strafe loop"
        ),
    )
    parser.add_argument(
        "--wide-only",
        action="store_true",
        help="run only wide-camera yaw alignment and skip arm-camera strafe",
    )
    parser.add_argument(
        "--resume-lateral",
        action="store_true",
        help=(
            "resume an interrupted arm-camera strafe using the configured "
            "locked-target size threshold"
        ),
    )
    return parser


def _enable_lateral_resume(config: dict) -> None:
    align_config = dict(config.get("pregrasp_red_align", {}))
    tracking_ratio = float(
        align_config["strict_tracking_min_linear_size_ratio"]
    )
    align_config["strict_motion_min_linear_size_ratio"] = tracking_ratio
    config["pregrasp_red_align"] = align_config


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.wide_only and args.skip_wide_parallel:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "--wide-only and --skip-wide-parallel are mutually exclusive",
                },
                ensure_ascii=False,
            )
        )
        return 2
    if args.resume_lateral and not args.skip_wide_parallel:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "--resume-lateral requires --skip-wide-parallel",
                },
                ensure_ascii=False,
            )
        )
        return 2
    try:
        config = load_config(args.config_dir)
    except ConfigError as exc:
        print(json.dumps({"ok": False, "reason": f"config: {exc}"}, ensure_ascii=False))
        return 2

    if args.max_pulses is not None:
        if args.max_pulses < 0:
            print(json.dumps({"ok": False, "reason": "max-pulses must be >= 0"}, ensure_ascii=False))
            return 2
        align_config = dict(config.get("pregrasp_red_align", {}))
        align_config["max_pulses"] = int(args.max_pulses)
        config["pregrasp_red_align"] = align_config

    if args.skip_wide_parallel:
        align_config = dict(config.get("pregrasp_red_align", {}))
        wide_config = dict(align_config.get("wide_parallel", {}))
        wide_config["enabled"] = False
        align_config["wide_parallel"] = wide_config
        config["pregrasp_red_align"] = align_config

    if args.resume_lateral:
        _enable_lateral_resume(config)

    dry_run = not args.robot
    mission = LargeQuadrupedMission(
        config,
        dry_run=dry_run,
        udp_fallback=args.udp_fallback,
        axis_fallback=args.axis_fallback,
        skip_arm=True,
    )
    ok = False
    reason = "alignment did not start"
    cleanup_errors: list[str] = []
    autonomous_enabled = False
    start_pose: tuple[float, float, float] | None = None
    end_pose: tuple[float, float, float] | None = None
    try:
        mission.motion.start()
        mission.state_reader.start()
        startup_timeout = float(
            config.get("navigation", {}).get("startup_sensor_timeout_s", 3.0)
        )
        mission.state_reader.wait_until_ready(
            startup_timeout,
            require_ultrasound=True,
        )
        if not mission._pregrasp_ultrasound_ready():  # noqa: SLF001
            reason = "pregrasp ultrasound safety gate rejected the current sample"
        else:
            mission.motion.set_autonomous()
            autonomous_enabled = True
            start_pose = mission.state_reader.pose()
            ok = bool(  # noqa: SLF001
                mission._run_pregrasp_red_alignment(wide_only=args.wide_only)
            )
            end_pose = mission.state_reader.pose()
            reason = "aligned" if ok else "alignment stopped before success"
    except KeyboardInterrupt:
        reason = "operator interrupted alignment"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        if autonomous_enabled:
            try:
                mission.motion.stop()
                mission.motion.set_manual()
            except Exception as exc:
                cleanup_errors.append(f"restore manual mode: {exc}")
        cleanup_errors.extend(mission._cleanup())  # noqa: SLF001
        if cleanup_errors:
            ok = False
            reason = f"{reason}; cleanup failed: {'; '.join(cleanup_errors)}"

    odom_delta = None
    if start_pose is not None and end_pose is not None:
        odom_delta = {
            "x_m": end_pose[0] - start_pose[0],
            "y_m": end_pose[1] - start_pose[1],
            "yaw_rad": end_pose[2] - start_pose[2],
        }

    print(
        json.dumps(
            {
                "ok": ok,
                "reason": reason,
                "robot_output": bool(args.robot),
                "max_pulses": config.get("pregrasp_red_align", {}).get("max_pulses"),
                "wide_parallel_enabled": config.get("pregrasp_red_align", {})
                .get("wide_parallel", {})
                .get("enabled", True),
                "wide_only": bool(args.wide_only),
                "resume_lateral": bool(args.resume_lateral),
                "start_pose": start_pose,
                "end_pose": end_pose,
                "odom_delta": odom_delta,
                "cleanup_errors": cleanup_errors,
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
