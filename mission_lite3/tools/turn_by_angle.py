from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

from ..config_loader import ConfigError, load_config
from ..lite3_motion import Lite3MotionController
from ..state_reader import StateReader


def _angle_delta(end: float, start: float) -> float:
    return (float(end) - float(start) + math.pi) % (2.0 * math.pi) - math.pi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Perform one small feedback-verified Lite3 in-place turn"
    )
    parser.add_argument("--degrees", type=float, required=True)
    parser.add_argument("--speed-rad-s", type=float, default=0.08)
    parser.add_argument("--tolerance-deg", type=float, default=0.10)
    parser.add_argument("--final-tolerance-deg", type=float, default=0.20)
    parser.add_argument("--max-corrections", type=int, default=2)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--robot", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.degrees) or not 0.0 < abs(args.degrees) <= 5.0:
        raise SystemExit("--degrees must be finite and in [-5, 5], excluding zero")
    if not math.isfinite(args.speed_rad_s) or not 0.02 <= args.speed_rad_s <= 0.15:
        raise SystemExit("--speed-rad-s must be in [0.02, 0.15]")
    if not math.isfinite(args.tolerance_deg) or not 0.02 <= args.tolerance_deg <= 0.5:
        raise SystemExit("--tolerance-deg must be in [0.02, 0.5]")
    if (
        not math.isfinite(args.final_tolerance_deg)
        or not 0.05 <= args.final_tolerance_deg <= 0.5
    ):
        raise SystemExit("--final-tolerance-deg must be in [0.05, 0.5]")
    if not 0 <= args.max_corrections <= 3:
        raise SystemExit("--max-corrections must be in [0, 3]")

    try:
        config = load_config(args.config_dir)
    except ConfigError as exc:
        print(json.dumps({"ok": False, "reason": f"config: {exc}"}))
        return 2

    dry_run = not args.robot
    reader = StateReader(config, dry_run=dry_run)
    motion = Lite3MotionController(config, dry_run=dry_run)

    def guard(vx: float, vy: float, wz: float) -> None:
        del vx, vy, wz
        error = reader.safety_error(require_ultrasound=False, require_fresh=True)
        if error:
            raise RuntimeError(f"motion guard rejected command: {error}")

    motion.configure_safety(guard, reader.pose, feedback_required=not dry_run)
    motion.yaw_tolerance_rad = math.radians(args.tolerance_deg)
    start_pose = None
    end_pose = None
    headings_before = None
    headings_after = None
    front_before = None
    front_after = None
    autonomous = False
    cleanup_errors: list[str] = []
    corrections: list[dict[str, float]] = []
    reason = "turn did not start"
    ok = False
    try:
        motion.start()
        reader.start()
        reader.wait_until_ready(
            float(config["navigation"].get("startup_sensor_timeout_s", 3.0)),
            require_ultrasound=True,
        )
        start_pose = reader.pose()
        headings_before = reader.headings()
        front_before = reader.state.front_ultrasound_m
        motion.set_autonomous()
        autonomous = True
        motion.turn_by(math.radians(args.degrees), wz=args.speed_rad_s)
        for correction_index in range(args.max_corrections + 1):
            deadline = time.monotonic() + max(0.0, args.settle_seconds)
            while time.monotonic() < deadline:
                reader.poll()
                time.sleep(0.03)
            end_pose = reader.pose()
            headings_after = reader.headings()
            measured_deg = math.degrees(
                _angle_delta(end_pose[2], start_pose[2])
            )
            error_deg = float(args.degrees) - measured_deg
            if abs(error_deg) <= args.final_tolerance_deg:
                break
            if correction_index >= args.max_corrections:
                raise RuntimeError(
                    "final yaw error exceeds tolerance: "
                    f"requested={args.degrees:.3f}deg "
                    f"measured={measured_deg:.3f}deg "
                    f"error={error_deg:.3f}deg"
                )
            corrections.append(
                {
                    "measured_deg": measured_deg,
                    "command_deg": error_deg,
                }
            )
            motion.turn_by(math.radians(error_deg), wz=args.speed_rad_s)
        front_after = reader.state.front_ultrasound_m
        ok = True
        reason = "completed"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        if autonomous:
            try:
                motion.stop()
                motion.set_manual()
            except Exception as exc:
                cleanup_errors.append(f"restore manual mode: {exc}")
        try:
            reader.close()
        except Exception as exc:
            cleanup_errors.append(f"state reader: {exc}")
        try:
            motion.close()
        except Exception as exc:
            cleanup_errors.append(f"motion: {exc}")

    delta_deg = None
    if start_pose is not None and end_pose is not None:
        delta_deg = math.degrees(_angle_delta(end_pose[2], start_pose[2]))
    if cleanup_errors:
        ok = False
        reason = f"{reason}; cleanup failed: {'; '.join(cleanup_errors)}"
    print(
        json.dumps(
            {
                "ok": ok,
                "reason": reason,
                "robot_output": bool(args.robot),
                "requested_deg": args.degrees,
                "measured_delta_deg": delta_deg,
                "start_pose": start_pose,
                "end_pose": end_pose,
                "headings_before_rad": headings_before,
                "headings_after_rad": headings_after,
                "odom_delta_deg": (
                    math.degrees(
                        _angle_delta(headings_after[0], headings_before[0])
                    )
                    if headings_before is not None and headings_after is not None
                    else None
                ),
                "imu_delta_deg": (
                    math.degrees(
                        _angle_delta(headings_after[1], headings_before[1])
                    )
                    if headings_before is not None and headings_after is not None
                    else None
                ),
                "front_ultrasound_before_m": front_before,
                "front_ultrasound_after_m": front_after,
                "corrections": corrections,
                "cleanup_errors": cleanup_errors,
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
