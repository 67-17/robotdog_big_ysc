from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ..config_loader import load_config
from ..lite3_motion import Lite3MotionController
from ..state_reader import StateReader


MAX_DISTANCE_M = 0.10
MAX_SPEED_MPS = 0.06
MAX_VY_CORRECTION_MPS = 0.04
MAX_WZ_CORRECTION_RAD_S = 0.12


def normalize_angle_rad(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def body_delta(
    reference_pose: tuple[float, float, float],
    current_pose: tuple[float, float, float],
) -> tuple[float, float, float]:
    x0, y0, yaw0 = (float(value) for value in reference_pose)
    x, y, yaw = (float(value) for value in current_pose)
    dx = x - x0
    dy = y - y0
    forward = math.cos(yaw0) * dx + math.sin(yaw0) * dy
    lateral = -math.sin(yaw0) * dx + math.cos(yaw0) * dy
    return forward, lateral, normalize_angle_rad(yaw - yaw0)


def correction_velocity(
    reference_pose: tuple[float, float, float],
    current_pose: tuple[float, float, float],
    *,
    base_vx: float,
) -> tuple[float, float, float]:
    _forward, lateral, yaw_error = body_delta(reference_pose, current_pose)
    vy = max(
        -MAX_VY_CORRECTION_MPS,
        min(MAX_VY_CORRECTION_MPS, -1.0 * lateral),
    )
    wz = max(
        -MAX_WZ_CORRECTION_RAD_S,
        min(MAX_WZ_CORRECTION_RAD_S, -1.2 * yaw_error),
    )
    if abs(lateral) <= 0.003:
        vy = 0.0
    if abs(math.degrees(yaw_error)) <= 0.30:
        wz = 0.0
    return float(base_vx), vy, wz


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Guarded short straight probe with odometry/IMU pose hold"
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--direction", choices=["forward", "backward"], default="backward")
    parser.add_argument("--distance-m", type=float, default=0.05)
    parser.add_argument("--speed-mps", type=float, default=0.05)
    parser.add_argument(
        "--front-stop-cm",
        type=float,
        default=None,
        help="for forward motion, stop output when front ultrasound reaches this distance",
    )
    parser.add_argument("--robot", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    distance = abs(float(args.distance_m))
    speed = abs(float(args.speed_mps))
    if not 0.01 <= distance <= MAX_DISTANCE_M:
        print(json.dumps({"ok": False, "reason": "distance must be within 0.01 to 0.10m"}))
        return 2
    if not 0.02 <= speed <= MAX_SPEED_MPS:
        print(json.dumps({"ok": False, "reason": "speed must be within 0.02 to 0.06m/s"}))
        return 2
    base_vx = speed if args.direction == "forward" else -speed
    duration = distance / speed
    front_stop_m = (
        None if args.front_stop_cm is None else float(args.front_stop_cm) / 100.0
    )
    if front_stop_m is not None and not 0.28 <= front_stop_m <= 4.50:
        print(json.dumps({"ok": False, "reason": "front-stop-cm must be within 28 to 450cm"}))
        return 2
    if not args.robot:
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "direction": args.direction,
                    "base_vx_mps": base_vx,
                    "duration_s": duration,
                    "nominal_distance_m": distance,
                    "motion_command_count": 0,
                    "front_stop_m": front_stop_m,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if not args.yes:
        print(json.dumps({"ok": False, "reason": "real probe requires --robot --yes"}))
        return 2

    config = load_config(args.config_dir)
    motion = Lite3MotionController(config, dry_run=False)
    reader = StateReader(config, dry_run=False)
    result = {
        "ok": False,
        "dry_run": False,
        "direction": args.direction,
        "base_vx_mps": base_vx,
        "duration_s": duration,
        "nominal_distance_m": distance,
        "motion_command_count": 0,
        "front_stop_m": front_stop_m,
        "manual_mode_restored": False,
        "cleanup_errors": [],
    }
    motion_started = False
    reader_started = False
    autonomous_may_be_enabled = False
    try:
        motion.start()
        motion_started = True
        reader.start()
        reader_started = True
        reader.wait_until_ready(
            float(config["navigation"].get("startup_sensor_timeout_s", 3.0)),
            require_ultrasound=True,
        )

        def guard(_vx: float, _vy: float, _wz: float) -> None:
            error = reader.safety_error(require_ultrasound=True, require_fresh=True)
            if error:
                raise RuntimeError(f"motion guard rejected straight probe: {error}")

        motion.configure_safety(guard, reader.pose, feedback_required=True)
        reference_pose = reader.pose()
        result["start_pose"] = reference_pose
        result["front_ultrasound_m"] = reader.state.front_ultrasound_m

        if (
            base_vx > 0.0
            and front_stop_m is not None
            and float(reader.state.front_ultrasound_m) <= front_stop_m
        ):
            result["reason"] = "front_stop_already_reached"
            result["end_front_ultrasound_m"] = reader.state.front_ultrasound_m
            result["ok"] = True
            print(json.dumps(result, ensure_ascii=False))
            return 0

        autonomous_may_be_enabled = True
        motion.set_autonomous()
        def velocity_provider() -> tuple[float, float, float]:
            pose = reader.pose()
            if (
                base_vx > 0.0
                and front_stop_m is not None
                and float(reader.state.front_ultrasound_m) <= front_stop_m
            ):
                return 0.0, 0.0, 0.0
            return correction_velocity(
                reference_pose,
                pose,
                base_vx=base_vx,
            )

        motion.hold_velocity_feedback(velocity_provider, duration)
        result["motion_command_count"] = 1
        motion.stop()
        motion.set_manual()
        autonomous_may_be_enabled = False
        result["manual_mode_restored"] = True
        end_pose = reader.pose()
        result["end_pose"] = end_pose
        forward, lateral, yaw_delta = body_delta(reference_pose, end_pose)
        result["measured_forward_m"] = forward
        result["measured_lateral_m"] = lateral
        result["measured_yaw_delta_deg"] = math.degrees(yaw_delta)
        reader.poll()
        result["end_front_ultrasound_m"] = reader.state.front_ultrasound_m
        result["ok"] = True
    except KeyboardInterrupt:
        result["reason"] = "operator interrupted straight probe"
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if autonomous_may_be_enabled:
            try:
                motion.stop()
            except Exception as exc:
                result["cleanup_errors"].append(f"stop: {exc}")
            try:
                motion.set_manual()
                result["manual_mode_restored"] = True
            except Exception as exc:
                result["cleanup_errors"].append(f"restore manual: {exc}")
        if reader_started:
            try:
                reader.close()
            except Exception as exc:
                result["cleanup_errors"].append(f"reader close: {exc}")
        if motion_started:
            try:
                motion.close()
            except Exception as exc:
                result["cleanup_errors"].append(f"motion close: {exc}")
        if result["cleanup_errors"]:
            result["ok"] = False

    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
