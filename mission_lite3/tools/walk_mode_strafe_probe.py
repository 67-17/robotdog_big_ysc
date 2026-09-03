from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

from ..config_loader import load_config
from ..lite3_motion import Lite3MotionController
from ..state_reader import StateReader


MAX_PROBE_DISTANCE_M = 0.05
PROBE_SPEED_MPS = 0.08
RETURN_TOLERANCE_M = 0.02


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare walk mode, strafe at most 5cm, and return to origin",
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--robot", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser


def _body_delta(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> tuple[float, float, float]:
    delta_x = float(end[0]) - float(start[0])
    delta_y = float(end[1]) - float(start[1])
    yaw = float(start[2])
    forward = math.cos(yaw) * delta_x + math.sin(yaw) * delta_y
    lateral = -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y
    yaw_delta = (float(end[2]) - yaw + math.pi) % (2.0 * math.pi) - math.pi
    return forward, lateral, yaw_delta


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.robot:
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "planned_distance_m": MAX_PROBE_DISTANCE_M,
                    "planned_sequence": [
                        "prepare walk mode",
                        "closed-loop left strafe",
                        "closed-loop return to origin",
                        "restore manual mode",
                    ],
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
    state_reader = StateReader(config, dry_run=False)
    result = {
        "ok": False,
        "dry_run": False,
        "planned_distance_m": MAX_PROBE_DISTANCE_M,
        "manual_mode_restored": False,
        "cleanup_errors": [],
    }
    motion_started = False
    state_started = False
    autonomous_enabled = False
    start_pose: Optional[tuple[float, float, float]] = None
    try:
        motion.start()
        motion_started = True
        state_reader.start()
        state_started = True
        state_reader.wait_until_ready(
            float(config["navigation"].get("startup_sensor_timeout_s", 3.0)),
            require_ultrasound=True,
        )

        def guard(_vx: float, _vy: float, _wz: float) -> None:
            error = state_reader.safety_error(
                require_ultrasound=True,
                require_fresh=True,
            )
            if error:
                raise RuntimeError(f"probe safety rejected state: {error}")

        motion.configure_safety(guard, state_reader.pose, feedback_required=True)
        start_pose = state_reader.pose()
        result["start_pose"] = start_pose
        motion.prepare_walk()
        time.sleep(1.0)
        motion.set_autonomous()
        autonomous_enabled = True
        time.sleep(0.2)

        motion.strafe_distance_pose_hold(
            MAX_PROBE_DISTANCE_M,
            speed_mps=PROBE_SPEED_MPS,
            completion_tolerance_m=0.005,
        )
        reached_pose = state_reader.pose()
        forward_m, lateral_m, yaw_rad = _body_delta(start_pose, reached_pose)
        result["reached_pose"] = reached_pose
        result["outbound_delta"] = {
            "forward_m": forward_m,
            "lateral_m": lateral_m,
            "yaw_deg": math.degrees(yaw_rad),
        }
        if not 0.035 <= lateral_m <= 0.065:
            raise RuntimeError(f"outbound lateral distance rejected: {lateral_m:.3f}m")

        motion.strafe_distance_pose_hold(
            -lateral_m,
            speed_mps=PROBE_SPEED_MPS,
            completion_tolerance_m=0.005,
        )
        final_pose = state_reader.pose()
        final_forward_m, final_lateral_m, final_yaw_rad = _body_delta(
            start_pose,
            final_pose,
        )
        result["final_pose"] = final_pose
        result["return_delta"] = {
            "forward_m": final_forward_m,
            "lateral_m": final_lateral_m,
            "yaw_deg": math.degrees(final_yaw_rad),
        }
        result["ok"] = (
            math.hypot(final_forward_m, final_lateral_m) <= RETURN_TOLERANCE_M
            and abs(math.degrees(final_yaw_rad)) <= 2.0
        )
        result["reason"] = (
            "walk-mode strafe and return verified"
            if result["ok"]
            else "return-to-origin tolerance failed"
        )
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        if start_pose is not None and state_started:
            try:
                current_pose = state_reader.pose()
                result["failure_pose"] = current_pose
                _, lateral_m, _ = _body_delta(start_pose, current_pose)
                if abs(lateral_m) >= 0.01:
                    motion.strafe_distance_pose_hold(
                        -lateral_m,
                        speed_mps=PROBE_SPEED_MPS,
                        completion_tolerance_m=0.005,
                    )
                    result["failure_return_attempted"] = True
            except Exception as return_exc:
                result["cleanup_errors"].append(f"failure return: {return_exc}")
    finally:
        if motion_started:
            for attempt in range(3):
                try:
                    motion.stop()
                except Exception as exc:
                    result["cleanup_errors"].append(
                        f"motion stop {attempt + 1}: {exc}"
                    )
                time.sleep(0.03)
        if autonomous_enabled:
            try:
                motion.set_manual()
                result["manual_mode_restored"] = True
            except Exception as exc:
                result["cleanup_errors"].append(f"restore manual mode: {exc}")
        if state_started:
            try:
                state_reader.close()
            except Exception as exc:
                result["cleanup_errors"].append(f"state reader close: {exc}")
        if motion_started:
            try:
                motion.close()
            except Exception as exc:
                result["cleanup_errors"].append(f"motion close: {exc}")
        if result["cleanup_errors"] or not result["manual_mode_restored"]:
            result["ok"] = False

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
