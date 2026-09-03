from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config_loader import load_config
from ..mission import LargeQuadrupedMission, MissionAbort, MissionState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test only the post-pickup base route to a placement letter; "
            "the arm is never started or commanded"
        ),
    )
    parser.add_argument("target", nargs="?", choices=tuple("ABCD"), default="D")
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--output-dir", default="placement_navigation_test_runs")
    parser.add_argument("--robot", action="store_true")
    parser.add_argument(
        "--confirm-safe-start",
        action="store_true",
        help=(
            "Confirm the base is at the just-grasped pickup pose facing the pickup "
            "box and the disabled arm is physically secured with safe clearance"
        ),
    )
    parser.add_argument("--udp-fallback", action="store_true")
    parser.add_argument("--axis-fallback", action="store_true")
    return parser


def _front_state(mission: LargeQuadrupedMission) -> dict[str, float]:
    state = mission.state_reader.state
    values = {
        "front_ultrasound_m": float(state.front_ultrasound_m),
        "x": float(state.x),
        "y": float(state.y),
        "yaw": float(state.yaw),
        "roll_deg": float(state.roll_deg),
        "pitch_deg": float(state.pitch_deg),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise MissionAbort(f"state contains a non-finite value: {values}")
    return values


def _startup_base_only(mission: LargeQuadrupedMission, dry_run: bool) -> None:
    mission.motion.start()
    mission.state_reader.start()
    if dry_run:
        mission._check_safety()
        return
    timeout_s = float(
        mission.config["navigation"].get("startup_sensor_timeout_s", 3.0)
    )
    mission.state_reader.wait_until_ready(timeout_s, require_ultrasound=True)
    if not mission.front_camera.open():
        raise MissionAbort(
            f"front camera failed to open: {mission.front_camera.source}"
        )
    first_frame_timeout_s = float(
        mission.config["camera"].get("startup_first_frame_timeout_s", 6.0)
    )
    if mission.front_camera.read(timeout_s=first_frame_timeout_s) is None:
        raise MissionAbort("front camera did not deliver a startup frame")
    mission._check_safety()


def _run_navigation(
    mission: LargeQuadrupedMission,
    target: str,
    result: dict[str, Any],
) -> None:
    mission.state = MissionState.PLACE_TO_LETTER_BOX
    mission.context.target_letter = target
    mission.context.placement_target_letter = target
    mission.context.placement_stage = "route"

    result["steps"].append("retreat_from_pickup_box_started")
    mission._retreat_from_pickup_box()
    result["steps"].append("retreat_from_pickup_box_completed")

    result["steps"].append("pickup_departure_yaw_alignment_started")
    yaw_aligned = mission._align_pickup_departure_yaw()
    result["pickup_departure_yaw_aligned"] = bool(yaw_aligned)
    result["steps"].append("pickup_departure_yaw_alignment_completed")

    mission._placement_route_active = True
    try:
        result["steps"].append(f"placement_navigation_{target}_started")
        route_completed = mission._run_resumable_scripted_route(
            "place_from_pickup",
            progress_attr="placement_route_action_index",
        )
    finally:
        mission._placement_route_active = False
    if not route_completed:
        raise MissionAbort("place_from_pickup route is unavailable")
    if not mission.context.placement_visual_approach_complete:
        raise MissionAbort("placement visual navigation did not complete")
    if not mission.context.dry_run:
        if not mission.context.placement_letter_centered_complete:
            raise MissionAbort(f"target {target} was not centered")
        if not mission.context.placement_ultrasound_approach_complete:
            raise MissionAbort("final 0.28m ultrasound approach did not complete")
    mission.motion.stop()
    result["steps"].append(f"placement_navigation_{target}_completed")


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    target = str(args.target).upper()
    dry_run = not bool(args.robot)
    if not dry_run and not args.confirm_safe_start:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": (
                        "real navigation test requires --robot --confirm-safe-start; "
                        "the base must be at the just-grasped pickup pose and the arm "
                        "must be physically secured"
                    ),
                    "target": target,
                    "arm_commands_sent": 0,
                },
                ensure_ascii=False,
            )
        )
        return 2

    config = load_config(args.config_dir)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "ok": False,
        "reason": "navigation test did not start",
        "dry_run": dry_run,
        "target": target,
        "run_dir": str(run_dir),
        "steps": [],
        "arm_backend_disabled": True,
        "arm_commands_sent": 0,
        "manual_mode_restored": False,
        "cleanup_errors": [],
    }
    mission = LargeQuadrupedMission(
        config,
        dry_run=dry_run,
        udp_fallback=args.udp_fallback,
        axis_fallback=args.axis_fallback,
        skip_arm=True,
    )
    autonomous_enabled = False
    started_at = time.monotonic()
    try:
        _startup_base_only(mission, dry_run)
        result["start_state"] = (
            {
                "front_ultrasound_m": 0.28,
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "roll_deg": 0.0,
                "pitch_deg": 0.0,
            }
            if dry_run
            else _front_state(mission)
        )
        mission.motion.set_autonomous()
        autonomous_enabled = True
        _run_navigation(mission, target, result)
        result["end_state"] = None if dry_run else _front_state(mission)
        result["target_centered"] = bool(
            mission.context.placement_letter_centered_complete
        )
        result["ultrasound_approach_complete"] = bool(
            mission.context.placement_ultrasound_approach_complete
        )
        result["visual_approach_complete"] = bool(
            mission.context.placement_visual_approach_complete
        )
        result["ok"] = True
        result["reason"] = "placement navigation test completed; release skipped"
    except KeyboardInterrupt:
        result["reason"] = "operator interrupted placement navigation test"
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if autonomous_enabled:
            try:
                mission.motion.stop()
            except Exception as exc:
                result["cleanup_errors"].append(f"motion stop: {exc}")
            try:
                mission.motion.set_manual()
                result["manual_mode_restored"] = True
            except Exception as exc:
                result["cleanup_errors"].append(f"restore manual mode: {exc}")
        result["cleanup_errors"].extend(mission._cleanup())
        if result["cleanup_errors"]:
            result["ok"] = False
        result["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
