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
        description="Run two pickup-transfer-placement cycles from the pickup area",
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument(
        "--targets",
        nargs=2,
        choices=tuple("ABCD"),
        default=("A", "D"),
        metavar=("FIRST", "SECOND"),
    )
    parser.add_argument("--output-dir", default="pickup_transfer_runs")
    parser.add_argument("--robot", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--udp-fallback", action="store_true")
    parser.add_argument("--axis-fallback", action="store_true")
    return parser


def _front_state(mission: LargeQuadrupedMission) -> dict[str, float]:
    state = mission.state_reader.poll()
    values = {
        "front_ultrasound_m": float(state.front_ultrasound_m),
        "x": float(state.x),
        "y": float(state.y),
        "yaw": float(state.yaw),
        "roll_deg": float(state.roll_deg),
        "pitch_deg": float(state.pitch_deg),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise MissionAbort(f"startup state contains a non-finite value: {values}")
    return values


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    targets = [str(letter).upper() for letter in args.targets]
    dry_run = not bool(args.robot)
    if not dry_run and not args.yes:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "real transfer mission requires --robot --yes",
                    "targets": targets,
                },
                ensure_ascii=False,
            )
        )
        return 2

    config = load_config(args.config_dir)
    config["scripted_route"] = dict(config["scripted_route"])
    # The operator has positioned the robot at the pickup-area 80 cm line.
    config["scripted_route"]["pickup_from_upper_inspection"] = []
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "ok": False,
        "dry_run": dry_run,
        "targets": targets,
        "run_dir": str(run_dir),
        "steps": [],
        "manual_mode_restored": False,
        "cleanup_errors": [],
    }
    mission = LargeQuadrupedMission(
        config,
        dry_run=dry_run,
        udp_fallback=args.udp_fallback,
        axis_fallback=args.axis_fallback,
        skip_arm=False,
    )
    autonomous_enabled = False
    started_at = time.monotonic()
    try:
        mission.motion.start()
        mission.state_reader.start()
        mission._require_arm_result(mission.arm.start(), "transfer preflight")
        if not dry_run:
            mission.state_reader.wait_until_ready(
                float(config["navigation"].get("startup_sensor_timeout_s", 3.0)),
                require_ultrasound=True,
            )
            if not mission.front_camera.open() or mission.front_camera.read() is None:
                raise MissionAbort("front camera did not deliver a startup frame")
        mission._check_safety()
        result["start_state"] = (
            {
                "front_ultrasound_m": 0.80,
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "roll_deg": 0.0,
                "pitch_deg": 0.0,
            }
            if dry_run
            else _front_state(mission)
        )
        autonomous_enabled = True
        mission.motion.set_autonomous()
        mission._require_arm_result(mission.arm.stow(), "initial moving pose")

        for cycle, target in enumerate(targets, start=1):
            mission.state = (
                MissionState.PICK_RED_BAR
                if cycle == 1
                else MissionState.SECOND_PICK_PLACE
            )
            result["steps"].append(f"cycle_{cycle}_pickup_{target}_started")
            if not mission._pick_target(target):
                raise MissionAbort(f"cycle {cycle} pickup failed for {target}")
            result["steps"].append(f"cycle_{cycle}_pickup_{target}_completed")

            mission.state = (
                MissionState.PLACE_TO_LETTER_BOX
                if cycle == 1
                else MissionState.SECOND_PICK_PLACE
            )
            if not mission._place_carried_bar():
                raise MissionAbort(f"cycle {cycle} placement failed for {target}")
            result["steps"].append(f"cycle_{cycle}_placement_{target}_completed")
            result[f"cycle_{cycle}_end_state"] = (
                None if dry_run else _front_state(mission)
            )

        mission.state = MissionState.FINISH_OR_SAFE_STOP
        mission._state_finish_or_safe_stop()
        result["first_outbound_lane_strafe_m"] = (
            mission.context.first_outbound_lane_strafe_m
        )
        result["placed_letters"] = list(mission.context.placed_letters)
        result["ok"] = mission.context.placed_letters == targets
        result["reason"] = "completed" if result["ok"] else "placement record mismatch"
    except KeyboardInterrupt:
        result["reason"] = "operator interrupted transfer mission"
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
        cleanup_errors = mission._cleanup()
        result["cleanup_errors"].extend(cleanup_errors)
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
