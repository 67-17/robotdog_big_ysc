from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from ..config_loader import load_config
from ..mission import LargeQuadrupedMission, MissionAbort, MissionState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one pickup from the pickup-entry position, stop after the object "
            "is held, and skip retreat, transfer, placement, and a second pickup"
        ),
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--output-dir", default="single_grasp_runs")
    parser.add_argument("--robot", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--udp-fallback", action="store_true")
    parser.add_argument("--axis-fallback", action="store_true")
    return parser


def _close_single_grasp_mission(
    mission: LargeQuadrupedMission,
    *,
    autonomous_enabled: bool,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    manual_mode_restored = False
    for attempt in range(3):
        try:
            mission.motion.stop()
        except Exception as exc:
            errors.append(f"motion stop {attempt + 1}: {exc}")
        if not mission.context.dry_run:
            time.sleep(0.03)
    if autonomous_enabled:
        try:
            mission.motion.set_manual()
            manual_mode_restored = True
        except Exception as exc:
            errors.append(f"restore manual mode: {exc}")
    else:
        manual_mode_restored = mission.context.dry_run

    for name, close in (
        ("front camera", mission.front_camera.release),
        ("wide camera", mission.wide_camera.release),
        ("arm camera", mission.arm_camera.release),
        ("arm", mission.arm.close),
        ("state reader", mission.state_reader.close),
        ("motion", mission.motion.close),
    ):
        try:
            close()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return manual_mode_restored, errors


def run_single_grasp(
    args: argparse.Namespace,
    *,
    mission_factory: Callable[..., LargeQuadrupedMission] = LargeQuadrupedMission,
) -> dict[str, Any]:
    dry_run = not bool(args.robot)
    config = load_config(args.config_dir)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "ok": False,
        "dry_run": dry_run,
        "object_held": False,
        "run_dir": str(run_dir),
        "steps": [],
        "manual_mode_restored": False,
        "cleanup_errors": [],
    }
    mission = mission_factory(
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
        mission._require_arm_result(mission.arm.start(), "single grasp preflight")
        if not dry_run:
            mission.state_reader.wait_until_ready(
                float(config["navigation"].get("startup_sensor_timeout_s", 3.0)),
                require_ultrasound=True,
            )
        mission._check_safety()
        mission.motion.prepare_walk()
        if not dry_run:
            time.sleep(1.0)
        mission.motion.set_autonomous()
        autonomous_enabled = True
        result["steps"].append("walk_mode_prepared")
        mission._require_arm_result(mission.arm.stow(), "single grasp moving pose")

        mission.state = MissionState.PICK_RED_BAR
        mission.context.pickup_stage = "pregrasp"
        mission.context.pickup_pregrasp_substage = "idle"
        mission.context.carried_bar = False
        result["steps"].append("pickup_entry_ready")

        mission.motion.stop()
        try:
            aligned = mission._run_pregrasp_base_sequence()
        finally:
            mission.motion.stop()
        if not aligned:
            raise MissionAbort("pickup-entry base adjustment or red-target alignment failed")
        result["steps"].append("base_adjustment_completed")

        mission._require_arm_result(mission.arm.camera_pose(), "single grasp ready pose")
        mission._settle_after_pregrasp_stop()
        result["steps"].append("grasp_ready")

        distance_mm = (
            260.0
            if getattr(mission.arm, "backend", "runtime") == "runtime"
            else mission._estimate_red_bar_distance_mm()
        )
        mission.context.carried_bar = mission._retry_grasp(distance_mm)
        result["object_held"] = bool(mission.context.carried_bar)
        if not mission.context.carried_bar:
            raise MissionAbort("single grasp ended without confirmed object_held")

        mission.motion.stop()
        result["steps"].append("one_object_held")
        result["reason"] = "one object held; retreat, transfer, placement, and second pickup skipped"
        result["ok"] = True
    except KeyboardInterrupt:
        result["reason"] = "operator interrupted single grasp test"
    except Exception as exc:
        result["object_held"] = bool(mission.context.carried_bar)
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        manual_mode_restored, cleanup_errors = _close_single_grasp_mission(
            mission,
            autonomous_enabled=autonomous_enabled,
        )
        result["manual_mode_restored"] = manual_mode_restored
        result["cleanup_errors"].extend(cleanup_errors)
        if cleanup_errors or not manual_mode_restored:
            result["ok"] = False
        result["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.robot and not args.yes:
        print(
            json.dumps(
                {
                    "ok": False,
                    "reason": "real single grasp requires --robot --yes",
                    "start_position": "pickup entry",
                },
                ensure_ascii=False,
            )
        )
        return 2
    result = run_single_grasp(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
