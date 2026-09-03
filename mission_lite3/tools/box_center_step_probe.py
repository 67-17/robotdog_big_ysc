from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2

from ..box_center_alignment import (
    BoxCenterAligner,
    BoxCenterAlignmentResult,
    BoxCenterMeasurement,
    annotate_box_centers,
    strafe_distance_for_box_center,
)
from ..camera import CameraSource
from ..config_loader import ConfigError, PROJECT_ROOT, load_config
from ..lite3_motion import Lite3MotionController
from ..state_reader import StateReader
from ..wide_camera import WideCameraUndistorter


STEP_LIMIT_M = 0.05
STEP_CORRECTION_LIMIT = 1
FULL_SINGLE_LIMIT_M = 0.25
FULL_TOTAL_LIMIT_M = 0.75
FULL_CORRECTION_LIMIT = 3
STEP_SETTLE_SECONDS = 0.8
MIN_DIRECTION_IMPROVEMENT_PX = 5.0
MAX_VISUAL_ROLLBACK_ERROR_FRACTION = 0.02
MAX_ODOM_RETURN_ERROR_M = 0.04
MAX_ODOM_RETURN_YAW_DEG = 2.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Guarded one-step placement-box center probe. Real output requires "
            "--robot --yes and is capped at one 0.05m correction."
        )
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--target-letter", choices=tuple("ABCD"), default="C")
    parser.add_argument("--profile", choices=("step", "full"), default="step")
    parser.add_argument("--robot", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--udp-fallback", action="store_true")
    parser.add_argument("--axis-fallback", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser


def evaluate_probe_result(
    alignment: BoxCenterAlignmentResult,
    *,
    rollback_measurement: Optional[BoxCenterMeasurement] = None,
    odom_return_error_m: Optional[float] = None,
    odom_return_yaw_deg: Optional[float] = None,
    profile: str = "step",
    external_return_attempted: bool = False,
    external_return_ok: bool = False,
    minimum_improvement_px: float = MIN_DIRECTION_IMPROVEMENT_PX,
) -> dict[str, Any]:
    initial = alignment.initial_error_px
    final = alignment.final_error_px
    improvement = (
        None
        if initial is None or final is None
        else abs(float(initial)) - abs(float(final))
    )
    if profile == "full":
        correction_executed = (
            1 <= alignment.correction_count <= FULL_CORRECTION_LIMIT
            and abs(float(alignment.visual_strafe_m)) <= FULL_TOTAL_LIMIT_M + 1e-6
            and abs(float(alignment.visual_strafe_m)) >= 0.01
        )
        alignment_verified = alignment.ok and alignment.reason == "aligned"
    else:
        correction_executed = (
            alignment.correction_count == 1
            and abs(float(alignment.visual_strafe_m)) <= STEP_LIMIT_M + 1e-6
            and abs(float(alignment.visual_strafe_m)) >= 0.01
        )
        alignment_verified = True
    direction_verified = (
        improvement is not None
        and improvement >= float(minimum_improvement_px)
    )
    internal_rollback_verified = (
        alignment.rollback_attempted
        and alignment.rollback_ok
        and abs(float(alignment.net_strafe_m)) <= 1e-6
    )
    external_return_verified = external_return_attempted and external_return_ok
    return_command_verified = internal_rollback_verified or external_return_verified
    rollback_visual_delta_px = (
        None
        if initial is None
        or rollback_measurement is None
        or rollback_measurement.target_error_px is None
        else abs(float(rollback_measurement.target_error_px) - float(initial))
    )
    visual_rollback_verified = (
        rollback_measurement is not None
        and rollback_measurement.ok
        and rollback_visual_delta_px is not None
        and rollback_visual_delta_px
        <= rollback_measurement.frame_width * MAX_VISUAL_ROLLBACK_ERROR_FRACTION
    )
    odom_return_verified = (
        odom_return_error_m is not None
        and odom_return_yaw_deg is not None
        and float(odom_return_error_m) <= MAX_ODOM_RETURN_ERROR_M
        and abs(float(odom_return_yaw_deg)) <= MAX_ODOM_RETURN_YAW_DEG
    )
    return {
        "ok": bool(
            correction_executed
            and direction_verified
            and alignment_verified
            and return_command_verified
            and visual_rollback_verified
            and odom_return_verified
        ),
        "correction_executed": correction_executed,
        "direction_verified": direction_verified,
        "alignment_verified": alignment_verified,
        "return_command_verified": return_command_verified,
        "internal_rollback_verified": internal_rollback_verified,
        "external_return_verified": external_return_verified,
        "visual_rollback_verified": visual_rollback_verified,
        "odom_return_verified": odom_return_verified,
        "error_improvement_px": improvement,
        "rollback_visual_delta_px": rollback_visual_delta_px,
        "odom_return_error_m": odom_return_error_m,
        "odom_return_yaw_deg": odom_return_yaw_deg,
    }


def _normalize_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _pose_return_error(
    start_pose: tuple[float, float, float],
    end_pose: tuple[float, float, float],
) -> tuple[float, float]:
    translation = math.hypot(
        float(end_pose[0]) - float(start_pose[0]),
        float(end_pose[1]) - float(start_pose[1]),
    )
    yaw_deg = math.degrees(_normalize_angle(float(end_pose[2]) - float(start_pose[2])))
    return translation, yaw_deg


def _resolve_project_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _profile_limits(profile: str) -> tuple[int, float, float]:
    if profile == "full":
        return FULL_CORRECTION_LIMIT, FULL_SINGLE_LIMIT_M, FULL_TOTAL_LIMIT_M
    return STEP_CORRECTION_LIMIT, STEP_LIMIT_M, STEP_LIMIT_M


def _probe_config(
    config: dict,
    output_dir: object,
    *,
    profile: str,
) -> dict[str, object]:
    corrections, single_limit, total_limit = _profile_limits(profile)
    center_config = dict(config.get("box_center_alignment", {}))
    center_config.update(
        {
            "enabled": True,
            "frames_per_measurement": 7,
            "min_valid_frames": 4,
            "max_center_range_fraction": 0.03,
            "tolerance_fraction": 0.05,
            "max_corrections": corrections,
            "max_single_strafe_m": single_limit,
            "max_total_strafe_m": total_limit,
            "settle_seconds": STEP_SETTLE_SECONDS,
            "alignment_run_log_dir": str(_resolve_project_path(output_dir)),
        }
    )
    speed = abs(float(center_config.get("strafe_speed_mps", 0.08)))
    if not 0.02 <= speed <= 0.08:
        raise ValueError("probe strafe speed must be within 0.02 to 0.08m/s")
    center_config["strafe_speed_mps"] = speed
    return center_config


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.robot and not args.yes:
        print(
            json.dumps(
                {"ok": False, "reason": "real probe requires --robot --yes"},
                ensure_ascii=False,
            )
        )
        return 2
    try:
        config = load_config(args.config_dir)
        output_dir = args.output_dir or (
            "box_center_full_runs" if args.profile == "full" else "box_center_step_runs"
        )
        center_config = _probe_config(
            config,
            output_dir,
            profile=args.profile,
        )
    except (ConfigError, ValueError) as exc:
        print(json.dumps({"ok": False, "reason": str(exc)}, ensure_ascii=False))
        return 2

    if not args.robot:
        corrections, single_limit, total_limit = _profile_limits(args.profile)
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "profile": args.profile,
                    "target_letter": args.target_letter,
                    "motion_command_count": 0,
                    "max_corrections": corrections,
                    "max_single_strafe_m": single_limit,
                    "max_total_strafe_m": total_limit,
                    "planned_sequence": [
                        "fresh odometry, IMU, and ultrasound",
                        "seven-frame placement measurement",
                        (
                            "up to three corrections capped at 0.25m each"
                            if args.profile == "full"
                            else "one correction capped at 0.05m"
                        ),
                        "seven-frame remeasurement",
                        "automatic rollback when still outside tolerance",
                        "stop and restore manual mode",
                    ],
                },
                ensure_ascii=False,
            )
        )
        return 0

    camera_config = config["camera"]
    camera = CameraSource(
        camera_config["front"],
        dry_run=False,
        flush_grab_frames=int(camera_config.get("flush_grab_frames", 4)),
        stale_frame_reconnect_count=int(camera_config.get("stale_frame_reconnect_count", 15)),
        digital_zoom=1.0,
        open_timeout_ms=int(camera_config.get("open_timeout_ms", 3000)),
        read_timeout_ms=int(camera_config.get("read_timeout_ms", 2000)),
        reconnect_backoff_s=float(camera_config.get("reconnect_backoff_s", 0.25)),
    )
    undistorter = WideCameraUndistorter.from_file(
        _resolve_project_path(camera_config["wide_calibration"])
    )
    motion = Lite3MotionController(
        config,
        dry_run=False,
        udp_fallback=args.udp_fallback,
        axis_fallback=args.axis_fallback,
    )
    state_reader = StateReader(config, dry_run=False)
    aligner = BoxCenterAligner(
        camera=camera,
        undistorter=undistorter,
        motion=motion,
        config=center_config,
    )
    result: dict[str, Any] = {
        "ok": False,
        "dry_run": False,
        "profile": args.profile,
        "target_letter": args.target_letter,
        "motion_command_count": 0,
        "max_single_strafe_m": float(center_config["max_single_strafe_m"]),
        "manual_mode_restored": False,
        "cleanup_errors": [],
    }
    motion_started = False
    state_started = False
    autonomous_may_be_enabled = False
    run_dir: Optional[Path] = None
    try:
        motion.start()
        motion_started = True
        state_reader.start()
        state_started = True
        state_reader.wait_until_ready(
            float(config.get("navigation", {}).get("startup_sensor_timeout_s", 3.0)),
            require_ultrasound=True,
        )
        state = state_reader.poll()
        ultrasound = state.front_ultrasound_m
        min_valid = float(config["safety"].get("front_ultrasound_min_valid_m", 0.03))
        if ultrasound is None or not math.isfinite(float(ultrasound)) or float(ultrasound) < min_valid:
            raise RuntimeError(f"front ultrasound is invalid: {ultrasound!r}")
        result["front_ultrasound_m"] = float(ultrasound)

        def guard(_vx: float, _vy: float, _wz: float) -> None:
            error = state_reader.safety_error(
                require_ultrasound=True,
                require_fresh=True,
            )
            if error:
                raise RuntimeError(f"motion guard rejected box-center probe: {error}")
            current = state_reader.state.front_ultrasound_m
            if current is None or not math.isfinite(float(current)) or float(current) < min_valid:
                raise RuntimeError(f"motion guard received invalid ultrasound: {current!r}")

        motion.configure_safety(guard, state_reader.pose, feedback_required=True)
        start_pose = state_reader.pose()
        result["start_pose"] = start_pose
        autonomous_may_be_enabled = True
        motion.set_autonomous()
        time.sleep(0.20)
        alignment = aligner.run("placement", args.target_letter)
        run_dir = Path(alignment.run_dir) if alignment.run_dir else None
        result["alignment"] = asdict(alignment)
        result["motion_command_count"] = alignment.motion_command_count
        external_return_attempted = False
        external_return_ok = False
        if alignment.ok and abs(float(alignment.net_strafe_m)) > 1e-6:
            external_return_attempted = True
            result["external_return_strafe_m"] = -float(alignment.net_strafe_m)
            strafe_distance_for_box_center(
                motion,
                -float(alignment.net_strafe_m),
                center_config,
            )
            result["motion_command_count"] += 1
            external_return_ok = True
        result["external_return_attempted"] = external_return_attempted
        result["external_return_ok"] = external_return_ok
        motion.stop()
        motion.set_manual()
        autonomous_may_be_enabled = False
        result["manual_mode_restored"] = True
        camera.release()
        time.sleep(STEP_SETTLE_SECONDS)

        def save_rollback_frame(index, raw, undistorted, frame_result) -> None:
            if run_dir is None:
                return
            if raw is not None:
                cv2.imwrite(str(run_dir / f"rollback_{index:03d}_raw.jpg"), raw)
            if undistorted is not None:
                cv2.imwrite(
                    str(run_dir / f"rollback_{index:03d}_annotated.jpg"),
                    annotate_box_centers(undistorted, frame_result),
                )

        rollback_measurement = aligner.measurer.measure(
            "placement",
            args.target_letter,
            frame_callback=save_rollback_frame,
        )
        result["rollback_measurement"] = asdict(rollback_measurement)
        end_pose = state_reader.pose()
        result["end_pose"] = end_pose
        odom_error_m, odom_yaw_deg = _pose_return_error(start_pose, end_pose)
        evaluation = evaluate_probe_result(
            alignment,
            rollback_measurement=rollback_measurement,
            odom_return_error_m=odom_error_m,
            odom_return_yaw_deg=odom_yaw_deg,
            profile=args.profile,
            external_return_attempted=external_return_attempted,
            external_return_ok=external_return_ok,
        )
        result["evaluation"] = evaluation
        result["ok"] = bool(evaluation["ok"])
        if not result["ok"]:
            result["reason"] = (
                "direction or rollback was not verified: "
                f"alignment_reason={alignment.reason} evaluation={evaluation}"
            )
    except KeyboardInterrupt:
        result["reason"] = "operator interrupted box-center step probe"
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        for _attempt in range(3):
            try:
                motion.stop()
            except Exception as exc:
                result["cleanup_errors"].append(f"stop: {exc}")
            time.sleep(0.03)
        if autonomous_may_be_enabled:
            try:
                motion.set_manual()
                result["manual_mode_restored"] = True
            except Exception as exc:
                result["cleanup_errors"].append(f"restore manual mode: {exc}")
        camera.release()
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
        if result["cleanup_errors"]:
            result["ok"] = False
        if run_dir is not None:
            try:
                (run_dir / "probe_result.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
            except Exception as exc:
                result["cleanup_errors"].append(f"write probe result: {exc}")
                result["ok"] = False

    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
