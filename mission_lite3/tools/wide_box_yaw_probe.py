from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..camera import CameraSource
from ..config_loader import load_config
from ..lite3_motion import Lite3MotionController
from ..state_reader import StateReader
from ..wide_camera import (
    WideCameraUndistorter,
    annotate_box_parallel,
    detect_box_parallel,
)


MAX_PROBE_SPEED_RAD_S = 0.12
MAX_PROBE_DURATION_S = 0.35
MIN_PROBE_SPEED_RAD_S = 0.03
MIN_PROBE_DURATION_S = 0.10


def validate_probe_parameters(speed_rad_s: float, duration_s: float) -> None:
    speed = abs(float(speed_rad_s))
    duration = float(duration_s)
    if not MIN_PROBE_SPEED_RAD_S <= speed <= MAX_PROBE_SPEED_RAD_S:
        raise ValueError(
            f"absolute speed must be within {MIN_PROBE_SPEED_RAD_S:.2f} to "
            f"{MAX_PROBE_SPEED_RAD_S:.2f} rad/s"
        )
    if not MIN_PROBE_DURATION_S <= duration <= MAX_PROBE_DURATION_S:
        raise ValueError(
            f"duration must be within {MIN_PROBE_DURATION_S:.2f} to "
            f"{MAX_PROBE_DURATION_S:.2f} seconds"
        )


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [
        float(sample["parallel_error_deg"])
        for sample in samples
        if sample.get("ok") and sample.get("parallel_error_deg") is not None
    ]
    full_error_range = float(max(errors) - min(errors)) if errors else None
    robust_error_range = (
        float(np.percentile(errors, 90) - np.percentile(errors, 10))
        if len(errors) >= 8
        else full_error_range
    )
    return {
        "successful_frames": len(errors),
        "requested_frames": len(samples),
        "median_parallel_error_deg": float(np.median(errors)) if errors else None,
        "error_range_deg": robust_error_range,
        "full_error_range_deg": full_error_range,
        "samples": samples,
    }


def normalize_angle_rad(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def measure_parallel(
    camera: CameraSource,
    undistorter: WideCameraUndistorter,
    *,
    frames: int,
    run_dir: Path,
    phase: str,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for index in range(1, frames + 1):
        frame = camera.read()
        if frame is None:
            samples.append(
                {"frame": index, "ok": False, "reason": "camera_read_failed"}
            )
            continue
        undistorted = undistorter.apply(frame)
        result = detect_box_parallel(undistorted)
        sample = {"frame": index, **asdict(result)}
        samples.append(sample)
        annotated = annotate_box_parallel(undistorted, result)
        cv2.imwrite(str(run_dir / f"{phase}_{index:03d}.jpg"), annotated)
    return summarize_samples(samples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Guarded low-amplitude yaw probe with wide-camera measurements before "
            "and after the motion"
        )
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--speed", type=float, default=0.08, help="signed angular rad/s")
    parser.add_argument("--duration", type=float, default=0.25)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--settle-seconds", type=float, default=0.60)
    parser.add_argument("--output-dir", default="wide_box_yaw_probe_runs")
    parser.add_argument("--robot", action="store_true", help="enable real robot output")
    parser.add_argument("--yes", action="store_true", help="confirm the motion area is clear")
    parser.add_argument("--udp-fallback", action="store_true")
    parser.add_argument("--axis-fallback", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        validate_probe_parameters(args.speed, args.duration)
    except ValueError as exc:
        print(json.dumps({"ok": False, "reason": str(exc)}, ensure_ascii=False))
        return 2
    if args.frames < 5:
        print(json.dumps({"ok": False, "reason": "frames must be at least 5"}, ensure_ascii=False))
        return 2
    if not 0.3 <= args.settle_seconds <= 2.0:
        print(
            json.dumps(
                {"ok": False, "reason": "settle-seconds must be within 0.3 to 2.0"},
                ensure_ascii=False,
            )
        )
        return 2

    nominal_yaw_deg = math.degrees(float(args.speed) * float(args.duration))
    if not args.robot:
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "speed_rad_s": float(args.speed),
                    "duration_s": float(args.duration),
                    "nominal_yaw_deg": nominal_yaw_deg,
                    "motion_command_count": 0,
                    "safety_sequence": [
                        "fresh odometry, IMU, and ultrasound",
                        "wide-camera baseline",
                        "autonomous mode",
                        "one bounded yaw pulse",
                        "stop and restore manual mode",
                        "wide-camera post measurement",
                    ],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if not args.yes:
        print(
            json.dumps(
                {"ok": False, "reason": "real robot probe requires --robot --yes"},
                ensure_ascii=False,
            )
        )
        return 2

    config = load_config(args.config_dir)
    project_root = Path(__file__).resolve().parents[2]
    calibration_path = Path(config["camera"]["wide_calibration"])
    if not calibration_path.is_absolute():
        calibration_path = project_root / calibration_path
    undistorter = WideCameraUndistorter.from_file(calibration_path)
    camera_cfg = config["camera"]
    camera = CameraSource(
        camera_cfg["front"],
        dry_run=False,
        flush_grab_frames=int(camera_cfg.get("flush_grab_frames", 4)),
        stale_frame_reconnect_count=int(camera_cfg.get("stale_frame_reconnect_count", 15)),
        digital_zoom=1.0,
        open_timeout_ms=int(camera_cfg.get("open_timeout_ms", 3000)),
        read_timeout_ms=int(camera_cfg.get("read_timeout_ms", 2000)),
        reconnect_backoff_s=float(camera_cfg.get("reconnect_backoff_s", 0.25)),
    )
    motion = Lite3MotionController(
        config,
        dry_run=False,
        udp_fallback=args.udp_fallback,
        axis_fallback=args.axis_fallback,
    )
    state_reader = StateReader(config, dry_run=False)
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "ok": False,
        "dry_run": False,
        "speed_rad_s": float(args.speed),
        "duration_s": float(args.duration),
        "nominal_yaw_deg": nominal_yaw_deg,
        "run_dir": str(run_dir),
        "motion_command_count": 0,
        "manual_mode_restored": False,
        "cleanup_errors": [],
    }
    motion_started = False
    state_started = False
    autonomous_may_be_enabled = False
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

        baseline = measure_parallel(
            camera,
            undistorter,
            frames=args.frames,
            run_dir=run_dir,
            phase="before",
        )
        result["before"] = baseline
        if baseline["successful_frames"] < 5:
            raise RuntimeError("wide-camera baseline has fewer than 5 valid frames")

        def guard(_vx: float, _vy: float, _wz: float) -> None:
            error = state_reader.safety_error(require_ultrasound=True, require_fresh=True)
            if error:
                raise RuntimeError(f"motion guard rejected yaw probe: {error}")
            current = state_reader.state.front_ultrasound_m
            if current is None or not math.isfinite(float(current)) or float(current) < min_valid:
                raise RuntimeError(f"motion guard received invalid ultrasound: {current!r}")

        motion.configure_safety(guard, state_reader.pose, feedback_required=True)
        start_pose = state_reader.pose()
        result["start_pose"] = start_pose
        autonomous_may_be_enabled = True
        motion.set_autonomous()
        time.sleep(0.20)
        motion.hold_velocity(0.0, 0.0, float(args.speed), float(args.duration))
        result["motion_command_count"] = 1
        motion.stop()
        motion.set_manual()
        autonomous_may_be_enabled = False
        result["manual_mode_restored"] = True

        # The Lite3 RTSP stream can retain pre-motion frames even when grab()
        # is used to drain OpenCV's local buffer.  Close the connection before
        # settling so the post measurement starts from a fresh RTSP session.
        camera.release()
        result["camera_reopened_after_motion"] = True
        deadline = time.monotonic() + float(args.settle_seconds)
        while time.monotonic() < deadline:
            state_reader.poll()
            time.sleep(0.02)
        end_pose = state_reader.pose()
        result["end_pose"] = end_pose
        result["measured_yaw_delta_deg"] = math.degrees(
            normalize_angle_rad(end_pose[2] - start_pose[2])
        )

        after = measure_parallel(
            camera,
            undistorter,
            frames=args.frames,
            run_dir=run_dir,
            phase="after",
        )
        result["after"] = after
        if after["successful_frames"] < 5:
            raise RuntimeError("wide-camera post measurement has fewer than 5 valid frames")
        before_error = float(baseline["median_parallel_error_deg"])
        after_error = float(after["median_parallel_error_deg"])
        error_delta = after_error - before_error
        result["parallel_error_delta_deg"] = error_delta
        result["recommended_correction_wz_sign"] = (
            -1 if error_delta > 0.0 and args.speed > 0.0 else
            1 if error_delta < 0.0 and args.speed > 0.0 else
            1 if error_delta > 0.0 and args.speed < 0.0 else
            -1
        )
        result["ok"] = True
    except KeyboardInterrupt:
        result["reason"] = "operator interrupted yaw probe"
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
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
