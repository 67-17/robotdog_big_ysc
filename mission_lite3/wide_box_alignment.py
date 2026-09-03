from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Optional

import cv2
import numpy as np

from .wide_camera import (
    BoxParallelResult,
    annotate_box_parallel,
    detect_box_parallel,
)
from .box_center_alignment import (
    detect_placement_box_centers,
    detect_placement_letter_candidates,
)


DEFAULT_WIDE_BOX_ALIGN_CONFIG = {
    "enabled": True,
    "frames_per_measurement": 12,
    "min_valid_frames": 8,
    "tolerance_deg": 1.5,
    "max_range_deg": 1.0,
    "correction_speed_rad_s": 0.10,
    "coarse_error_deg": 1.5,
    "coarse_pulse_seconds": 0.35,
    "fine_pulse_seconds": 0.15,
    "error_fraction_per_correction": 0.5,
    "motion_response_gain": 0.5,
    "min_pulse_seconds": 0.15,
    "max_pulse_seconds": 0.80,
    "settle_seconds": 2.0,
    "max_corrections": 6,
    "positive_error_wz_sign": -1,
    "run_log_dir": "wide_box_alignment_runs",
}


@dataclass(frozen=True)
class WideBoxMeasurement:
    ok: bool
    reason: str
    median_error_deg: Optional[float]
    error_range_deg: Optional[float]
    full_error_range_deg: Optional[float]
    successful_frames: int
    requested_frames: int


@dataclass(frozen=True)
class WideBoxAlignmentResult:
    ok: bool
    reason: str
    correction_count: int
    motion_command_count: int
    initial_error_deg: Optional[float]
    final_error_deg: Optional[float]
    final_error_range_deg: Optional[float]
    run_dir: Optional[str]


def detect_placement_row_parallel(
    frame,
    box_center_config: Mapping[str, object],
    *,
    min_row_span_fraction: float = 0.40,
    center_detector=detect_placement_box_centers,
    letter_detector=detect_placement_letter_candidates,
    parallel_detector=detect_box_parallel,
) -> BoxParallelResult:
    """Measure row perspective when at least one placement letter is visible."""
    centers = center_detector(frame, box_center_config)
    letters = letter_detector(frame, box_center_config)
    has_reference = bool(centers.ok and centers.centers) or bool(
        getattr(letters, "candidates", ())
    )
    if not has_reference:
        return BoxParallelResult(
            False,
            "placement_letter_reference_unavailable:"
            f"centers={centers.reason};letters={getattr(letters, 'reason', '')}",
        )
    result = parallel_detector(frame)
    if not result.ok:
        return result
    if result.box_x_range is None:
        return BoxParallelResult(False, "placement_row_span_unavailable")
    width = int(frame.shape[1])
    span = int(result.box_x_range[1]) - int(result.box_x_range[0])
    if span < max(1, int(round(width * float(min_row_span_fraction)))):
        return BoxParallelResult(
            False,
            "placement_row_span_too_small",
            top_angle_deg=result.top_angle_deg,
            seam_angle_deg=result.seam_angle_deg,
            parallel_error_deg=result.parallel_error_deg,
            confidence=result.confidence,
            box_x_range=result.box_x_range,
            top_line=result.top_line,
            seam_line=result.seam_line,
        )
    return result


def summarize_parallel_results(
    results: list[BoxParallelResult],
    *,
    requested_frames: int,
    min_valid_frames: int,
    max_range_deg: float,
) -> WideBoxMeasurement:
    errors = [
        float(result.parallel_error_deg)
        for result in results
        if result.ok and result.parallel_error_deg is not None
    ]
    if not errors:
        return WideBoxMeasurement(
            False,
            "no_valid_parallel_frames",
            None,
            None,
            None,
            0,
            requested_frames,
        )
    full_range = float(max(errors) - min(errors))
    robust_range = (
        float(np.percentile(errors, 90) - np.percentile(errors, 10))
        if len(errors) >= 8
        else full_range
    )
    if len(errors) < min_valid_frames:
        reason = "insufficient_valid_parallel_frames"
        ok = False
    elif robust_range > max_range_deg:
        reason = "parallel_measurement_unstable"
        ok = False
    else:
        reason = "stable"
        ok = True
    return WideBoxMeasurement(
        ok,
        reason,
        float(np.median(errors)),
        robust_range,
        full_range,
        len(errors),
        requested_frames,
    )


class WideBoxAligner:
    def __init__(
        self,
        *,
        camera,
        undistorter,
        motion,
        config: Mapping[str, object],
        detector: Callable[[object], BoxParallelResult] = detect_box_parallel,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._camera = camera
        self._undistorter = undistorter
        self._motion = motion
        self._detector = detector
        self._clock = clock
        self._sleep = sleep
        self._config = dict(DEFAULT_WIDE_BOX_ALIGN_CONFIG)
        self._config.update(config)

    def _safe_stop(self) -> None:
        try:
            self._motion.stop()
        except Exception:
            pass

    def _safe_release_camera(self) -> None:
        release = getattr(self._camera, "release", None)
        if release is not None:
            try:
                release()
            except Exception:
                pass

    def _create_run_dir(self) -> Optional[Path]:
        try:
            root = Path(str(self._config["run_log_dir"]))
            run_dir = root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except Exception:
            return None

    def _write_json(self, run_dir: Optional[Path], name: str, payload: object) -> None:
        if run_dir is None:
            return
        try:
            (run_dir / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _measure(
        self,
        run_dir: Optional[Path],
        measurement_index: int,
    ) -> WideBoxMeasurement:
        requested = max(1, int(self._config["frames_per_measurement"]))
        results: list[BoxParallelResult] = []
        for frame_index in range(1, requested + 1):
            frame = self._camera.read()
            if frame is None:
                results.append(BoxParallelResult(False, "camera_read_failed"))
                continue
            undistorted = self._undistorter.apply(frame)
            result = self._detector(undistorted)
            results.append(result)
            if run_dir is not None:
                try:
                    cv2.imwrite(
                        str(
                            run_dir
                            / f"measure_{measurement_index:02d}_{frame_index:03d}.jpg"
                        ),
                        annotate_box_parallel(undistorted, result),
                    )
                except Exception:
                    pass
        measurement = summarize_parallel_results(
            results,
            requested_frames=requested,
            min_valid_frames=max(1, int(self._config["min_valid_frames"])),
            max_range_deg=max(0.0, float(self._config["max_range_deg"])),
        )
        self._write_json(
            run_dir,
            f"measurement_{measurement_index:02d}.json",
            {
                **asdict(measurement),
                "samples": [asdict(result) for result in results],
            },
        )
        return measurement

    def run(self) -> WideBoxAlignmentResult:
        run_dir = self._create_run_dir()
        correction_count = 0
        initial_error: Optional[float] = None
        final_measurement: Optional[WideBoxMeasurement] = None

        def finish(ok: bool, reason: str) -> WideBoxAlignmentResult:
            measurement = final_measurement
            result = WideBoxAlignmentResult(
                ok=ok,
                reason=reason,
                correction_count=correction_count,
                motion_command_count=correction_count,
                initial_error_deg=initial_error,
                final_error_deg=(
                    None if measurement is None else measurement.median_error_deg
                ),
                final_error_range_deg=(
                    None if measurement is None else measurement.error_range_deg
                ),
                run_dir=None if run_dir is None else str(run_dir),
            )
            self._write_json(run_dir, "result.json", asdict(result))
            return result

        try:
            if not bool(self._config["enabled"]):
                return finish(True, "disabled")
            if not bool(
                getattr(self._undistorter, "calibration", {}).get(
                    "validated_for_control", False
                )
            ):
                return finish(False, "calibration_not_validated_for_control")

            max_corrections = max(0, int(self._config["max_corrections"]))
            for measurement_index in range(1, max_corrections + 2):
                final_measurement = self._measure(run_dir, measurement_index)
                if initial_error is None:
                    initial_error = final_measurement.median_error_deg
                if not final_measurement.ok:
                    return finish(False, final_measurement.reason)
                error = float(final_measurement.median_error_deg)
                if abs(error) <= float(self._config["tolerance_deg"]):
                    return finish(True, "aligned")
                if correction_count >= max_corrections:
                    return finish(False, "max_corrections")

                speed = abs(float(self._config["correction_speed_rad_s"]))
                error_fraction = float(
                    self._config.get("error_fraction_per_correction", 0.0)
                )
                response_gain = float(self._config.get("motion_response_gain", 1.0))
                if error_fraction > 0.0 and response_gain > 0.0:
                    requested_yaw_deg = abs(error) * error_fraction
                    duration = math.radians(requested_yaw_deg) / (
                        speed * response_gain
                    )
                    duration = max(
                        float(self._config.get("min_pulse_seconds", 0.05)),
                        min(
                            float(self._config.get("max_pulse_seconds", 1.0)),
                            duration,
                        ),
                    )
                else:
                    requested_yaw_deg = None
                    duration = (
                        float(self._config["coarse_pulse_seconds"])
                        if abs(error) >= float(self._config["coarse_error_deg"])
                        else float(self._config["fine_pulse_seconds"])
                    )
                positive_sign = int(self._config["positive_error_wz_sign"])
                if positive_sign not in {-1, 1}:
                    return finish(False, "invalid_positive_error_wz_sign")
                wz = math.copysign(speed, positive_sign * error)
                correction_count += 1
                self._write_json(
                    run_dir,
                    f"correction_{correction_count:02d}.json",
                    {
                        "parallel_error_deg": error,
                        "requested_yaw_deg": requested_yaw_deg,
                        "wz_rad_s": wz,
                        "duration_seconds": duration,
                    },
                )
                try:
                    self._motion.hold_velocity(0.0, 0.0, wz, duration)
                finally:
                    self._safe_stop()
                self._safe_release_camera()
                self._sleep(max(0.0, float(self._config["settle_seconds"])))
            return finish(False, "internal_loop_error")
        finally:
            self._safe_stop()
            self._safe_release_camera()
