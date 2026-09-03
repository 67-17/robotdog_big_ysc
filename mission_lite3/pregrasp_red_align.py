from __future__ import annotations

import argparse
import json
import math
import numbers
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Tuple

import cv2
import numpy as np


DEFAULT_ROI = (0.42, 0.55, 0.58, 0.85)
REFERENCE_LINEAR_SIZE_PX = 94.2391
DEFAULT_ALIGN_CONFIG = {
    "roi": DEFAULT_ROI,
    "reference_linear_size_px": REFERENCE_LINEAR_SIZE_PX,
    "strafe_speed_mps": 0.08,
    "pulse_seconds": 0.25,
    "min_pulse_seconds": 0.15,
    "max_pulse_seconds": 1.00,
    "horizontal_error_strafe_gain_m_per_px": 0.00080,
    "settle_seconds": 0.35,
    "loose_motion_min_linear_size_ratio": 0.75,
    "strict_motion_min_linear_size_ratio": 0.70,
    "strict_tracking_min_linear_size_ratio": 0.50,
    "loose_motion_min_center_y_ratio": 0.55,
    "loose_motion_stable_frames": 3,
    "loose_motion_center_tolerance_px": 20.0,
    "loose_motion_size_ratio_tolerance": 0.20,
    "reconnect_camera_after_pulse": True,
    "success_stable_frames": 3,
    "no_red_frame_limit": 5,
    "target_not_found_retries": 3,
    "target_search_enabled": True,
    "target_search_speed_mps": 0.08,
    "target_search_step_seconds": 1.00,
    "target_search_settle_seconds": 0.00,
    "target_search_bilateral_enabled": False,
    "target_search_until_found": False,
    "target_search_each_side_m": 1.00,
    "target_search_max_distance_m": 3.00,
    "target_search_require_odom_progress": False,
    "target_search_min_progress_m": 0.015,
    "target_search_max_stalled_pulses": 3,
    "target_search_max_net_lateral_m": 1.05,
    "target_search_return_to_origin_on_failure": True,
    "target_search_center_band": (0.40, 0.60),
    "target_search_min_distance_m": 0.0,
    "target_search_front_hold_enabled": False,
    "target_search_front_target_m": 0.28,
    "target_search_front_deadband_m": 0.015,
    "target_search_front_hold_kp_s": 0.8,
    "target_search_front_max_vx_mps": 0.025,
    "target_search_front_edge_far_m": 0.60,
    "target_search_front_edge_jump_m": 0.25,
    "target_search_front_edge_confirm_samples": 2,
    "acquire_only": False,
    "acquire_fine_max_strafe_distance_m": 0.15,
    "acquired_target_lost_frame_limit": 2,
    "target_search_odometry_stall_recovery_attempts": 2,
    "target_search_odometry_stall_recovery_settle_seconds": 0.30,
    "target_search_odometry_stall_recovery_pulse_seconds": 1.25,
    "max_pulses": 30,
    "max_seconds": 90.0,
    "max_strafe_distance_m": 0.50,
    "strafe_pose_hold_enabled": False,
    "forward_hold_kp_s": 1.0,
    "max_vx_correction_mps": 0.04,
    "forward_deadband_m": 0.005,
    "max_forward_drift_m": 0.15,
    "yaw_hold_kp_s": 1.2,
    "max_wz_correction_rad_s": 0.12,
    "yaw_deadband_deg": 0.30,
    "max_yaw_drift_deg": 5.0,
}

Roi = Tuple[float, float, float, float]


@dataclass(frozen=True)
class RedTarget:
    source: str
    center_px: Tuple[float, float]
    area_px: float
    frame_size: Tuple[int, int]
    track_id: Optional[int] = None
    stable: bool = False
    confidence: float = 0.0
    box: Optional[Tuple[Tuple[float, float], ...]] = None
    bbox_px: Optional[Tuple[int, int, int, int]] = None

    def __post_init__(self) -> None:
        if self.source not in {"strict", "loose"}:
            raise ValueError("source must be 'strict' or 'loose'")
        if (
            not isinstance(self.area_px, numbers.Real)
            or isinstance(self.area_px, bool)
            or not math.isfinite(float(self.area_px))
            or float(self.area_px) <= 0.0
        ):
            raise ValueError("area_px must be a positive finite number")
        if (
            not isinstance(self.frame_size, (list, tuple))
            or len(self.frame_size) != 2
        ):
            raise ValueError("frame_size must contain two positive integers")
        width, height = self.frame_size
        if any(
            not isinstance(value, numbers.Integral)
            or isinstance(value, bool)
            or int(value) <= 0
            for value in (width, height)
        ):
            raise ValueError("frame_size must contain two positive integers")
        frame_size = (int(width), int(height))
        if (
            not isinstance(self.center_px, (list, tuple))
            or len(self.center_px) != 2
        ):
            raise ValueError("center_px must contain two finite numbers")
        center_x, center_y = self.center_px
        if any(
            not isinstance(value, numbers.Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in (center_x, center_y)
        ):
            raise ValueError("center_px must contain two finite numbers")
        center = (float(center_x), float(center_y))
        if not (
            0.0 <= center[0] < frame_size[0]
            and 0.0 <= center[1] < frame_size[1]
        ):
            raise ValueError("center_px must be within frame_size")
        object.__setattr__(self, "frame_size", frame_size)
        object.__setattr__(self, "center_px", center)

    @property
    def linear_size_px(self) -> float:
        return math.sqrt(self.area_px)


class _TargetSearchFrontEdge(RuntimeError):
    def __init__(self, distance_m: float) -> None:
        self.distance_m = float(distance_m)
        super().__init__(f"front distance jumped to {self.distance_m:.3f}m")


class _TargetSearchTooClose(RuntimeError):
    def __init__(self, distance_m: float) -> None:
        self.distance_m = float(distance_m)
        super().__init__(f"front distance dropped to {self.distance_m:.3f}m")


@dataclass(frozen=True)
class FrameObservation:
    mode: str
    strict_targets: Tuple[RedTarget, ...]
    loose_targets: Tuple[RedTarget, ...]
    undistorted_frame: object

    def __post_init__(self) -> None:
        if self.mode not in {"strict", "loose", "none"}:
            raise ValueError("mode must be 'strict', 'loose', or 'none'")
        strict_targets = tuple(self.strict_targets)
        loose_targets = tuple(self.loose_targets)
        targets = strict_targets + loose_targets
        if any(not isinstance(target, RedTarget) for target in targets):
            raise ValueError("target collections must contain only RedTarget")
        valid_targets = (
            self.mode == "strict" and bool(strict_targets) and not loose_targets
        ) or (
            self.mode == "loose" and not strict_targets and bool(loose_targets)
        ) or (
            self.mode == "none" and not strict_targets and not loose_targets
        )
        if not valid_targets:
            raise ValueError("mode is inconsistent with target collections")
        if any(target.source != self.mode for target in targets):
            raise ValueError("target source is inconsistent with observation mode")
        if targets:
            shape = getattr(self.undistorted_frame, "shape", ())
            if len(shape) < 2:
                raise ValueError("undistorted_frame must have image dimensions")
            expected_frame_size = (int(shape[1]), int(shape[0]))
            if any(
                target.frame_size != expected_frame_size
                for target in targets
            ):
                raise ValueError(
                    "target frame_size must match undistorted_frame"
                )
        object.__setattr__(self, "strict_targets", strict_targets)
        object.__setattr__(self, "loose_targets", loose_targets)


@dataclass(frozen=True)
class AlignAction:
    name: str
    vy: float = 0.0
    reason: str = ""
    pulse_seconds: float = 0.0


@dataclass(frozen=True)
class StrafePoseCorrection:
    vx: float
    wz: float
    forward_drift_m: float
    yaw_error_rad: float


def normalize_angle_rad(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def plan_strafe_pose_correction(
    reference_pose: tuple[float, float, float],
    current_pose: tuple[float, float, float],
    config: Mapping[str, object],
) -> StrafePoseCorrection:
    reference_x, reference_y, reference_yaw = (
        float(value) for value in reference_pose
    )
    current_x, current_y, current_yaw = (float(value) for value in current_pose)
    delta_x = current_x - reference_x
    delta_y = current_y - reference_y
    forward_drift = (
        math.cos(reference_yaw) * delta_x
        + math.sin(reference_yaw) * delta_y
    )
    yaw_error = normalize_angle_rad(current_yaw - reference_yaw)

    forward_deadband = max(0.0, float(config["forward_deadband_m"]))
    yaw_deadband = math.radians(max(0.0, float(config["yaw_deadband_deg"])))
    vx = (
        0.0
        if abs(forward_drift) <= forward_deadband
        else -float(config["forward_hold_kp_s"]) * forward_drift
    )
    wz = (
        0.0
        if abs(yaw_error) <= yaw_deadband
        else -float(config["yaw_hold_kp_s"]) * yaw_error
    )
    max_vx = max(0.0, float(config["max_vx_correction_mps"]))
    max_wz = max(0.0, float(config["max_wz_correction_rad_s"]))
    return StrafePoseCorrection(
        vx=max(-max_vx, min(max_vx, vx)),
        wz=max(-max_wz, min(max_wz, wz)),
        forward_drift_m=forward_drift,
        yaw_error_rad=yaw_error,
    )


@dataclass(frozen=True)
class AlignmentResult:
    ok: bool
    reason: str
    pulse_count: int
    elapsed_seconds: float
    strafe_distance_m: float
    selected_track_id: Optional[int]
    max_abs_forward_drift_m: float = 0.0
    max_abs_yaw_error_deg: float = 0.0


@dataclass(frozen=True)
class DetectOnlyResult:
    ok: bool
    reason: str
    frames_saved: int
    selected_sources: Tuple[str, ...]
    selected_linear_sizes_px: Tuple[Optional[float], ...]
    motion_command_count: int = 0


class AlignmentLogWriter:
    def create_run_dir(self, root: Path, run_name: str) -> Path:
        run_dir = root / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def write_json(self, path: Path, payload: object) -> None:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)

    def write_image(self, path: Path, image) -> None:
        if not cv2.imwrite(str(path), image):
            raise OSError(f"failed to write image: {path}")


def _detect_loose_red_targets(
    frame_bgr,
    red_ranges,
    min_area_px: float,
    cv2_module,
) -> Tuple[RedTarget, ...]:
    hsv = cv2_module.cvtColor(frame_bgr, cv2_module.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for hsv_range in red_ranges:
        lower = np.asarray(hsv_range["lower"], dtype=np.uint8)
        upper = np.asarray(hsv_range["upper"], dtype=np.uint8)
        mask = cv2_module.bitwise_or(
            mask,
            cv2_module.inRange(hsv, lower, upper),
        )

    component_count, _labels, stats, centroids = (
        cv2_module.connectedComponentsWithStats(mask, connectivity=8)
    )
    height, width = frame_bgr.shape[:2]
    targets = []
    for component_index in range(1, component_count):
        area_px = float(stats[component_index, cv2_module.CC_STAT_AREA])
        if area_px < min_area_px:
            continue
        left = int(stats[component_index, cv2_module.CC_STAT_LEFT])
        top = int(stats[component_index, cv2_module.CC_STAT_TOP])
        component_width = int(stats[component_index, cv2_module.CC_STAT_WIDTH])
        component_height = int(stats[component_index, cv2_module.CC_STAT_HEIGHT])
        targets.append(
            RedTarget(
                source="loose",
                center_px=(
                    float(centroids[component_index, 0]),
                    float(centroids[component_index, 1]),
                ),
                area_px=area_px,
                frame_size=(width, height),
                bbox_px=(left, top, component_width, component_height),
            )
        )
    return tuple(targets)


class ArmRedObserver:
    def __init__(
        self,
        *,
        detector_config: Mapping[str, object],
        undistorter,
        detect_candidates_fn: Callable,
        tracker,
        loose_detector_fn: Optional[Callable] = None,
        loose_min_area_px: Optional[float] = None,
        strict_min_center_y_ratio: float = 0.45,
        cv2_module=None,
    ) -> None:
        self._detector_config = detector_config
        self._undistorter = undistorter
        self._detect_candidates = detect_candidates_fn
        self._tracker = tracker
        self._loose_detector = loose_detector_fn
        self._cv2 = cv2 if cv2_module is None else cv2_module
        self._strict_min_center_y_ratio = float(strict_min_center_y_ratio)
        if not 0.0 <= self._strict_min_center_y_ratio < 1.0:
            raise ValueError("strict_min_center_y_ratio must be within [0, 1)")

        configured_min_area = float(
            detector_config["geometry"]["min_area_px"]  # type: ignore[index]
        )
        default_loose_min_area = max(200.0, configured_min_area * 0.25)
        self._loose_min_area_px = float(
            default_loose_min_area
            if loose_min_area_px is None
            else loose_min_area_px
        )
        if (
            not math.isfinite(self._loose_min_area_px)
            or self._loose_min_area_px <= 0.0
        ):
            raise ValueError("loose_min_area_px must be a positive finite number")

    @classmethod
    def from_files(cls, detector_config_path, calibration_path) -> "ArmRedObserver":
        from mission_lite3.arm.runtime.camera_calibration import (
            FrameUndistorter,
            load_calibration,
        )
        from mission_lite3.arm.runtime.strip_detection import (
            StripTracker,
            detect_candidates,
            load_config,
        )

        detector_config = load_config(detector_config_path)
        calibration = load_calibration(calibration_path)
        return cls(
            detector_config=detector_config,
            undistorter=FrameUndistorter(calibration),
            detect_candidates_fn=detect_candidates,
            tracker=StripTracker(detector_config),
        )

    @staticmethod
    def _strict_target(tracked_strip, frame_size: Tuple[int, int]) -> RedTarget:
        box = getattr(tracked_strip, "box", None)
        normalized_box = (
            tuple(
                (float(point[0]), float(point[1]))
                for point in box
            )
            if box is not None
            else None
        )
        return RedTarget(
            source="strict",
            center_px=(
                float(tracked_strip.center_px[0]),
                float(tracked_strip.center_px[1]),
            ),
            area_px=float(tracked_strip.area_px),
            frame_size=frame_size,
            track_id=int(tracked_strip.track_id),
            stable=bool(tracked_strip.stable),
            confidence=float(tracked_strip.confidence),
            box=normalized_box,
        )

    def observe(self, frame) -> FrameObservation:
        undistorted_frame = self._undistorter.apply(frame)
        candidates, _masks = self._detect_candidates(
            undistorted_frame,
            self._detector_config,
        )
        tracked_strips = self._tracker.update(candidates)
        height, width = undistorted_frame.shape[:2]
        frame_size = (width, height)
        strict_targets = tuple(
            self._strict_target(tracked_strip, frame_size)
            for tracked_strip in tracked_strips
            if (
                tracked_strip.color == "red"
                and float(tracked_strip.center_px[1])
                >= self._strict_min_center_y_ratio * height
            )
        )
        if strict_targets:
            return FrameObservation(
                mode="strict",
                strict_targets=strict_targets,
                loose_targets=(),
                undistorted_frame=undistorted_frame,
            )

        red_ranges = self._detector_config["colors"]["red"]  # type: ignore[index]
        if self._loose_detector is None:
            loose_targets = _detect_loose_red_targets(
                undistorted_frame,
                red_ranges,
                self._loose_min_area_px,
                self._cv2,
            )
        else:
            loose_targets = tuple(
                self._loose_detector(
                    undistorted_frame,
                    red_ranges,
                    self._loose_min_area_px,
                )
            )
        return FrameObservation(
            mode="loose" if loose_targets else "none",
            strict_targets=(),
            loose_targets=loose_targets,
            undistorted_frame=undistorted_frame,
        )


def _roi_pixels(roi: Roi, frame_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    width, height = frame_size
    left, top, right, bottom = roi
    if all(0.0 <= value <= 1.0 for value in roi):
        left, right = left * width, right * width
        top, bottom = top * height, bottom * height
    return (
        max(0, min(width - 1, int(round(left)))),
        max(0, min(height - 1, int(round(top)))),
        max(0, min(width - 1, int(round(right)))),
        max(0, min(height - 1, int(round(bottom)))),
    )


def _draw_target(cv2_module, frame, target: RedTarget, *, selected: bool) -> None:
    color = (0, 255, 255) if selected else (
        (0, 200, 0) if target.source == "strict" else (0, 165, 255)
    )
    thickness = 3 if selected else 2
    if target.box is not None:
        points = np.rint(np.asarray(target.box)).astype(np.int32).reshape((-1, 1, 2))
        cv2_module.polylines(frame, [points], True, color, thickness)
    elif target.bbox_px is not None:
        left, top, width, height = target.bbox_px
        cv2_module.rectangle(
            frame,
            (left, top),
            (left + width - 1, top + height - 1),
            color,
            thickness,
        )

    center = (
        int(round(target.center_px[0])),
        int(round(target.center_px[1])),
    )
    cv2_module.circle(frame, center, 4 if selected else 3, color, -1)
    label = f"{target.source} {target.linear_size_px:.1f}px"
    cv2_module.putText(
        frame,
        label,
        (center[0] + 6, max(12, center[1] - 6)),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2_module.LINE_AA,
    )


def annotate_observation(
    observation: FrameObservation,
    roi: Roi,
    selected: Optional[RedTarget] = None,
    action=None,
    cv2_module=None,
):
    drawing = cv2 if cv2_module is None else cv2_module
    annotated = observation.undistorted_frame.copy()
    height, width = annotated.shape[:2]
    left, top, right, bottom = _roi_pixels(roi, (width, height))
    drawing.rectangle(
        annotated,
        (left, top),
        (right, bottom),
        (255, 255, 0),
        2,
    )

    targets = observation.strict_targets + observation.loose_targets
    for target in targets:
        _draw_target(drawing, annotated, target, selected=target is selected)
    if selected is not None and all(target is not selected for target in targets):
        _draw_target(drawing, annotated, selected, selected=True)

    action_name = "none" if action is None else getattr(action, "name", str(action))
    linear_size = "--" if selected is None else f"{selected.linear_size_px:.1f}px"
    status = (
        f"mode={observation.mode} action={action_name} "
        f"linear_size={linear_size}"
    )
    drawing.putText(
        annotated,
        status,
        (12, 24),
        drawing.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        drawing.LINE_AA,
    )
    return annotated


def _normalized_center(target: RedTarget) -> Tuple[float, float]:
    width, height = target.frame_size
    center_x, center_y = target.center_px
    return center_x / width, center_y / height


def _distance_to_roi(target: RedTarget, roi: Roi) -> float:
    center_x, center_y = _normalized_center(target)
    left, top, right, bottom = roi
    dx = max(left - center_x, 0.0, center_x - right)
    dy = max(top - center_y, 0.0, center_y - bottom)
    return math.hypot(dx, dy)


def _inside_roi(target: RedTarget, roi: Roi) -> bool:
    center_x, center_y = _normalized_center(target)
    left, top, right, bottom = roi
    return left <= center_x <= right and top <= center_y <= bottom


def select_strict_target(
    targets: Iterable[RedTarget],
    roi: Roi = DEFAULT_ROI,
    locked_track_id: Optional[int] = None,
) -> Optional[RedTarget]:
    candidates = list(targets)
    if locked_track_id is not None:
        for target in candidates:
            if target.track_id == locked_track_id:
                return target

    inside = [target for target in candidates if _inside_roi(target, roi)]
    if inside:
        return min(
            inside,
            key=lambda target: (
                abs(target.linear_size_px - REFERENCE_LINEAR_SIZE_PX),
                not target.stable,
                -target.confidence,
                -target.area_px,
            ),
        )
    return min(candidates, key=lambda target: _distance_to_roi(target, roi), default=None)


def select_loose_target(
    targets: Iterable[RedTarget],
    roi: Roi = DEFAULT_ROI,
) -> Optional[RedTarget]:
    return min(targets, key=lambda target: _distance_to_roi(target, roi), default=None)


def choose_alignment_action(
    target: RedTarget,
    config: Mapping[str, object],
) -> AlignAction:
    roi = tuple(float(value) for value in config.get("roi", DEFAULT_ROI))
    left, _top, right, _bottom = roi
    width, _height = target.frame_size
    center_x, _center_y = target.center_px
    strafe_speed = abs(float(config.get("strafe_speed_mps", 0.08)))
    legacy_pulse_seconds = float(config.get("pulse_seconds", 0.25))
    min_pulse_seconds = float(
        config.get("min_pulse_seconds", legacy_pulse_seconds)
    )
    max_pulse_seconds = float(
        config.get("max_pulse_seconds", legacy_pulse_seconds)
    )

    def pulse_for_error(error_px: float, normalization_span_px: float) -> float:
        gain = float(config.get("horizontal_error_strafe_gain_m_per_px", 0.0))
        if gain > 0.0 and strafe_speed > 0.0:
            proportional_seconds = abs(float(error_px)) * gain / strafe_speed
            return max(
                min_pulse_seconds,
                min(max_pulse_seconds, proportional_seconds),
            )
        error_ratio = abs(float(error_px)) / max(1.0, normalization_span_px)
        return min_pulse_seconds + min(1.0, error_ratio) * (
            max_pulse_seconds - min_pulse_seconds
        )

    if center_x < left * width:
        error_px = left * width - center_x
        pulse_seconds = pulse_for_error(error_px, left * width)
        return AlignAction(
            "strafe_left",
            vy=strafe_speed,
            reason="target_left_of_roi",
            pulse_seconds=pulse_seconds,
        )
    if center_x > right * width:
        right_span = max(1.0, width - right * width)
        error_px = center_x - right * width
        pulse_seconds = pulse_for_error(error_px, right_span)
        return AlignAction(
            "strafe_right",
            vy=-strafe_speed,
            reason="target_right_of_roi",
            pulse_seconds=pulse_seconds,
        )

    return AlignAction("hold", reason="target_horizontally_aligned")


def _camera_frame(camera):
    frame = camera.read()
    if (
        isinstance(frame, tuple)
        and len(frame) == 2
        and isinstance(frame[0], (bool, np.bool_))
    ):
        ok, image = frame
        return image if ok else None
    return frame


def run_detect_only(
    *,
    camera,
    observer,
    config: Mapping[str, object],
    frame_count: int = 10,
    log_dir="pregrasp_detect_only_runs",
    writer=None,
) -> DetectOnlyResult:
    requested_frames = max(1, int(frame_count))
    align_config = dict(DEFAULT_ALIGN_CONFIG)
    align_config.update(config)
    roi = tuple(float(value) for value in align_config["roi"])
    log_writer = AlignmentLogWriter() if writer is None else writer
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = log_writer.create_run_dir(Path(log_dir), run_name)
    selected_sources = []
    selected_sizes = []
    frames_saved = 0
    reason = "completed"
    log_writer.write_json(
        run_dir / "request.json",
        {
            "mode": "detect_only",
            "frame_count": requested_frames,
            "config": align_config,
            "motion_enabled": False,
        },
    )

    try:
        for frame_index in range(1, requested_frames + 1):
            frame = _camera_frame(camera)
            if frame is None:
                reason = "camera_read_failed"
                break
            observation = observer.observe(frame)
            if observation.strict_targets:
                selected = select_strict_target(
                    observation.strict_targets,
                    roi,
                )
            else:
                selected = select_loose_target(
                    observation.loose_targets,
                    roi,
                )
            action = (
                choose_alignment_action(selected, align_config)
                if selected is not None
                else AlignAction("hold", reason="no_red_target")
            )
            selected_source = "none" if selected is None else selected.source
            selected_size = (
                None if selected is None else selected.linear_size_px
            )
            selected_sources.append(selected_source)
            selected_sizes.append(selected_size)
            decision = {
                "frame_index": frame_index,
                "mode": observation.mode,
                "selected": None if selected is None else asdict(selected),
                "selected_source": selected_source,
                "selected_linear_size_px": selected_size,
                "reference_linear_size_px": float(
                    align_config.get(
                        "reference_linear_size_px",
                        REFERENCE_LINEAR_SIZE_PX,
                    )
                ),
                "planned_lateral_action": asdict(action),
                "motion_command_count": 0,
            }
            log_writer.write_json(
                run_dir / f"decision_{frame_index:04d}.json",
                decision,
            )
            log_writer.write_image(
                run_dir / f"undistorted_{frame_index:04d}.jpg",
                observation.undistorted_frame,
            )
            log_writer.write_image(
                run_dir / f"annotated_{frame_index:04d}.jpg",
                annotate_observation(
                    observation,
                    roi,
                    selected=selected,
                    action=action,
                ),
            )
            frames_saved += 1
    finally:
        release = getattr(camera, "release", None)
        if release is not None:
            release()

    result = DetectOnlyResult(
        ok=frames_saved == requested_frames,
        reason=reason,
        frames_saved=frames_saved,
        selected_sources=tuple(selected_sources),
        selected_linear_sizes_px=tuple(selected_sizes),
    )
    log_writer.write_json(run_dir / "result.json", asdict(result))
    return result


class PregraspRedAligner:
    def __init__(
        self,
        *,
        camera,
        observer,
        motion,
        config: Mapping[str, object],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        pose_provider: Optional[Callable[[], tuple[float, float, float]]] = None,
        search_origin_pose: Optional[tuple[float, float, float]] = None,
        tag_boundary_provider: Optional[Callable[[], Mapping[int, float]]] = None,
        front_distance_provider: Optional[
            Callable[[], Optional[tuple[float, float]]]
        ] = None,
        log_dir=None,
        writer=None,
    ) -> None:
        self._camera = camera
        self._observer = observer
        self._motion = motion
        self._config = dict(DEFAULT_ALIGN_CONFIG)
        self._config.update(config)
        self._clock = clock
        self._sleep = sleep
        self._pose_provider = pose_provider
        self._search_origin_pose = search_origin_pose
        self._tag_boundary_provider = tag_boundary_provider
        self._front_distance_provider = front_distance_provider
        self._log_root = Path(
            "logs/pregrasp_red_align" if log_dir is None else log_dir
        )
        self._writer = AlignmentLogWriter() if writer is None else writer

    def _try_stop(self) -> bool:
        try:
            self._motion.stop()
            return True
        except BaseException:
            return False

    def _safe_stop(self) -> None:
        self._try_stop()

    def _safe_release_camera(self) -> None:
        release = getattr(self._camera, "release", None)
        if release is None:
            return
        try:
            release()
        except Exception:
            pass

    def _safe_create_run_dir(self) -> Optional[Path]:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        try:
            return self._writer.create_run_dir(self._log_root, run_name)
        except Exception:
            return None

    def _safe_write_json(
        self,
        run_dir: Optional[Path],
        filename: str,
        payload: object,
    ) -> None:
        if run_dir is None:
            return
        try:
            self._writer.write_json(run_dir / filename, payload)
        except Exception:
            pass

    def _safe_log_frame(
        self,
        run_dir: Optional[Path],
        frame_index: int,
        observation: FrameObservation,
        selected: Optional[RedTarget],
        action: AlignAction,
        decision: Mapping[str, object],
    ) -> None:
        if run_dir is None:
            return
        try:
            self._writer.write_json(
                run_dir / f"decision_{frame_index:04d}.json",
                decision,
            )
            self._writer.write_image(
                run_dir / f"undistorted_{frame_index:04d}.jpg",
                observation.undistorted_frame,
            )
            annotated = annotate_observation(
                observation,
                tuple(float(value) for value in self._config["roi"]),
                selected=selected,
                action=action,
            )
            self._writer.write_image(
                run_dir / f"annotated_{frame_index:04d}.jpg",
                annotated,
            )
        except Exception:
            pass

    def _read_frame(self):
        return _camera_frame(self._camera)

    def _loose_motion_candidates(
        self,
        targets: Iterable[RedTarget],
    ) -> tuple[RedTarget, ...]:
        return tuple(
            target for target in targets if self._target_is_near_field(target)
        )

    def _target_is_near_field(
        self,
        target: RedTarget,
        *,
        minimum_size_ratio_key: str = "loose_motion_min_linear_size_ratio",
    ) -> bool:
        reference_size = max(
            1e-9,
            float(
                self._config.get(
                    "reference_linear_size_px",
                    REFERENCE_LINEAR_SIZE_PX,
                )
            ),
        )
        minimum_size = reference_size * max(
            0.0,
            float(self._config[minimum_size_ratio_key]),
        )
        minimum_y_ratio = float(
            self._config["loose_motion_min_center_y_ratio"]
        )
        return (
            target.linear_size_px >= minimum_size
            and target.center_px[1]
            >= minimum_y_ratio * target.frame_size[1]
        )

    def _same_loose_motion_candidate(
        self,
        previous: RedTarget,
        current: RedTarget,
    ) -> bool:
        center_tolerance = max(
            0.0,
            float(self._config["loose_motion_center_tolerance_px"]),
        )
        center_distance = math.hypot(
            current.center_px[0] - previous.center_px[0],
            current.center_px[1] - previous.center_px[1],
        )
        size_ratio = current.linear_size_px / max(
            previous.linear_size_px,
            1e-9,
        )
        size_tolerance = max(
            0.0,
            float(self._config["loose_motion_size_ratio_tolerance"]),
        )
        return (
            center_distance <= center_tolerance
            and 1.0 - size_tolerance <= size_ratio <= 1.0 + size_tolerance
        )

    def _refresh_camera_after_pulse(self) -> None:
        if not bool(self._config["reconnect_camera_after_pulse"]):
            return
        self._reconnect_camera("post_pregrasp_strafe")

    def _reconnect_camera(self, reason: str) -> None:
        reconnect = getattr(self._camera, "reconnect", None)
        if reconnect is not None:
            reconnect(reason)

    def run(self) -> AlignmentResult:
        try:
            return self._run()
        finally:
            self._safe_stop()
            self._safe_release_camera()

    def _run(self) -> AlignmentResult:
        start_time = self._clock()
        pulse_count = 0
        strafe_distance_m = 0.0
        alignment_pulse_count = 0
        alignment_strafe_distance_m = 0.0
        target_search_active = bool(
            self._config.get("acquire_only", False)
            and self._config.get("target_search_enabled", True)
        )
        target_search_distance_m = 0.0
        target_search_net_m = 0.0
        target_search_phase = "left"
        target_search_origin_pose: Optional[tuple[float, float, float]] = None
        target_search_stalled_pulses = 0
        target_search_stall_recovery_attempts = 0
        target_search_stall_recovery_pending = False
        target_search_front_distance_m: Optional[float] = None
        target_search_front_sample_at: Optional[float] = None
        target_search_front_far_samples = 0
        target_search_front_edge_latched = False
        target_search_front_edge_pending = False
        target_search_front_too_close = False
        target_search_strict_track_id: Optional[int] = None
        target_search_strict_track_frames = 0
        locked_track_id: Optional[int] = None
        stable_track_id: Optional[int] = None
        stable_frame_count = 0
        last_loose_motion_candidate: Optional[RedTarget] = None
        loose_motion_stable_count = 0
        no_red_frame_count = 0
        target_not_found_retry_count = 0
        frame_index = 0
        max_abs_forward_drift_m = 0.0
        max_abs_yaw_error_deg = 0.0
        alignment_max_strafe_distance_m = float(
            self._config["max_strafe_distance_m"]
        )
        if bool(self._config.get("acquire_only", False)):
            alignment_max_strafe_distance_m = min(
                alignment_max_strafe_distance_m,
                float(
                    self._config.get(
                        "acquire_fine_max_strafe_distance_m",
                        0.05,
                    )
                ),
            )
        reference_pose: Optional[tuple[float, float, float]] = None
        if bool(self._config["strafe_pose_hold_enabled"]):
            if self._pose_provider is None:
                raise RuntimeError("strafe pose hold requires a pose provider")
            reference_pose = tuple(float(value) for value in self._pose_provider())
        run_dir = self._safe_create_run_dir()
        self._safe_write_json(
            run_dir,
            "request.json",
            {"config": dict(self._config)},
        )

        def elapsed_seconds() -> float:
            return max(0.0, float(self._clock() - start_time))

        def valid_pose() -> Optional[tuple[float, float, float]]:
            if self._pose_provider is None:
                return None
            try:
                pose = tuple(float(value) for value in self._pose_provider())
            except Exception:
                return None
            if len(pose) != 3 or not all(math.isfinite(value) for value in pose):
                return None
            return pose

        def target_search_front_velocity(*, interrupt_on_edge: bool) -> float:
            nonlocal target_search_front_distance_m
            nonlocal target_search_front_sample_at
            nonlocal target_search_front_far_samples
            nonlocal target_search_front_edge_latched
            nonlocal target_search_front_edge_pending
            nonlocal target_search_front_too_close

            if (
                not bool(
                    self._config.get("target_search_front_hold_enabled", False)
                )
                or self._front_distance_provider is None
            ):
                return 0.0
            try:
                sample = self._front_distance_provider()
            except Exception:
                return 0.0
            if sample is None or len(sample) != 2:
                return 0.0
            try:
                distance_m = float(sample[0])
                sample_at = float(sample[1])
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(distance_m) or not math.isfinite(sample_at):
                return 0.0

            is_new_sample = sample_at != target_search_front_sample_at
            if is_new_sample:
                target_search_front_sample_at = sample_at
                target_search_front_distance_m = distance_m

            target_m = float(
                self._config.get("target_search_front_target_m", 0.28)
            )
            deadband_m = max(
                0.0,
                float(
                    self._config.get("target_search_front_deadband_m", 0.02)
                ),
            )
            target_search_front_too_close = distance_m < target_m - deadband_m
            far_m = float(
                self._config.get("target_search_front_edge_far_m", 0.60)
            )
            jump_m = float(
                self._config.get("target_search_front_edge_jump_m", 0.25)
            )
            is_far = distance_m >= far_m and distance_m - target_m >= jump_m

            if target_search_front_edge_latched:
                if is_far:
                    return 0.0
                target_search_front_edge_latched = False
                target_search_front_edge_pending = False
                target_search_front_far_samples = 0

            if is_new_sample:
                target_search_front_far_samples = (
                    target_search_front_far_samples + 1 if is_far else 0
                )
            if is_far:
                required = max(
                    1,
                    int(
                        self._config.get(
                            "target_search_front_edge_confirm_samples",
                            2,
                        )
                    ),
                )
                if target_search_front_far_samples >= required:
                    target_search_front_edge_latched = True
                    target_search_front_edge_pending = True
                    if interrupt_on_edge:
                        raise _TargetSearchFrontEdge(distance_m)
                return 0.0

            error_m = distance_m - target_m
            if abs(error_m) <= deadband_m:
                return 0.0
            correction = float(
                self._config.get("target_search_front_hold_kp_s", 0.8)
            ) * error_m
            max_vx = max(
                0.0,
                float(
                    self._config.get("target_search_front_max_vx_mps", 0.025)
                ),
            )
            return max(-max_vx, min(max_vx, correction))

        def search_lateral_m(
            pose: tuple[float, float, float],
        ) -> Optional[float]:
            if target_search_origin_pose is None:
                return None
            origin_x, origin_y, origin_yaw = target_search_origin_pose
            dx = pose[0] - origin_x
            dy = pose[1] - origin_y
            return -math.sin(origin_yaw) * dx + math.cos(origin_yaw) * dy

        def return_search_to_origin() -> tuple[bool, str]:
            if (
                target_search_origin_pose is None
                or not bool(
                    self._config.get(
                        "target_search_return_to_origin_on_failure",
                        True,
                    )
                )
            ):
                return True, "not_required"
            current_pose = valid_pose()
            if current_pose is None:
                return False, "odometry_unavailable"
            current_lateral = search_lateral_m(current_pose)
            if current_lateral is None or not math.isfinite(current_lateral):
                return False, "odometry_invalid"
            if abs(current_lateral) <= 0.02:
                return True, "already_at_origin"
            requested = -current_lateral
            speed = abs(float(self._config["target_search_speed_mps"]))
            try:
                pose_hold = getattr(
                    self._motion,
                    "strafe_distance_pose_hold",
                    None,
                )
                if callable(pose_hold):
                    pose_hold(
                        requested,
                        speed_mps=speed,
                        forward_hold_kp_s=0.0,
                        max_vx_correction_mps=0.0,
                    )
                else:
                    self._motion.hold_velocity(
                        0.0,
                        math.copysign(speed, requested),
                        0.0,
                        abs(requested) / speed,
                    )
            except Exception as exc:
                return False, f"motion_failed:{type(exc).__name__}:{exc}"
            finally:
                self._safe_stop()
            final_pose = valid_pose()
            if final_pose is None:
                return False, "final_odometry_unavailable"
            residual = search_lateral_m(final_pose)
            if residual is None or abs(residual) > 0.05:
                return False, f"residual={0.0 if residual is None else residual:.3f}m"
            return True, f"returned={current_lateral:.3f}m"

        def finish(ok: bool, reason: str) -> AlignmentResult:
            if not ok and target_search_origin_pose is not None:
                self._try_stop()
                returned, return_reason = return_search_to_origin()
                if not returned:
                    reason = f"{reason};search_origin_return_failed:{return_reason}"
            if not self._try_stop():
                ok = False
                reason = "stop_failed"
            result = AlignmentResult(
                ok=ok,
                reason=reason,
                pulse_count=pulse_count,
                elapsed_seconds=elapsed_seconds(),
                strafe_distance_m=strafe_distance_m,
                selected_track_id=locked_track_id,
                max_abs_forward_drift_m=max_abs_forward_drift_m,
                max_abs_yaw_error_deg=max_abs_yaw_error_deg,
            )
            self._safe_write_json(run_dir, "result.json", asdict(result))
            return result

        if target_search_active:
            require_search_odom = bool(
                self._config.get("target_search_require_odom_progress", False)
            )
            target_search_origin_pose = (
                tuple(float(value) for value in self._search_origin_pose)
                if self._search_origin_pose is not None
                else reference_pose
                if reference_pose is not None
                else valid_pose()
            )
            if require_search_odom and target_search_origin_pose is None:
                return finish(False, "target_search_odometry_unavailable")
            current_pose = valid_pose()
            if target_search_origin_pose is not None and current_pose is not None:
                measured = search_lateral_m(current_pose)
                if measured is not None and math.isfinite(measured):
                    target_search_net_m = measured
            print(
                "[pregrasp] acquire-only starts distance-held lateral left search "
                f"net={target_search_net_m:.3f}m",
                flush=True,
            )

        try:
            while True:
                if (
                    not (
                        target_search_active
                        and bool(
                            self._config.get(
                                "target_search_until_found",
                                False,
                            )
                        )
                    )
                    and elapsed_seconds() >= float(self._config["max_seconds"])
                ):
                    return finish(False, "max_seconds")

                frame = self._read_frame()
                if frame is None:
                    return finish(False, "camera_read_failed")

                frame_index += 1
                observation = self._observer.observe(frame)
                selected: Optional[RedTarget]
                strict_motion_qualified = False
                acquired_track_locked = bool(
                    self._config.get("acquire_only", False)
                    and locked_track_id is not None
                    and not target_search_active
                )
                if observation.strict_targets:
                    strict_motion_targets = tuple(
                        target
                        for target in observation.strict_targets
                        if self._target_is_near_field(
                            target,
                            minimum_size_ratio_key=(
                                "strict_tracking_min_linear_size_ratio"
                                if (
                                    locked_track_id is not None
                                    and target.track_id == locked_track_id
                                )
                                else "strict_motion_min_linear_size_ratio"
                            ),
                        )
                    )
                    selected = select_strict_target(
                        strict_motion_targets or observation.strict_targets,
                        tuple(float(value) for value in self._config["roi"]),
                        locked_track_id=locked_track_id,
                    )
                    if (
                        acquired_track_locked
                        and (
                            selected is None
                            or selected.track_id != locked_track_id
                        )
                    ):
                        selected = None
                    strict_motion_qualified = selected in strict_motion_targets
                    no_red_frame_count = (
                        0
                        if strict_motion_qualified
                        else no_red_frame_count + 1
                    )
                    if (
                        strict_motion_qualified
                        and selected is not None
                        and selected.track_id is not None
                        and not target_search_active
                    ):
                        locked_track_id = selected.track_id
                elif observation.loose_targets:
                    if acquired_track_locked:
                        selected = None
                        no_red_frame_count += 1
                    else:
                        motion_candidates = self._loose_motion_candidates(
                            observation.loose_targets
                        )
                        selected = (
                            max(motion_candidates, key=lambda target: target.area_px)
                            if motion_candidates
                            else select_loose_target(
                                observation.loose_targets,
                                tuple(float(value) for value in self._config["roi"]),
                            )
                        )
                        no_red_frame_count = (
                            0 if motion_candidates else no_red_frame_count + 1
                        )
                else:
                    selected = None
                    no_red_frame_count += 1

                if selected is None:
                    action = AlignAction("hold", reason="no_red_target")
                    last_loose_motion_candidate = None
                    loose_motion_stable_count = 0
                elif selected.source == "strict":
                    last_loose_motion_candidate = None
                    loose_motion_stable_count = 0
                    action = (
                        choose_alignment_action(selected, self._config)
                        if (
                            strict_motion_qualified
                            and selected.stable
                            and selected.track_id is not None
                        )
                        else AlignAction(
                            "hold",
                            reason=(
                                "strict_target_not_stable"
                                if strict_motion_qualified
                                else "strict_target_not_motion_qualified"
                            ),
                        )
                    )
                elif selected in self._loose_motion_candidates(
                    observation.loose_targets
                ):
                    if (
                        last_loose_motion_candidate is not None
                        and self._same_loose_motion_candidate(
                            last_loose_motion_candidate,
                            selected,
                        )
                    ):
                        loose_motion_stable_count += 1
                    else:
                        loose_motion_stable_count = 1
                    last_loose_motion_candidate = selected
                    required_loose_frames = max(
                        1,
                        int(self._config["loose_motion_stable_frames"]),
                    )
                    action = (
                        choose_alignment_action(selected, self._config)
                        if loose_motion_stable_count >= required_loose_frames
                        else AlignAction(
                            "hold",
                            reason="loose_target_stabilizing",
                        )
                    )
                else:
                    last_loose_motion_candidate = None
                    loose_motion_stable_count = 0
                    action = AlignAction(
                        "hold",
                        reason="loose_target_not_motion_qualified",
                    )

                search_centered = False
                search_just_acquired = False
                search_candidate_pending_center = False
                if (
                    target_search_active
                    and selected is not None
                    and selected.source == "strict"
                    and selected.track_id is not None
                ):
                    if target_search_strict_track_id == selected.track_id:
                        target_search_strict_track_frames += 1
                    else:
                        target_search_strict_track_id = selected.track_id
                        target_search_strict_track_frames = 1
                search_target_qualified = bool(
                    selected is not None
                    and selected.source == "strict"
                    and selected.track_id is not None
                    and (
                        strict_motion_qualified
                        or target_search_strict_track_frames >= 2
                    )
                )
                if (
                    target_search_active
                    and selected is not None
                    and selected.source == "strict"
                    and selected.track_id is not None
                    and not search_target_qualified
                ):
                    pending_center_fraction = selected.center_px[0] / max(
                        1.0,
                        float(selected.frame_size[0]),
                    )
                    pending_center_band = tuple(
                        float(value)
                        for value in self._config["target_search_center_band"]
                    )
                    search_candidate_pending_center = (
                        pending_center_band[0]
                        <= pending_center_fraction
                        <= pending_center_band[1]
                    )
                if target_search_active and search_target_qualified:
                    assert selected is not None
                    center_fraction = selected.center_px[0] / max(
                        1.0,
                        float(selected.frame_size[0]),
                    )
                    center_band = tuple(
                        float(value)
                        for value in self._config["target_search_center_band"]
                    )
                    search_centered = (
                        center_band[0] <= center_fraction <= center_band[1]
                        and target_search_distance_m
                        >= float(
                            self._config.get(
                                "target_search_min_distance_m",
                                0.0,
                            )
                        )
                    )
                    if search_centered:
                        locked_track_id = selected.track_id
                        target_search_active = False
                        search_just_acquired = True
                        no_red_frame_count = 0
                        stable_track_id = None
                        stable_frame_count = 0
                        action = AlignAction(
                            "hold",
                            reason="target_acquired_in_search_center_band",
                        )
                        self._safe_stop()
                        next_stage = (
                            "final lateral alignment"
                            if bool(self._config.get("acquire_only", False))
                            else "fine alignment"
                        )
                        print(
                            "[pregrasp] left search acquired strict red target; "
                            f"stop base and continue {next_stage} "
                            f"center_x={center_fraction:.3f} "
                            f"travel={target_search_distance_m:.3f}m "
                            f"net={target_search_net_m:.3f}m",
                            flush=True,
                        )

                if target_search_active:
                    search_speed = abs(
                        float(self._config["target_search_speed_mps"])
                    )
                    bilateral_search = bool(
                        self._config.get("target_search_bilateral_enabled", False)
                    )
                    search_until_found = bool(
                        self._config.get("target_search_until_found", False)
                    )
                    total_remaining_m = (
                        math.inf
                        if search_until_found
                        else max(
                            0.0,
                            float(self._config["target_search_max_distance_m"])
                            - target_search_distance_m,
                        )
                    )
                    each_side_m = float(
                        self._config.get(
                            "target_search_each_side_m",
                            self._config["target_search_max_distance_m"],
                        )
                    )
                    tag_centers: Mapping[int, float] = {}
                    if self._tag_boundary_provider is not None:
                        try:
                            tag_centers = self._tag_boundary_provider() or {}
                        except Exception as exc:
                            print(
                                f"[pregrasp] pickup tag boundary unavailable: {exc}",
                                flush=True,
                            )
                    tag_config = self._config.get("pickup_tag_boundary", {})
                    if not isinstance(tag_config, Mapping):
                        tag_config = {}
                    if bool(tag_config.get("enabled", False)) and bool(
                        tag_config.get("search_boundaries_enabled", True)
                    ):
                        left_id = int(tag_config.get("left_tag_id", 5))
                        right_id = int(tag_config.get("right_tag_id", 4))
                        left_x = tag_centers.get(left_id)
                        right_x = tag_centers.get(right_id)
                        if (
                            target_search_phase == "left"
                            and left_x is not None
                            and float(left_x)
                            >= float(tag_config.get("left_stop_center_x_px", 700.0))
                        ):
                            target_search_phase = "right"
                            print(
                                "[pregrasp] ID 5 reached left search boundary; "
                                f"reverse right center_x={float(left_x):.1f}",
                                flush=True,
                            )
                        elif (
                            target_search_phase == "right"
                            and right_x is not None
                            and float(right_x)
                            <= float(tag_config.get("right_stop_center_x_px", 620.0))
                        ):
                            target_search_phase = "left"
                            print(
                                "[pregrasp] ID 4 reached right search boundary; "
                                f"reverse left center_x={float(right_x):.1f}",
                                flush=True,
                            )
                    if not search_until_found and total_remaining_m <= 1e-9:
                        return finish(
                            False,
                            (
                                "target_not_found_after_bilateral_search"
                                if bilateral_search
                                else "target_not_found_after_left_search"
                            ),
                        )
                    if bilateral_search and target_search_phase == "left":
                        side_remaining_m = each_side_m - target_search_net_m
                        if side_remaining_m <= 1e-9:
                            target_search_phase = "right"
                    if bilateral_search and target_search_phase == "right":
                        side_remaining_m = target_search_net_m + each_side_m
                        if side_remaining_m <= 1e-9:
                            if search_until_found:
                                target_search_phase = "left"
                                side_remaining_m = each_side_m - target_search_net_m
                            else:
                                return finish(
                                    False,
                                    "target_not_found_after_bilateral_search",
                                )
                        search_sign = -1.0
                        search_name = "search_right"
                        search_reason = "searching_right_for_red_target"
                    else:
                        side_remaining_m = (
                            each_side_m - target_search_net_m
                            if bilateral_search
                            else total_remaining_m
                        )
                        search_sign = 1.0
                        search_name = "search_left"
                        search_reason = "searching_left_for_red_target"
                    boundary_m = float(
                        self._config.get(
                            "target_search_max_net_lateral_m",
                            each_side_m,
                        )
                    )
                    boundary_remaining_m = (
                        boundary_m - target_search_net_m
                        if search_sign > 0.0
                        else boundary_m + target_search_net_m
                    )
                    boundary_turn_margin_m = max(
                        0.001,
                        2.0
                        * float(
                            self._config.get(
                                "target_search_min_progress_m",
                                0.015,
                            )
                        ),
                    )
                    if boundary_remaining_m <= boundary_turn_margin_m:
                        if search_until_found and bilateral_search:
                            target_search_phase = (
                                "right" if search_sign > 0.0 else "left"
                            )
                            search_sign = -search_sign
                            search_name = (
                                "search_left" if search_sign > 0.0 else "search_right"
                            )
                            search_reason = "search_boundary_reversal"
                            side_remaining_m = max(
                                0.0,
                                boundary_m
                                - target_search_net_m
                                if search_sign > 0.0
                                else boundary_m + target_search_net_m,
                            )
                            boundary_remaining_m = side_remaining_m
                        else:
                            return finish(False, "target_search_field_boundary")
                    requested_search_pulse_seconds = float(
                        self._config["target_search_step_seconds"]
                    )
                    if target_search_stall_recovery_pending:
                        requested_search_pulse_seconds = max(
                            requested_search_pulse_seconds,
                            float(
                                self._config.get(
                                    "target_search_odometry_stall_recovery_pulse_seconds",
                                    1.25,
                                )
                            ),
                        )
                        search_reason = "recovering_target_search_odometry_stall"
                    search_pulse_seconds = min(
                        requested_search_pulse_seconds,
                        side_remaining_m / search_speed,
                        total_remaining_m / search_speed,
                        boundary_remaining_m / search_speed,
                    )
                    action = AlignAction(
                        search_name,
                        vy=search_sign * search_speed,
                        reason=search_reason,
                        pulse_seconds=search_pulse_seconds,
                    )
                if target_search_active and search_candidate_pending_center:
                    action = AlignAction(
                        "hold",
                        reason="strict_center_candidate_confirming",
                    )

                if (
                    selected is not None
                    and selected.source == "strict"
                    and strict_motion_qualified
                    and selected.track_id is not None
                    and selected.stable
                    and action.name == "hold"
                    and not search_just_acquired
                ):
                    if stable_track_id == selected.track_id:
                        stable_frame_count += 1
                    else:
                        stable_track_id = selected.track_id
                        stable_frame_count = 1
                else:
                    stable_track_id = None
                    stable_frame_count = 0

                decision = {
                    "frame_index": frame_index,
                    "mode": observation.mode,
                    "selected": None if selected is None else asdict(selected),
                    "selected_track_id": locked_track_id,
                    "action": asdict(action),
                    "stable_frame_count": stable_frame_count,
                    "loose_motion_stable_count": loose_motion_stable_count,
                    "no_red_frame_count": no_red_frame_count,
                    "target_not_found_retry_count": (
                        target_not_found_retry_count
                    ),
                    "target_search_active": target_search_active,
                    "target_search_centered": search_centered,
                    "target_search_distance_m": target_search_distance_m,
                    "target_search_net_m": target_search_net_m,
                    "target_search_phase": target_search_phase,
                    "pickup_tag_centers": dict(tag_centers) if target_search_active else {},
                    "target_search_front_distance_m": target_search_front_distance_m,
                    "target_search_front_far_samples": target_search_front_far_samples,
                    "target_search_front_edge_latched": target_search_front_edge_latched,
                    "target_search_front_edge_pending": target_search_front_edge_pending,
                    "target_search_strict_track_frames": target_search_strict_track_frames,
                    "target_search_candidate_pending_center": search_candidate_pending_center,
                    "pulse_count": pulse_count,
                    "strafe_distance_m": strafe_distance_m,
                    "elapsed_seconds": elapsed_seconds(),
                }
                self._safe_log_frame(
                    run_dir,
                    frame_index,
                    observation,
                    selected,
                    action,
                    decision,
                )

                search_deadline_exempt = bool(
                    self._config.get("target_search_until_found", False)
                ) and (target_search_active or search_just_acquired)
                if (
                    not search_deadline_exempt
                    and elapsed_seconds() >= float(self._config["max_seconds"])
                ):
                    return finish(False, "max_seconds")

                if not target_search_active and no_red_frame_count >= int(
                    self._config["no_red_frame_limit"]
                ):
                    max_target_retries = max(
                        0,
                        int(self._config["target_not_found_retries"]),
                    )
                    if target_not_found_retry_count < max_target_retries:
                        target_not_found_retry_count += 1
                        no_red_frame_count = 0
                        locked_track_id = None
                        stable_track_id = None
                        stable_frame_count = 0
                        last_loose_motion_candidate = None
                        loose_motion_stable_count = 0
                        self._safe_stop()
                        self._reconnect_camera(
                            "pregrasp_target_not_found_retry_"
                            f"{target_not_found_retry_count}"
                        )
                        continue
                    if bool(self._config["target_search_enabled"]):
                        target_search_active = True
                        target_search_distance_m = 0.0
                        target_search_net_m = 0.0
                        target_search_phase = "left"
                        target_search_stalled_pulses = 0
                        target_search_strict_track_id = None
                        target_search_strict_track_frames = 0
                        require_search_odom = bool(
                            self._config.get(
                                "target_search_require_odom_progress",
                                False,
                            )
                        )
                        target_search_origin_pose = (
                            valid_pose() if require_search_odom else None
                        )
                        if require_search_odom and target_search_origin_pose is None:
                            return finish(
                                False,
                                "target_search_odometry_unavailable",
                            )
                        no_red_frame_count = 0
                        locked_track_id = None
                        stable_track_id = None
                        stable_frame_count = 0
                        last_loose_motion_candidate = None
                        loose_motion_stable_count = 0
                        self._safe_stop()
                        print(
                            "[pregrasp] red target still missing after "
                            f"{max_target_retries} camera reconnects; "
                            "start "
                            + (
                                "continuous left search until target is found"
                                if bool(
                                    self._config.get(
                                        "target_search_until_found",
                                        False,
                                    )
                                )
                                else "bounded bilateral search, each side "
                                f"{float(self._config.get('target_search_each_side_m', 0.0)):.2f}m"
                                if bool(
                                    self._config.get(
                                        "target_search_bilateral_enabled",
                                        False,
                                    )
                                )
                                else "bounded left search up to "
                                f"{float(self._config['target_search_max_distance_m']):.2f}m"
                            ),
                            flush=True,
                        )
                        continue
                    return finish(False, "target_not_found")

                if (
                    search_just_acquired
                    and bool(self._config.get("acquire_only", False))
                    and selected is not None
                    and _inside_roi(
                        selected,
                        tuple(float(value) for value in self._config["roi"]),
                    )
                ):
                    return finish(True, "target_acquired_and_aligned")

                if (
                    bool(self._config.get("acquire_only", False))
                    and locked_track_id is not None
                    and not target_search_active
                    and no_red_frame_count
                    > int(
                        self._config.get(
                            "acquired_target_lost_frame_limit",
                            2,
                        )
                    )
                ):
                    return finish(False, "acquired_target_lost")

                if stable_frame_count >= int(
                    self._config["success_stable_frames"]
                ):
                    return finish(
                        True,
                        (
                            "target_acquired_and_aligned"
                            if bool(self._config.get("acquire_only", False))
                            else "aligned"
                        ),
                    )

                if action.name == "hold":
                    continue

                pulse_seconds = float(action.pulse_seconds)
                pulse_distance = abs(action.vy) * pulse_seconds
                search_pulse = action.name in {"search_left", "search_right"}
                stall_recovery_pulse = bool(
                    search_pulse and target_search_stall_recovery_pending
                )
                if stall_recovery_pulse:
                    target_search_stall_recovery_pending = False
                search_pose_before: Optional[tuple[float, float, float]] = None
                search_net_before = target_search_net_m
                require_search_odom = bool(
                    self._config.get(
                        "target_search_require_odom_progress",
                        False,
                    )
                )
                if search_pulse and require_search_odom:
                    search_pose_before = valid_pose()
                    measured_before = (
                        None
                        if search_pose_before is None
                        else search_lateral_m(search_pose_before)
                    )
                    if measured_before is None:
                        return finish(
                            False,
                            "target_search_odometry_unavailable",
                        )
                    search_net_before = measured_before
                if (
                    not search_pulse
                    and alignment_pulse_count >= int(self._config["max_pulses"])
                ):
                    return finish(False, "max_pulses")
                if (
                    not search_deadline_exempt
                    and
                    elapsed_seconds() + pulse_seconds
                    > float(self._config["max_seconds"])
                ):
                    return finish(False, "max_seconds")
                if (
                    not search_pulse
                    and alignment_strafe_distance_m + pulse_distance
                    > alignment_max_strafe_distance_m
                ):
                    return finish(False, "max_strafe_distance")

                planned_pulse_count = pulse_count + 1
                correction = StrafePoseCorrection(0.0, 0.0, 0.0, 0.0)
                if reference_pose is not None:
                    assert self._pose_provider is not None
                    correction = plan_strafe_pose_correction(
                        reference_pose,
                        self._pose_provider(),
                        self._config,
                    )
                    max_abs_forward_drift_m = max(
                        max_abs_forward_drift_m,
                        abs(correction.forward_drift_m),
                    )
                    max_abs_yaw_error_deg = max(
                        max_abs_yaw_error_deg,
                        abs(math.degrees(correction.yaw_error_rad)),
                    )
                    if (
                        not search_pulse
                        and abs(correction.forward_drift_m)
                        > float(self._config["max_forward_drift_m"])
                    ):
                        return finish(False, "forward_drift_limit")
                    if abs(math.degrees(correction.yaw_error_rad)) > float(
                        self._config["max_yaw_drift_deg"]
                    ):
                        return finish(False, "yaw_drift_limit")
                if search_pulse:
                    correction = StrafePoseCorrection(
                        vx=target_search_front_velocity(
                            interrupt_on_edge=False
                        ),
                        wz=correction.wz,
                        forward_drift_m=correction.forward_drift_m,
                        yaw_error_rad=correction.yaw_error_rad,
                    )
                    if target_search_front_edge_pending:
                        target_search_phase = (
                            "right" if action.vy > 0.0 else "left"
                        )
                        target_search_stalled_pulses = 0
                        target_search_front_edge_pending = False
                        print(
                            "[pregrasp] front ultrasound reached pickup edge; "
                            f"distance={target_search_front_distance_m:.3f}m "
                            f"reverse={target_search_phase}",
                            flush=True,
                        )
                        self._safe_stop()
                        continue
                elif (
                    bool(self._config.get("acquire_only", False))
                    and locked_track_id is not None
                ):
                    correction = StrafePoseCorrection(
                        vx=target_search_front_velocity(
                            interrupt_on_edge=False
                        ),
                        wz=correction.wz,
                        forward_drift_m=correction.forward_drift_m,
                        yaw_error_rad=correction.yaw_error_rad,
                    )
                self._safe_write_json(
                    run_dir,
                    f"pulse_{planned_pulse_count:04d}.json",
                    {
                        "vy": action.vy,
                        "mode": "target_search" if search_pulse else "alignment",
                        "target_search_distance_m": target_search_distance_m,
                        **asdict(correction),
                        "yaw_error_deg": math.degrees(correction.yaw_error_rad),
                    },
                )
                pulse_count = planned_pulse_count
                strafe_distance_m += pulse_distance
                if not search_pulse:
                    alignment_pulse_count += 1
                    alignment_strafe_distance_m += pulse_distance
                motion_error: Optional[BaseException] = None
                front_edge_during_pulse: Optional[_TargetSearchFrontEdge] = None
                too_close_during_pulse: Optional[_TargetSearchTooClose] = None
                try:
                    feedback_hold = getattr(
                        self._motion,
                        "hold_velocity_feedback",
                        None,
                    )
                    if reference_pose is not None and feedback_hold is not None:
                        assert self._pose_provider is not None

                        def velocity_provider() -> tuple[float, float, float]:
                            nonlocal max_abs_forward_drift_m
                            nonlocal max_abs_yaw_error_deg
                            live_correction = plan_strafe_pose_correction(
                                reference_pose,
                                self._pose_provider(),
                                self._config,
                            )
                            forward_abs = abs(live_correction.forward_drift_m)
                            yaw_abs_deg = abs(
                                math.degrees(live_correction.yaw_error_rad)
                            )
                            max_abs_forward_drift_m = max(
                                max_abs_forward_drift_m,
                                forward_abs,
                            )
                            max_abs_yaw_error_deg = max(
                                max_abs_yaw_error_deg,
                                yaw_abs_deg,
                            )
                            if (
                                not search_pulse
                                and forward_abs
                                > float(self._config["max_forward_drift_m"])
                            ):
                                raise RuntimeError("forward_drift_limit")
                            if yaw_abs_deg > float(
                                self._config["max_yaw_drift_deg"]
                            ):
                                raise RuntimeError("yaw_drift_limit")
                            return (
                                (
                                    target_search_front_velocity(
                                        interrupt_on_edge=True
                                    )
                                    if (
                                        search_pulse
                                        or (
                                            bool(
                                                self._config.get(
                                                    "acquire_only",
                                                    False,
                                                )
                                            )
                                            and locked_track_id is not None
                                        )
                                    )
                                    else live_correction.vx
                                ),
                                action.vy,
                                live_correction.wz,
                            )

                        feedback_hold(velocity_provider, pulse_seconds)
                    else:
                        self._motion.hold_velocity(
                            correction.vx,
                            action.vy,
                            correction.wz,
                            pulse_seconds,
                        )
                except BaseException as exc:
                    motion_error = exc
                finally:
                    self._safe_stop()
                if isinstance(motion_error, _TargetSearchTooClose):
                    too_close_during_pulse = motion_error
                    motion_error = None
                if search_pulse and isinstance(
                    motion_error,
                    _TargetSearchFrontEdge,
                ):
                    front_edge_during_pulse = motion_error
                    motion_error = None
                    target_search_phase = (
                        "right" if action.vy > 0.0 else "left"
                    )
                    target_search_stalled_pulses = 0
                    target_search_front_edge_pending = False
                if search_pulse:
                    if require_search_odom:
                        search_pose_after = valid_pose()
                        measured_after = (
                            None
                            if search_pose_after is None
                            else search_lateral_m(search_pose_after)
                        )
                        if measured_after is None:
                            return finish(
                                False,
                                "target_search_odometry_unavailable",
                            )
                        actual_pulse_m = measured_after - search_net_before
                        target_search_net_m = measured_after
                        target_search_distance_m += abs(actual_pulse_m)
                        required_progress_m = min(
                            float(
                                self._config.get(
                                    "target_search_min_progress_m",
                                    0.015,
                                )
                            ),
                            max(0.001, pulse_distance * 0.5),
                        )
                        if actual_pulse_m * action.vy < -required_progress_m * abs(
                            action.vy
                        ):
                            return finish(
                                False,
                                "target_search_odometry_wrong_direction",
                            )
                        if (
                            front_edge_during_pulse is not None
                            or too_close_during_pulse is not None
                        ):
                            target_search_stalled_pulses = 0
                        elif abs(actual_pulse_m) < required_progress_m:
                            target_search_stalled_pulses += 1
                        else:
                            target_search_stalled_pulses = 0
                            target_search_stall_recovery_attempts = 0
                        max_stalled_pulses = int(
                            self._config.get(
                                "target_search_max_stalled_pulses",
                                3,
                            )
                        )
                        stall_needs_recovery = bool(
                            front_edge_during_pulse is None
                            and too_close_during_pulse is None
                            and (
                                stall_recovery_pulse
                                or target_search_stalled_pulses
                                >= max_stalled_pulses
                            )
                        )
                        if stall_needs_recovery:
                            settle_seconds = max(
                                0.0,
                                float(
                                    self._config.get(
                                        "target_search_odometry_stall_recovery_settle_seconds",
                                        0.30,
                                    )
                                ),
                            )
                            self._safe_stop()
                            if settle_seconds > 0.0:
                                self._sleep(settle_seconds)
                            settled_pose = valid_pose()
                            settled_net_m = (
                                None
                                if settled_pose is None
                                else search_lateral_m(settled_pose)
                            )
                            if settled_net_m is None:
                                return finish(
                                    False,
                                    "target_search_odometry_unavailable",
                                )
                            settled_pulse_m = settled_net_m - search_net_before
                            if settled_pulse_m * action.vy < -required_progress_m * abs(
                                action.vy
                            ):
                                return finish(
                                    False,
                                    "target_search_odometry_wrong_direction",
                                )
                            target_search_distance_m += max(
                                0.0,
                                abs(settled_pulse_m) - abs(actual_pulse_m),
                            )
                            target_search_net_m = settled_net_m
                            if abs(settled_pulse_m) >= required_progress_m:
                                target_search_stalled_pulses = 0
                                target_search_stall_recovery_attempts = 0
                                print(
                                    "[pregrasp] target search odometry recovered "
                                    "after stopped refresh "
                                    f"progress={settled_pulse_m:.3f}m",
                                    flush=True,
                                )
                            else:
                                max_recovery_attempts = max(
                                    0,
                                    int(
                                        self._config.get(
                                            "target_search_odometry_stall_recovery_attempts",
                                            2,
                                        )
                                    ),
                                )
                                if (
                                    target_search_stall_recovery_attempts
                                    >= max_recovery_attempts
                                ):
                                    return finish(
                                        False,
                                        "target_search_odometry_stalled",
                                    )
                                target_search_stall_recovery_attempts += 1
                                target_search_stalled_pulses = 0
                                target_search_stall_recovery_pending = True
                                print(
                                    "[pregrasp] target search odometry stalled; "
                                    "schedule bounded recovery pulse "
                                    f"attempt={target_search_stall_recovery_attempts}/"
                                    f"{max_recovery_attempts}",
                                    flush=True,
                                )
                        if abs(target_search_net_m) > float(
                            self._config.get(
                                "target_search_max_net_lateral_m",
                                1.05,
                            )
                        ) + 1e-6:
                            if bool(
                                self._config.get("target_search_until_found", False)
                                and self._config.get(
                                    "target_search_bilateral_enabled", False
                                )
                            ):
                                target_search_phase = (
                                    "right" if target_search_net_m > 0.0 else "left"
                                )
                            else:
                                return finish(False, "target_search_field_boundary")
                    else:
                        target_search_distance_m += pulse_distance
                        target_search_net_m += action.vy * pulse_seconds
                if front_edge_during_pulse is not None:
                    print(
                        "[pregrasp] front ultrasound reached pickup edge "
                        "during strafe; "
                        f"distance={front_edge_during_pulse.distance_m:.3f}m "
                        f"reverse={target_search_phase}",
                        flush=True,
                    )
                    continue
                if too_close_during_pulse is not None:
                    print(
                        "[pregrasp] front ultrasound became too close during "
                        "strafe; stop lateral motion "
                        f"distance={too_close_during_pulse.distance_m:.3f}m",
                        flush=True,
                    )
                    continue
                if motion_error is not None:
                    if isinstance(
                        motion_error,
                        (KeyboardInterrupt, SystemExit),
                    ):
                        raise motion_error
                    if search_pulse:
                        return finish(
                            False,
                            "target_search_motion_failed:"
                            f"{type(motion_error).__name__}:{motion_error}",
                        )
                    raise motion_error
                settle_key = (
                    "target_search_settle_seconds"
                    if search_pulse
                    else "settle_seconds"
                )
                requested_settle = max(0.0, float(self._config[settle_key]))
                remaining_seconds = (
                    math.inf
                    if search_deadline_exempt
                    else max(
                        0.0,
                        float(self._config["max_seconds"]) - elapsed_seconds(),
                    )
                )
                settle_seconds = min(requested_settle, remaining_seconds)
                if settle_seconds > 0.0:
                    self._sleep(settle_seconds)
                if settle_seconds < requested_settle:
                    return finish(False, "max_seconds")
                if (
                    not search_pulse
                    and not (
                        bool(self._config.get("acquire_only", False))
                        and locked_track_id is not None
                    )
                ):
                    self._refresh_camera_after_pulse()
                last_loose_motion_candidate = None
                loose_motion_stable_count = 0
        except BaseException as exc:
            reason = (
                "interrupted"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "exception"
            )
            self._safe_write_json(
                run_dir,
                "result.json",
                asdict(
                    AlignmentResult(
                        ok=False,
                        reason=reason,
                        pulse_count=pulse_count,
                        elapsed_seconds=elapsed_seconds(),
                        strafe_distance_m=strafe_distance_m,
                        selected_track_id=locked_track_id,
                        max_abs_forward_drift_m=max_abs_forward_drift_m,
                        max_abs_yaw_error_deg=max_abs_yaw_error_deg,
                    )
                ),
            )
            raise


def detect_only_main(argv=None) -> int:
    from mission_lite3.camera import CameraSource
    from mission_lite3.config_loader import load_config

    config = load_config()
    arm_config = config.get("arm", {})
    align_config = config.get("pregrasp_red_align", {})
    project_root = Path(__file__).resolve().parent.parent

    def resolve_path(value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else project_root / path

    parser = argparse.ArgumentParser(
        description=(
            "Capture undistorted pregrasp red-target evidence without "
            "sending robot motion commands"
        )
    )
    parser.add_argument(
        "--device",
        default=str(arm_config.get("camera_device", "/dev/video4")),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=int(arm_config.get("camera_width", 1280)),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=int(arm_config.get("camera_height", 720)),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--detector-config",
        default=str(arm_config.get("runtime_config")),
    )
    parser.add_argument(
        "--calibration",
        default=str(arm_config.get("calibration")),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            align_config.get(
                "detect_only_log_dir",
                "pregrasp_detect_only_runs",
            )
        ),
    )
    args = parser.parse_args(argv)
    observer = ArmRedObserver.from_files(
        resolve_path(args.detector_config),
        resolve_path(args.calibration),
    )
    camera = CameraSource(
        args.device,
        args.width,
        args.height,
        dry_run=False,
        flush_grab_frames=2,
        stale_frame_reconnect_count=15,
    )
    result = run_detect_only(
        camera=camera,
        observer=observer,
        config=align_config,
        frame_count=args.frames,
        log_dir=resolve_path(args.output_dir),
    )
    print(json.dumps(asdict(result), ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(detect_only_main())
