import argparse
import importlib.util
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "strip_detector_config.json"
DEFAULT_SPEED = 10.0
DEFAULT_ACCELERATION = 10.0
DEFAULT_GRASP_READY_POSE_DEG = {
    "b": -5.624999976,
    "s": -32.281250007,
    "e": 54.681640649,
    "w": -0.703124983,
}
DEFAULT_GRASP_READY_GRIPPER_H = -45.0
DEFAULT_GRASP_READY_TOLERANCE_DEG = 2.0
GRASP_READY_NEGATIVE_E_COMPENSATION_DEG = 5.0
FINAL_APPROACH_ELBOW_FIRST_MIN_DELTA_DEG = 5.0
FINAL_GRASP_ABSOLUTE_JOINTS = ("e",)
CARGO_POSE_STAGE = "MOVE_TO_CARGO_POSE"
CARGO_POSE_NAME = "货运姿态"
DEFAULT_TRANSPORT_POSE_DEG = DEFAULT_GRASP_READY_POSE_DEG
DEFAULT_TRANSPORT_GRIPPER_H = DEFAULT_GRASP_READY_GRIPPER_H
VISION_WINDOW_NAME = "grasp vision"
FINE_ALIGNMENT_CENTER_TOLERANCE_PX = 8.0
FINE_ALIGNMENT_ANGLE_TOLERANCE_DEG = 3.0
SQUARE_FACE_HORIZONTAL_GAIN_DEG_PER_PX = -1.0 / 80.0
SQUARE_FACE_VERTICAL_GAIN_DEG_PER_PX = 1.0 / 120.0
TARGET_VERTICAL_GAIN_DEG_PER_PX = 1.0 / 120.0
SQUARE_FACE_SIZE_GAIN_DEG_PER_RATIO = -4.0
SQUARE_FACE_ANGLE_GAIN_DEG_PER_DEG = -1.0 / 10.0
DEFAULT_FINAL_VIEW_TOO_MUCH_RED_JOINT = "s"
DEFAULT_FINAL_VIEW_TOO_MUCH_RED_DELTA_DEG = 1.0
DEFAULT_FINAL_VIEW_TOO_SMALL_RED_JOINT = "e"
DEFAULT_FINAL_VIEW_TOO_SMALL_RED_DELTA_DEG = 1.0
DEFAULT_FINAL_VIEW_CORRECTION_MAX_STEPS = 3
DEFAULT_FINAL_S_FORWARD_MATCH_JOINT = "s"
DEFAULT_FINAL_S_FORWARD_MATCH_DELTA_DEG = 1.0
DEFAULT_FINAL_S_FORWARD_MATCH_MAX_S_DEG = 55.0
DEFAULT_FINAL_S_FORWARD_MIN_PROGRESS_DEG = 0.1
DEFAULT_FINAL_S_FORWARD_LIMIT_RECOVERY_E_DELTA_DEG = -40.0
DEFAULT_FINAL_S_FORWARD_LIMIT_RECOVERY_POST_MAX_S_DEG = 70.0
DEFAULT_FINAL_S_FORWARD_ALLOWED_FEEDBACK = (
    "target_lost",
    "target_too_far",
    "target_too_near",
)
DEFAULT_INITIAL_RED_SEARCH_JOINT = "b"
DEFAULT_INITIAL_RED_SEARCH_DELTA_DEG = 3.0
DEFAULT_INITIAL_RED_SEARCH_MAX_STEPS = 6
DEFAULT_INITIAL_RED_SEARCH_CENTER_TOLERANCE_PX = 40.0
DEFAULT_INITIAL_RED_SEARCH_SETTLE_ATTEMPTS = 10
DEFAULT_INITIAL_RED_SEARCH_CENTERED_SETTLE_ATTEMPTS = 12
DEFAULT_INITIAL_RED_SEARCH_USE_B_SAFETY_BOUNDS = True
DEFAULT_INITIAL_RED_SEARCH_B_MIN_DEG = -30.0
DEFAULT_INITIAL_RED_SEARCH_B_MAX_DEG = 30.0
DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_RECOVERY_ENABLED = False
DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_MIN_Y_RATIO = 0.65
DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_JOINT = "e"
DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_DELTA_DEG = 2.0
DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_MAX_TOTAL_DEG = 16.0
POST_MOTION_FLUSH_FRAMES = 12
POST_MOTION_DETECT_RETRY_ATTEMPTS = 3
POST_MOTION_TARGET_REACQUIRE_ROUNDS = 3
POST_MOTION_TARGET_REACQUIRE_ATTEMPTS = 6
POST_MOTION_TARGET_REACQUIRE_SETTLE_SECONDS = 0.15
FINAL_APPROACH_RED_VISIBILITY_FRAMES = 3
FINAL_APPROACH_NO_RED_E_RECOVERY_DELTA_DEG = 1.0
FINAL_APPROACH_NO_RED_E_RECOVERY_MAX_STEPS = 5
ALIGNMENT_MAX_DIRECTION_REVERSALS = 2
ALIGNMENT_MAX_STAGNANT_STEPS = 3
ALIGNMENT_MIN_PROGRESS_RATIO = 0.05
LOCKED_TARGET_REBIND_MAX_DISTANCE_PX = 180.0
SQUARE_FACE_TOP_EDGE_AVOID_RATIO = 0.16
REQUIRED_DETECTION_FIELDS = (
    "color",
    "center_px",
    "angle_deg",
    "size_px",
    "confidence",
    "stable_frames",
    "stable",
    "grasp_candidate",
)


@dataclass
class GraspResult:
    ok: bool
    stage: str
    reason: str = ""
    feedback: str = ""
    object_held: bool = False
    target: Optional[Dict[str, Any]] = None
    plan: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TerminalStepError(RuntimeError):
    def __init__(self, stage: str, reason: str):
        super().__init__(reason)
        self.stage = stage

def _load_local_module(name: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_grasp_reference() -> Dict[str, Any]:
    return {
        "start_sequence": {
            "grasp_ready_pose_deg": dict(DEFAULT_GRASP_READY_POSE_DEG),
            "open_gripper_h": DEFAULT_GRASP_READY_GRIPPER_H,
            "pose_tolerance_deg": DEFAULT_GRASP_READY_TOLERANCE_DEG,
        },
        "target": {
            "center_px": [640.0, 520.0],
            "angle_deg": -90.0,
            "size_px": [124.0, 73.0],
        },
        "square_face_target": {
            "center_px": [590.0, 339.5],
            "size_px": [162.0, 137.0],
        },
        "initial_red_search": {
            "enabled": True,
            "joint": DEFAULT_INITIAL_RED_SEARCH_JOINT,
            "delta_deg": DEFAULT_INITIAL_RED_SEARCH_DELTA_DEG,
            "max_steps": DEFAULT_INITIAL_RED_SEARCH_MAX_STEPS,
            "center_tolerance_px": DEFAULT_INITIAL_RED_SEARCH_CENTER_TOLERANCE_PX,
            "settle_attempts": DEFAULT_INITIAL_RED_SEARCH_SETTLE_ATTEMPTS,
            "centered_settle_attempts": DEFAULT_INITIAL_RED_SEARCH_CENTERED_SETTLE_ATTEMPTS,
            "use_b_safety_bounds": DEFAULT_INITIAL_RED_SEARCH_USE_B_SAFETY_BOUNDS,
            "b_min_deg": DEFAULT_INITIAL_RED_SEARCH_B_MIN_DEG,
            "b_max_deg": DEFAULT_INITIAL_RED_SEARCH_B_MAX_DEG,
            "lower_middle_e_recovery_enabled": (
                DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_RECOVERY_ENABLED
            ),
            "lower_middle_min_y_ratio": DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_MIN_Y_RATIO,
            "lower_middle_e_joint": DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_JOINT,
            "lower_middle_e_delta_deg": DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_DELTA_DEG,
            "lower_middle_e_max_total_deg": (
                DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_MAX_TOTAL_DEG
            ),
        },
        "final_view_match": {
            "enabled": True,
            "reference_image": "grasp_samples/20260626_172257_318_019719.jpg",
            "center_tolerance_px": [90.0, 90.0],
            "size_ratio_tolerance": 0.45,
            "area_ratio_tolerance": 0.65,
            "angle_tolerance_deg": 20.0,
            "min_area_px": 5000.0,
            "allow_less_visible_red": True,
            "min_visible_long_side_ratio": 0.85,
            "max_visible_depth_ratio": 1.05,
        },
        "final_view_correction": {
            "enabled": True,
            "too_much_red_joint": DEFAULT_FINAL_VIEW_TOO_MUCH_RED_JOINT,
            "too_much_red_delta_deg": DEFAULT_FINAL_VIEW_TOO_MUCH_RED_DELTA_DEG,
            "too_small_red_joint": DEFAULT_FINAL_VIEW_TOO_SMALL_RED_JOINT,
            "too_small_red_delta_deg": DEFAULT_FINAL_VIEW_TOO_SMALL_RED_DELTA_DEG,
            "max_steps": DEFAULT_FINAL_VIEW_CORRECTION_MAX_STEPS,
        },
        "visual_servo": {
            "square_face": {
                "horizontal_joint": "b",
                "horizontal_tolerance_px": 20.0,
                "horizontal_gain_deg_per_px": -0.03333333333333333,
                "horizontal_max_jog_deg": 6.0,
                "horizontal_min_jog_deg": 3.0,
                "vertical_joint": "s",
                "vertical_tolerance_px": 20.0,
                "vertical_gain_deg_per_px": 0.008333333333333333,
                "vertical_max_jog_deg": 3.0,
                "vertical_min_jog_deg": 3.0,
                "vertical_priority_above_px": 60.0,
                "size_joint": "s",
                "size_ratio_tolerance": 0.2,
                "size_gain_deg_per_ratio": -4.0,
                "size_max_jog_deg": 2.0,
                "angle_joint": "w",
                "angle_tolerance_deg": 8.0,
                "angle_gain_deg_per_deg": -0.1,
                "angle_max_jog_deg": 1.0,
                "final_pose_adjustment_enabled": True,
                "final_pose_max_adjust_deg": 2.0,
            }
        },
        "approach_sequence": {
            "requires_reteach": True,
            "alignment_offset_scale": {
                "b": 0.5,
                "s": 0.5,
                "e": 0.5,
                "w": 0.5,
            },
            "pose_tolerance_deg": 2.5,
        },
        "terminal_sequence": {
            "close_gripper_h": 25.0,
        },
    }


def load_grasp_reference(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_ready_target(target: Mapping[str, Any]) -> bool:
    return (
        target.get("color") == "red"
        and bool(target.get("stable"))
        and bool(target.get("grasp_candidate"))
        and bool(target.get("angle_reliable"))
    )


def _field_as_float_pair(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return default
    return default


def _select_grasp_target_fallback(
    detections: Iterable[Any],
    *,
    image_size: Optional[Tuple[int, int]] = None,
) -> Optional[Any]:
    red_detections = [
        detection
        for detection in detections
        if str(getattr(detection, "color", "")) == "red"
    ]
    if not red_detections:
        return None

    if image_size is not None:
        image_center = (float(image_size[0]) / 2.0, float(image_size[1]) / 2.0)
    else:
        image_center = (0.0, 0.0)

    def score(detection: Any) -> Tuple[float, float, float, float]:
        center = _field_as_float_pair(
            getattr(detection, "center_px", None),
            default=image_center,
        )
        center_distance = math.hypot(
            center[0] - image_center[0],
            center[1] - image_center[1],
        )
        ready_rank = 0.0 if bool(getattr(detection, "grasp_candidate", False)) else 1.0
        stable_rank = 0.0 if bool(getattr(detection, "stable", False)) else 1.0
        confidence_rank = -float(getattr(detection, "confidence", 0.0))
        return (ready_rank, stable_rank, center_distance, confidence_rank)

    return min(red_detections, key=score)


def _select_square_face_target(
    detections: Iterable[Any],
    reference: Mapping[str, Any],
    *,
    image_size: Optional[Tuple[int, int]] = None,
    locked_track_id: Optional[Any] = None,
) -> Optional[Any]:
    red_detections = [
        detection
        for detection in detections
        if str(getattr(detection, "color", "")) == "red"
    ]
    if locked_track_id is not None:
        return next(
            (
                detection
                for detection in red_detections
                if getattr(detection, "track_id", None) == locked_track_id
            ),
            None,
        )
    if not red_detections:
        return None

    reference_center = _field_as_float_pair(
        reference.get("center_px"),
        default=(0.0, 0.0),
    )
    reference_size = _field_as_float_pair(
        reference.get("size_px"),
        default=(1.0, 1.0),
    )
    reference_long = max(reference_size)
    reference_short = max(1e-6, min(reference_size))
    reference_aspect = reference_long / reference_short
    image_width = max(1.0, float(image_size[0])) if image_size else max(1.0, reference_long)
    image_height = max(1.0, float(image_size[1])) if image_size else max(1.0, reference_long)

    def score(detection: Any) -> Tuple[float, float, float, float, float]:
        size = _field_as_float_pair(
            getattr(detection, "size_px", None),
            default=(0.0, 0.0),
        )
        long_side = max(size)
        short_side = min(size)
        if long_side <= 0.0 or short_side <= 0.0:
            aspect_error = float("inf")
            size_error = float("inf")
        else:
            aspect = long_side / short_side
            aspect_error = abs(math.log(aspect / reference_aspect))
            size_error = abs(math.log(long_side / reference_long)) + abs(
                math.log(short_side / reference_short)
            )
        center = _field_as_float_pair(
            getattr(detection, "center_px", None),
            default=reference_center,
        )
        center_distance = math.hypot(
            (center[0] - reference_center[0]) / image_width,
            (center[1] - reference_center[1]) / image_height,
        )
        stable_rank = 0.0 if bool(getattr(detection, "stable", False)) else 1.0
        top_edge_rank = 1.0 if center[1] < image_height * SQUARE_FACE_TOP_EDGE_AVOID_RATIO else 0.0
        confidence_rank = -float(getattr(detection, "confidence", 0.0))
        return (
            stable_rank,
            top_edge_rank,
            aspect_error,
            center_distance,
            size_error,
            confidence_rank,
        )

    return min(red_detections, key=score)


class StripCameraVision:
    def __init__(
        self,
        *,
        capture: Any,
        strip_detection: Any,
        config: Mapping[str, Any],
        frames_per_detect: int = 5,
        undistorter: Optional[Any] = None,
        show_window: bool = False,
        cv2_module: Optional[Any] = None,
        window_name: str = VISION_WINDOW_NAME,
    ):
        self.capture = capture
        self.strip_detection = strip_detection
        self.config = config
        self.frames_per_detect = max(1, int(frames_per_detect))
        self.undistorter = undistorter
        self.tracker = strip_detection.StripTracker(config)
        self.show_window = bool(show_window)
        self.cv2_module = cv2_module
        self.window_name = window_name
        self._window_opened = False
        self._last_display_frame: Optional[Any] = None
        self._last_frame_bgr: Optional[Any] = None
        self.display_error: Optional[str] = None
        self._abort_requested = False
        self._external_abort_checker: Optional[Callable[[], bool]] = None
        self._run_log_directory: Optional[Path] = None
        self._detection_log_index = 0
        self._red_visibility_log_index = 0
        self._ready_target_reference: Optional[Dict[str, Any]] = None
        self._locked_target_id: Optional[Any] = None
        self._locked_target_snapshot: Optional[Dict[str, Any]] = None
        self._last_loose_red_hint: Optional[Dict[str, Any]] = None
        self._last_red_visibility_check: Optional[Dict[str, Any]] = None
        self.log_errors: List[str] = []
        if self.show_window and self.cv2_module is None:
            try:
                import cv2

                self.cv2_module = cv2
            except ImportError as exc:
                raise RuntimeError("opencv-python is required for --show-vision") from exc

    def configure_ready_target_selection(self, reference: Mapping[str, Any]) -> None:
        self._ready_target_reference = dict(reference)
        self._locked_target_id = None
        self._locked_target_snapshot = None

    def _select_target(
        self,
        detections: Iterable[Any],
        image_size: Optional[Tuple[int, int]],
    ) -> Optional[Any]:
        detections = list(detections)
        if self._ready_target_reference:
            selected = _select_square_face_target(
                detections,
                self._ready_target_reference,
                image_size=image_size,
                locked_track_id=self._locked_target_id,
            )
            if selected is None and self._locked_target_id is not None:
                selected = self._rebind_locked_target(detections)
            return selected
        selector = getattr(self.strip_detection, "select_grasp_target", None)
        if selector is not None:
            return selector(
                detections,
                image_size=image_size,
            )
        return _select_grasp_target_fallback(
            detections,
            image_size=image_size,
        )

    def _rebind_locked_target(self, detections: Iterable[Any]) -> Optional[Any]:
        snapshot = self._locked_target_snapshot or {}
        previous_center = snapshot.get("center_px")
        if (
            not isinstance(previous_center, (list, tuple))
            or len(previous_center) != 2
        ):
            return None

        previous_u = float(previous_center[0])
        previous_v = float(previous_center[1])
        candidates = [
            detection
            for detection in detections
            if str(getattr(detection, "color", "")) == "red"
            and bool(getattr(detection, "stable", False))
        ]
        if not candidates:
            return None

        def distance(detection: Any) -> float:
            center = getattr(detection, "center_px", None)
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                return float("inf")
            return math.hypot(float(center[0]) - previous_u, float(center[1]) - previous_v)

        selected = min(candidates, key=distance)
        if distance(selected) > LOCKED_TARGET_REBIND_MAX_DISTANCE_PX:
            return None
        self._locked_target_id = getattr(selected, "track_id", None)
        return selected

    @staticmethod
    def _detection_color(detection: Any) -> Tuple[int, int, int]:
        if getattr(detection, "color", None) == "green":
            return (0, 255, 0)
        if getattr(detection, "color", None) == "red":
            return (0, 0, 255)
        return (255, 255, 255)

    @staticmethod
    def _point(point: Any) -> Tuple[int, int]:
        return (int(round(float(point[0]))), int(round(float(point[1]))))

    def _annotate_detection_frame(
        self,
        frame_bgr: Any,
        detections: Iterable[Any],
        selected: Optional[Any],
    ) -> Any:
        cv2_module = self.cv2_module
        annotated = frame_bgr.copy() if hasattr(frame_bgr, "copy") else frame_bgr
        selected_id = getattr(selected, "track_id", None)
        for detection in detections:
            color = self._detection_color(detection)
            box = getattr(detection, "box", None)
            if box is not None and hasattr(cv2_module, "line"):
                points = [self._point(point) for point in box]
                for index, start in enumerate(points):
                    end = points[(index + 1) % len(points)]
                    cv2_module.line(
                        annotated,
                        start,
                        end,
                        color,
                        2,
                        getattr(cv2_module, "LINE_AA", 16),
                    )
            center = getattr(detection, "center_px", None)
            if center is None:
                continue
            center_point = self._point(center)
            radius = 6 if getattr(detection, "track_id", None) == selected_id else 3
            if hasattr(cv2_module, "circle"):
                cv2_module.circle(
                    annotated,
                    center_point,
                    radius,
                    color,
                    -1,
                    getattr(cv2_module, "LINE_AA", 16),
                )
            if hasattr(cv2_module, "putText"):
                label = (
                    f"id={getattr(detection, 'track_id', '?')} "
                    f"{getattr(detection, 'color', '?')} "
                    f"stable={int(bool(getattr(detection, 'stable', False)))}"
                )
                cv2_module.putText(
                    annotated,
                    label,
                    (center_point[0] + 8, center_point[1] - 8),
                    getattr(cv2_module, "FONT_HERSHEY_SIMPLEX", 0),
                    0.45,
                    color,
                    1,
                    getattr(cv2_module, "LINE_AA", 16),
                )
        return annotated

    def _show_detection_frame(
        self,
        frame_bgr: Any,
        detections: Iterable[Any],
        selected: Optional[Any],
    ) -> None:
        if not self.show_window or self.cv2_module is None:
            return
        try:
            annotated = self._annotate_detection_frame(frame_bgr, detections, selected)
            self._last_display_frame = annotated
            self.cv2_module.imshow(self.window_name, annotated)
            self._window_opened = True
            key = int(self.cv2_module.waitKey(1)) & 0xFF
        except Exception as exc:
            self.display_error = str(exc)
            self.show_window = False
            return
        self._handle_window_key(key)

    def _handle_window_key(self, key: int) -> bool:
        if key in (ord("q"), ord("Q"), 27):
            self._abort_requested = True
            self.show_window = False
            return True
        return False

    def set_abort_checker(self, checker: Optional[Callable[[], bool]]) -> None:
        self._external_abort_checker = checker

    def set_run_log_directory(self, directory: Path) -> None:
        self._run_log_directory = Path(directory)
        self._run_log_directory.mkdir(parents=True, exist_ok=True)

    def _write_detection_frames(
        self,
        frame_bgr: Any,
        detections: Iterable[Any],
        selected: Optional[Any],
    ) -> None:
        if self._run_log_directory is None or self.cv2_module is None:
            return
        self._detection_log_index += 1
        prefix = f"detection_{self._detection_log_index:03d}"
        raw_path = self._run_log_directory / f"{prefix}_raw.jpg"
        annotated_path = self._run_log_directory / f"{prefix}_annotated.jpg"
        try:
            annotated = self._annotate_detection_frame(
                frame_bgr,
                detections,
                selected,
            )
            raw_ok = bool(self.cv2_module.imwrite(str(raw_path), frame_bgr))
            annotated_ok = bool(
                self.cv2_module.imwrite(str(annotated_path), annotated)
            )
            if not raw_ok or not annotated_ok:
                self.log_errors.append(f"failed to write detection frames: {prefix}")
        except Exception as exc:
            self.log_errors.append(f"failed to write detection frames: {exc}")

    def _annotate_red_visibility_frame(
        self,
        frame_bgr: Any,
        hint: Optional[Mapping[str, Any]],
        result: str,
    ) -> Any:
        cv2_module = self.cv2_module
        annotated = frame_bgr.copy() if hasattr(frame_bgr, "copy") else frame_bgr
        bbox = hint.get("bbox_px") if isinstance(hint, Mapping) else None
        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) >= 4
            and hasattr(cv2_module, "line")
        ):
            x, y, width, height = [int(round(float(value))) for value in bbox[:4]]
            points = [
                (x, y),
                (x + width, y),
                (x + width, y + height),
                (x, y + height),
            ]
            for index, start in enumerate(points):
                cv2_module.line(
                    annotated,
                    start,
                    points[(index + 1) % len(points)],
                    (0, 0, 255),
                    2,
                    getattr(cv2_module, "LINE_AA", 16),
                )
        if hasattr(cv2_module, "putText"):
            cv2_module.putText(
                annotated,
                f"final approach red: {result}",
                (24, 38),
                getattr(cv2_module, "FONT_HERSHEY_SIMPLEX", 0),
                0.7,
                (0, 0, 255) if result == "visible" else (0, 255, 255),
                2,
                getattr(cv2_module, "LINE_AA", 16),
            )
        return annotated

    def _write_red_visibility_frames(
        self,
        frame_bgr: Optional[Any],
        hint: Optional[Mapping[str, Any]],
        result: str,
        check_index: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        if (
            frame_bgr is None
            or self._run_log_directory is None
            or self.cv2_module is None
        ):
            return None, None
        prefix = f"final_approach_visibility_{check_index:03d}"
        raw_path = self._run_log_directory / f"{prefix}_raw.jpg"
        annotated_path = self._run_log_directory / f"{prefix}_annotated.jpg"
        try:
            annotated = self._annotate_red_visibility_frame(frame_bgr, hint, result)
            raw_ok = bool(self.cv2_module.imwrite(str(raw_path), frame_bgr))
            annotated_ok = bool(
                self.cv2_module.imwrite(str(annotated_path), annotated)
            )
            if not raw_ok or not annotated_ok:
                self.log_errors.append(
                    f"failed to write final approach visibility frames: {prefix}"
                )
                return None, None
        except Exception as exc:
            self.log_errors.append(
                f"failed to write final approach visibility frames: {exc}"
            )
            return None, None
        return str(raw_path), str(annotated_path)

    def _record_red_visibility_check(
        self,
        *,
        visible: Optional[bool],
        inspected_frames: int,
        max_frames: int,
        frame_bgr: Optional[Any],
        hint: Optional[Mapping[str, Any]],
    ) -> None:
        self._red_visibility_log_index += 1
        result = {
            True: "visible",
            False: "not_visible",
            None: "camera_unknown",
        }[visible]
        raw_image, annotated_image = self._write_red_visibility_frames(
            frame_bgr,
            hint,
            result,
            self._red_visibility_log_index,
        )
        self._last_red_visibility_check = {
            "result": result,
            "visible": visible,
            "max_frames": int(max_frames),
            "inspected_frames": int(inspected_frames),
            "loose_red_hint": dict(hint) if isinstance(hint, Mapping) else None,
            "raw_image": raw_image,
            "annotated_image": annotated_image,
        }

    def last_red_visibility_check(self) -> Optional[Dict[str, Any]]:
        if self._last_red_visibility_check is None:
            return None
        details = dict(self._last_red_visibility_check)
        hint = details.get("loose_red_hint")
        if isinstance(hint, Mapping):
            details["loose_red_hint"] = dict(hint)
        return details

    def _find_loose_red_hint(
        self,
        frame_bgr: Any,
        image_size: Optional[Tuple[int, int]],
    ) -> Optional[Dict[str, Any]]:
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None
        if image_size is None:
            return None
        colors = self.config.get("colors", {})
        red_ranges = colors.get("red", []) if isinstance(colors, Mapping) else []
        if not red_ranges:
            return None
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for hsv_range in red_ranges:
            lower = np.asarray(hsv_range["lower"], dtype=np.uint8)
            upper = np.asarray(hsv_range["upper"], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        morphology = self.config.get("morphology", {})
        open_kernel = int(morphology.get("open_kernel", 3)) if isinstance(morphology, Mapping) else 3
        close_kernel = int(morphology.get("close_kernel", 5)) if isinstance(morphology, Mapping) else 5
        if open_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel, open_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        if close_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        height, width = mask.shape[:2]
        roi = self.config.get("roi", [0.0, 0.0, 1.0, 1.0])
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            x1, y1, x2, y2 = [float(value) for value in roi]
            if all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
                left, top, right, bottom = (
                    int(x1 * width),
                    int(y1 * height),
                    int(x2 * width),
                    int(y2 * height),
                )
            else:
                left, top, right, bottom = int(x1), int(y1), int(x2), int(y2)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            roi_mask[max(0, top):min(height, bottom), max(0, left):min(width, right)] = 255
            mask = cv2.bitwise_and(mask, roi_mask)
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return None
        geometry = self.config.get("geometry", {})
        formal_min_area = (
            float(geometry.get("min_area_px", 1200.0))
            if isinstance(geometry, Mapping)
            else 1200.0
        )
        min_area = max(100.0, formal_min_area * 0.25)
        best_label = None
        best_area = 0.0
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area and area > best_area:
                best_label = label
                best_area = area
        if best_label is None:
            return None
        x = float(stats[best_label, cv2.CC_STAT_LEFT])
        y = float(stats[best_label, cv2.CC_STAT_TOP])
        w = float(stats[best_label, cv2.CC_STAT_WIDTH])
        h = float(stats[best_label, cv2.CC_STAT_HEIGHT])
        return {
            "color": "red",
            "center_px": [
                float(centroids[best_label][0]),
                float(centroids[best_label][1]),
            ],
            "area_px": best_area,
            "bbox_px": [x, y, w, h],
            "image_size": [int(image_size[0]), int(image_size[1])],
        }

    def loose_red_hint(self) -> Optional[Dict[str, Any]]:
        return dict(self._last_loose_red_hint) if self._last_loose_red_hint else None

    def abort_requested(self) -> bool:
        if self._abort_requested:
            return True
        return bool(
            self._external_abort_checker is not None
            and self._external_abort_checker()
        )

    def hold_window_until_closed(self) -> None:
        if (
            not self.show_window
            or not self._window_opened
            or self._last_display_frame is None
            or self.cv2_module is None
        ):
            return
        while not self.abort_requested():
            try:
                self.cv2_module.imshow(self.window_name, self._last_display_frame)
                key = int(self.cv2_module.waitKey(50)) & 0xFF
            except Exception as exc:
                self.display_error = str(exc)
                self.show_window = False
                return
            if self._handle_window_key(key):
                return

    def detect(self) -> Optional[Dict[str, Any]]:
        last_selected: Optional[Dict[str, Any]] = None
        last_frame = None
        last_detections: Iterable[Any] = ()
        last_selected_object = None
        image_size = None
        for _ in range(self.frames_per_detect):
            ok, frame_bgr = self.capture.read()
            if not ok or frame_bgr is None:
                break
            if self.undistorter is not None:
                frame_bgr = self.undistorter.apply(frame_bgr)
            self._last_frame_bgr = frame_bgr
            if hasattr(frame_bgr, "shape") and len(frame_bgr.shape) >= 2:
                height, width = frame_bgr.shape[:2]
                image_size = (width, height)
            self._last_loose_red_hint = self._find_loose_red_hint(frame_bgr, image_size)
            candidates, _masks = self.strip_detection.detect_candidates(frame_bgr, self.config)
            detections = self.tracker.update(candidates)
            selected = self._select_target(detections, image_size)
            last_frame = frame_bgr
            last_detections = detections
            last_selected_object = selected
            self._show_detection_frame(frame_bgr, detections, selected)
            if selected is not None:
                selected_dict = self.strip_detection.tracked_strip_to_dict(selected)
                last_selected = selected_dict
                if (
                    self._ready_target_reference
                    and selected_dict.get("color") == "red"
                    and bool(selected_dict.get("stable"))
                ):
                    if self._locked_target_id is None:
                        self._locked_target_id = selected_dict.get("track_id")
                    if selected_dict.get("track_id") == self._locked_target_id:
                        self._locked_target_snapshot = dict(selected_dict)
                if _is_ready_target(selected_dict):
                    self._write_detection_frames(frame_bgr, detections, selected)
                    return selected_dict
        if last_frame is not None:
            self._write_detection_frames(
                last_frame,
                last_detections,
                last_selected_object,
            )
        return last_selected

    def flush_after_motion(self, frames: Optional[int] = None) -> None:
        frame_count = max(1, int(frames if frames is not None else POST_MOTION_FLUSH_FRAMES))
        last_frame = None
        for _ in range(frame_count):
            ok, frame_bgr = self.capture.read()
            if not ok or frame_bgr is None:
                break
            if self.undistorter is not None:
                frame_bgr = self.undistorter.apply(frame_bgr)
            self._last_frame_bgr = frame_bgr
            last_frame = frame_bgr
        if self.show_window and last_frame is not None and self.cv2_module is not None:
            try:
                self._last_display_frame = last_frame
                self.cv2_module.imshow(self.window_name, last_frame)
                self._window_opened = True
                key = int(self.cv2_module.waitKey(1)) & 0xFF
                self._handle_window_key(key)
            except Exception as exc:
                self.display_error = str(exc)
                self.show_window = False

    def red_visible_after_motion(self, *, max_frames: int = FINAL_APPROACH_RED_VISIBILITY_FRAMES) -> Optional[bool]:
        max_frames = max(1, int(max_frames))
        inspected_frames = 0
        last_frame = None
        self._last_loose_red_hint = None
        for _ in range(max_frames):
            ok, frame_bgr = self.capture.read()
            if not ok or frame_bgr is None:
                break
            if self.undistorter is not None:
                frame_bgr = self.undistorter.apply(frame_bgr)
            self._last_frame_bgr = frame_bgr
            if not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
                continue
            height, width = frame_bgr.shape[:2]
            inspected_frames += 1
            last_frame = frame_bgr
            hint = self._find_loose_red_hint(frame_bgr, (width, height))
            self._last_loose_red_hint = hint
            if hint is not None:
                self._record_red_visibility_check(
                    visible=True,
                    inspected_frames=inspected_frames,
                    max_frames=max_frames,
                    frame_bgr=frame_bgr,
                    hint=hint,
                )
                return True
        if inspected_frames == 0:
            self._record_red_visibility_check(
                visible=None,
                inspected_frames=0,
                max_frames=max_frames,
                frame_bgr=None,
                hint=None,
            )
            return None
        self._last_loose_red_hint = None
        self._record_red_visibility_check(
            visible=False,
            inspected_frames=inspected_frames,
            max_frames=max_frames,
            frame_bgr=last_frame,
            hint=None,
        )
        return False

    def match_final_grasp_view(
        self,
        reference: Mapping[str, Any],
        *,
        target_hint: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        match_config = reference.get("final_view_match", {})
        if not isinstance(match_config, Mapping) or not bool(match_config.get("enabled", False)):
            return None
        final_grasp_matcher = _load_local_module("final_grasp_matcher")
        return final_grasp_matcher.match_final_grasp_view(
            self._last_frame_bgr,
            self.config,
            match_config,
            base_dir=MODULE_DIR,
            target_hint=target_hint,
        )

    def close(self) -> None:
        if hasattr(self.capture, "release"):
            self.capture.release()
        if self._window_opened and self.cv2_module is not None and hasattr(self.cv2_module, "destroyWindow"):
            try:
                self.cv2_module.destroyWindow(self.window_name)
            except Exception:
                pass
            self._window_opened = False


def open_strip_camera_vision(
    *,
    device: str,
    config_path: Path = DEFAULT_CONFIG_PATH,
    calibration_path: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
    frames_per_detect: int = 5,
    show_window: bool = False,
) -> StripCameraVision:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for camera grasp") from exc

    strip_detection = _load_local_module("strip_detection")
    config = strip_detection.load_config(config_path)
    camera_config = dict(config["camera"])
    if width is not None:
        camera_config["width"] = int(width)
    if height is not None:
        camera_config["height"] = int(height)
    if fps is not None:
        camera_config["fps"] = int(fps)

    undistorter = None
    if calibration_path is not None and Path(calibration_path).exists():
        camera_calibration = _load_local_module("camera_calibration")
        calibration = camera_calibration.load_calibration(calibration_path)
        image_size = (int(camera_config["width"]), int(camera_config["height"]))
        if tuple(calibration["image_size"]) != image_size:
            raise ValueError(
                "camera resolution does not match calibration: "
                f"calibrated {calibration['image_size'][0]}x"
                f"{calibration['image_size'][1]}, "
                f"current {image_size[0]}x{image_size[1]}"
            )
        undistorter = camera_calibration.FrameUndistorter(calibration)
    capture_device: Any = int(device) if str(device).isdigit() else device
    capture = cv2.VideoCapture(capture_device)
    if camera_config.get("fourcc") is not None:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_config["fourcc"]))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_config["width"]))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_config["height"]))
    capture.set(cv2.CAP_PROP_FPS, int(camera_config["fps"]))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open camera device: {device}")
    return StripCameraVision(
        capture=capture,
        strip_detection=strip_detection,
        config=config,
        frames_per_detect=frames_per_detect,
        undistorter=undistorter,
        show_window=show_window,
        cv2_module=cv2,
    )


class ArmGraspStateMachine:
    def __init__(
        self,
        *,
        vision: Any,
        motion: Any,
        reference: Optional[Mapping[str, Any]] = None,
        dry_run: bool = False,
        single_step: bool = False,
        max_align_steps: int = 5,
        max_jog_deg: float = 1.0,
        grasp_window_center_tolerance_px: float = 80.0,
        grasp_window_size_ratio_tolerance: float = 0.2,
        spd: float = DEFAULT_SPEED,
        acc: float = DEFAULT_ACCELERATION,
        final_spd: Optional[float] = None,
        final_acc: Optional[float] = None,
        run_grasp_ready: bool = True,
        stop_after_final_pose: bool = False,
        abort_checker: Optional[Callable[[], bool]] = None,
    ):
        self.vision = vision
        self.motion = motion
        self.reference = dict(reference or default_grasp_reference())
        self.dry_run = bool(dry_run)
        self.single_step = bool(single_step)
        self.max_align_steps = int(max_align_steps)
        self.max_jog_deg = float(max_jog_deg)
        self.grasp_window_center_tolerance_px = float(grasp_window_center_tolerance_px)
        self.grasp_window_size_ratio_tolerance = float(grasp_window_size_ratio_tolerance)
        self.spd = float(spd)
        self.acc = float(acc)
        self.final_spd = self.spd if final_spd is None else float(final_spd)
        self.final_acc = self.acc if final_acc is None else float(final_acc)
        self.run_grasp_ready = bool(run_grasp_ready)
        self.stop_after_final_pose = bool(stop_after_final_pose)
        self.abort_checker = abort_checker
        configure_target_selection = getattr(
            self.vision,
            "configure_ready_target_selection",
            None,
        )
        square_face_reference = self._square_face_target_reference()
        if callable(configure_target_selection) and square_face_reference:
            configure_target_selection(square_face_reference)

    def _failure(
        self,
        stage: str,
        reason: str,
        *,
        feedback: str,
        target: Optional[Mapping[str, Any]] = None,
        plan: Optional[List[Dict[str, Any]]] = None,
        object_held: bool = False,
    ) -> GraspResult:
        return GraspResult(
            False,
            stage,
            reason,
            feedback=feedback,
            object_held=object_held,
            target=dict(target) if target else None,
            plan=plan or [],
        )

    def _abort_requested(self) -> bool:
        if self.abort_checker is not None and bool(self.abort_checker()):
            return True
        vision_abort_requested = getattr(self.vision, "abort_requested", None)
        if callable(vision_abort_requested):
            return bool(vision_abort_requested())
        return False

    def _abort_failure(
        self,
        plan: List[Dict[str, Any]],
        target: Optional[Mapping[str, Any]] = None,
    ) -> GraspResult:
        return self._failure(
            "ABORTED",
            "user aborted",
            feedback="user_aborted",
            target=target,
            plan=plan,
        )

    def _grasp_ready_plan(self) -> Dict[str, Any]:
        return {
            "stage": "MOVE_TO_GRASP_READY_POSE",
            "joints_deg": self._grasp_ready_pose_degrees(),
        }

    def _grasp_ready_pose_degrees(self) -> Dict[str, float]:
        start_sequence = self.reference.get("start_sequence", {})
        grasp_ready_pose = start_sequence.get(
            "grasp_ready_pose_deg",
            start_sequence.get("transport_pose_deg", DEFAULT_GRASP_READY_POSE_DEG),
        )
        return {joint: float(value) for joint, value in dict(grasp_ready_pose).items()}

    def _current_motion_pose_degrees(self) -> Optional[Dict[str, float]]:
        current_pose_degrees = getattr(self.motion, "current_pose_degrees", None)
        if not callable(current_pose_degrees):
            return None
        pose = current_pose_degrees()
        if not pose:
            return None
        joints = {}
        for joint in ("b", "s", "e", "w"):
            if joint in pose and pose[joint] is not None:
                joints[joint] = float(pose[joint])
        return joints or None

    def _pose_relative_to_current_alignment(
        self,
        reference_pose: Mapping[str, float],
        alignment_offset_scale: Any = 1.0,
    ) -> Tuple[Dict[str, float], str, Optional[Dict[str, float]], Dict[str, float]]:
        current_pose = self._current_motion_pose_degrees()
        base_pose = self._alignment_base_pose_degrees()
        adjusted_pose = {joint: float(value) for joint, value in dict(reference_pose).items()}
        if not current_pose:
            return adjusted_pose, "reference_absolute_pose", None, base_pose

        adjusted_any_joint = False
        for joint, reference_value in reference_pose.items():
            if joint not in current_pose or joint not in base_pose:
                continue
            if isinstance(alignment_offset_scale, Mapping):
                scale = float(alignment_offset_scale.get(joint, 1.0))
            else:
                scale = float(alignment_offset_scale)
            if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
                raise ValueError(
                    f"alignment_offset_scale for {joint} must be between 0 and 1"
                )
            adjusted_pose[joint] = float(reference_value) + scale * (
                float(current_pose[joint]) - float(base_pose[joint])
            )
            adjusted_any_joint = True
        if not adjusted_any_joint:
            return adjusted_pose, "reference_absolute_pose", current_pose, base_pose
        return adjusted_pose, "current_aligned_pose", current_pose, base_pose

    def _alignment_base_pose_degrees(self) -> Dict[str, float]:
        approach_sequence = self.reference.get("approach_sequence", {})
        if isinstance(approach_sequence, Mapping):
            alignment_base_pose = approach_sequence.get("alignment_base_pose_deg")
            if alignment_base_pose:
                return {
                    joint: float(value)
                    for joint, value in dict(alignment_base_pose).items()
                }
        return self._grasp_ready_pose_degrees()

    def _grasp_ready_open_plan(self) -> Optional[Dict[str, Any]]:
        start_sequence = self.reference.get("start_sequence", {})
        open_gripper_h = start_sequence.get("open_gripper_h", DEFAULT_GRASP_READY_GRIPPER_H)
        if open_gripper_h is None:
            return None
        return {
            "stage": "OPEN_GRIPPER",
            "open_gripper_h": float(open_gripper_h),
        }

    def _approach_open_plan(self) -> Optional[Dict[str, Any]]:
        start_sequence = self.reference.get("start_sequence", {})
        open_gripper_h = start_sequence.get("open_gripper_h", DEFAULT_GRASP_READY_GRIPPER_H)
        if open_gripper_h is None:
            return None
        return {
            "stage": "OPEN_GRIPPER_FOR_APPROACH",
            "open_gripper_h": float(open_gripper_h),
            "reason": "open gripper before moving to final grasp pose",
        }

    def _grasp_ready_pose_tolerance_degrees(self) -> float:
        start_sequence = self.reference.get("start_sequence", {})
        return float(start_sequence.get("pose_tolerance_deg", DEFAULT_GRASP_READY_TOLERANCE_DEG))

    def _grasp_ready_command_and_expected_pose(
        self,
        expected_pose: Mapping[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float], bool]:
        expected_targets = {joint: float(value) for joint, value in dict(expected_pose).items()}
        command_targets = dict(expected_targets)
        current_pose = self._current_motion_pose_degrees()
        if current_pose is None or "e" not in expected_targets or "e" not in current_pose:
            return command_targets, expected_targets, False

        e_delta = expected_targets["e"] - float(current_pose["e"])
        if e_delta >= -self._grasp_ready_pose_tolerance_degrees():
            return command_targets, expected_targets, False

        command_targets["e"] = expected_targets["e"] - GRASP_READY_NEGATIVE_E_COMPENSATION_DEG
        return command_targets, expected_targets, True

    def _square_face_target_reference(self) -> Mapping[str, Any]:
        square_face_target = self.reference.get("square_face_target", {})
        if square_face_target:
            return square_face_target
        target_reference = self.reference.get("target", {})
        if target_reference.get("allow_square_face_grasp") or target_reference.get("allow_square_face_tracking"):
            return target_reference
        return {}

    def _target_has_final_grasp_flags(self, target: Mapping[str, Any]) -> bool:
        reference_target = self.reference.get("target", {})
        if target.get("color") != "red" or not bool(target.get("stable")):
            return False
        if bool(reference_target.get("require_grasp_candidate", True)) and not bool(target.get("grasp_candidate")):
            return False
        if bool(reference_target.get("require_angle_reliable", True)) and not bool(target.get("angle_reliable")):
            return False
        return True

    def _target_ready_for_square_face_tracking(self, target: Mapping[str, Any]) -> bool:
        if not self._square_face_target_reference():
            return False
        return target.get("color") == "red" and bool(target.get("stable"))

    def _target_ready_for_tracking(self, target: Mapping[str, Any]) -> bool:
        if self._target_has_final_grasp_flags(target):
            return True
        return self._target_ready_for_square_face_tracking(target)

    def _center_error_px(self, target: Mapping[str, Any]) -> Tuple[float, float]:
        reference_target = self.reference.get("target", {})
        target_center = target.get("center_px", [0.0, 0.0])
        reference_center = reference_target.get("center_px", [0.0, 0.0])
        return (
            float(target_center[0]) - float(reference_center[0]),
            float(target_center[1]) - float(reference_center[1]),
        )

    def _size_ratio_to_reference(self, target: Mapping[str, Any]) -> float:
        reference_target = self.reference.get("target", {})
        target_size = target.get("size_px", [0.0, 0.0])
        reference_size = reference_target.get("size_px", [0.0, 0.0])
        target_long = max(float(target_size[0]), float(target_size[1]))
        reference_long = max(float(reference_size[0]), float(reference_size[1]))
        if reference_long <= 0.0:
            return 1.0
        return target_long / reference_long

    def _angle_error_deg(self, target: Mapping[str, Any]) -> float:
        reference_target = self.reference.get("target", {})
        target_angle = float(target.get("angle_deg", 0.0))
        reference_angle = float(reference_target.get("angle_deg", target_angle))
        return target_angle - reference_angle

    def _target_ready_for_closure(self, target: Mapping[str, Any]) -> bool:
        if not self._target_has_final_grasp_flags(target):
            return False
        error_u, error_v = self._center_error_px(target)
        if abs(error_u) > FINE_ALIGNMENT_CENTER_TOLERANCE_PX:
            return False
        if abs(error_v) > FINE_ALIGNMENT_CENTER_TOLERANCE_PX:
            return False
        ratio = self._size_ratio_to_reference(target)
        if ratio < 1.0 - self.grasp_window_size_ratio_tolerance:
            return False
        if ratio > 1.0 + self.grasp_window_size_ratio_tolerance:
            return False
        reference_target = self.reference.get("target", {})
        if bool(reference_target.get("require_angle_reliable", True)):
            if abs(self._angle_error_deg(target)) > FINE_ALIGNMENT_ANGLE_TOLERANCE_DEG:
                return False
        return True

    def _execute_gripper_open(self, plan: List[Dict[str, Any]]) -> Optional[GraspResult]:
        open_step = self._grasp_ready_open_plan()
        if open_step is None:
            return None
        plan.append(open_step)
        if self.dry_run:
            return None
        try:
            self.motion.open_gripper(
                angle=open_step["open_gripper_h"],
                spd=self.spd,
                acc=self.acc,
            )
        except Exception as exc:
            return self._failure(
                "ARM_CONTROL",
                f"open gripper failed: {exc}",
                feedback="arm_control_failed",
                plan=plan,
            )
        return None

    def _execute_grasp_ready_pose(self, plan: List[Dict[str, Any]]) -> Optional[GraspResult]:
        open_failure = self._execute_gripper_open(plan)
        if open_failure:
            return open_failure
        if self._abort_requested():
            return self._abort_failure(plan)

        grasp_ready_step = self._grasp_ready_plan()
        plan.append(grasp_ready_step)
        if not self.dry_run and grasp_ready_step["joints_deg"]:
            try:
                command_targets, expected_targets, compensated = (
                    self._grasp_ready_command_and_expected_pose(
                        grasp_ready_step["joints_deg"]
                    )
                )
                if compensated and hasattr(self.motion, "move_joints_with_expected_targets"):
                    grasp_ready_step["action"] = "move_joints_with_expected_targets"
                    grasp_ready_step["command_joints_deg"] = dict(command_targets)
                    grasp_ready_step["expected_joints_deg"] = dict(expected_targets)
                    grasp_ready_step["e_negative_compensation_deg"] = (
                        GRASP_READY_NEGATIVE_E_COMPENSATION_DEG
                    )
                    self.motion.move_joints_with_expected_targets(
                        command_targets,
                        expected_targets,
                        spd=self.spd,
                        acc=self.acc,
                        tolerance_degrees=self._grasp_ready_pose_tolerance_degrees(),
                    )
                else:
                    grasp_ready_step["action"] = "move_joints"
                    self.motion.move_joints(
                        grasp_ready_step["joints_deg"],
                        spd=self.spd,
                        acc=self.acc,
                        tolerance_degrees=self._grasp_ready_pose_tolerance_degrees(),
                    )
            except Exception as exc:
                return self._failure(
                    "ARM_CONTROL",
                    f"grasp ready pose failed: {exc}",
                    feedback="arm_control_failed",
                    plan=plan,
                )
        return None

    def _detection_validation_error(self, target: Any) -> Optional[str]:
        if not isinstance(target, Mapping):
            return "target detection must be an object"
        missing = [field for field in REQUIRED_DETECTION_FIELDS if field not in target]
        if missing:
            return "target detection missing required fields: " + ", ".join(missing)
        return None

    def _detect_ready_target(
        self,
        plan: List[Dict[str, Any]],
        *,
        allow_unstable_retry: bool = False,
        retry_attempts: Optional[int] = None,
        detection_stage: str = "DETECT_RED_STRIP",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[GraspResult]]:
        if retry_attempts is None:
            retry_attempts = POST_MOTION_DETECT_RETRY_ATTEMPTS
        retry_attempts = max(0, int(retry_attempts))
        attempts = 1 + (retry_attempts if allow_unstable_retry else 0)
        last_target: Optional[Dict[str, Any]] = None
        for attempt_index in range(attempts):
            if self._abort_requested():
                return None, self._abort_failure(plan, last_target)
            target = self.vision.detect()
            is_last_attempt = attempt_index >= attempts - 1
            if not target:
                if self._abort_requested():
                    return None, self._abort_failure(plan, last_target)
                if allow_unstable_retry and not is_last_attempt:
                    continue
                return None, self._failure(
                    detection_stage,
                    "red strip target lost",
                    feedback="target_lost",
                    plan=plan,
                )
            validation_error = self._detection_validation_error(target)
            if validation_error:
                invalid_target = dict(target) if isinstance(target, Mapping) else None
                return None, self._failure(
                    detection_stage,
                    validation_error,
                    feedback="target_lost",
                    target=invalid_target,
                    plan=plan,
                )
            last_target = dict(target)
            detect_step = {"stage": detection_stage, "target": dict(target)}
            if allow_unstable_retry and not self._target_ready_for_tracking(target) and not is_last_attempt:
                detect_step["post_motion_retry"] = True
            plan.append(detect_step)
            if self._abort_requested():
                return None, self._abort_failure(plan, target)
            if self._target_ready_for_tracking(target):
                return dict(target), None
            if allow_unstable_retry and not is_last_attempt:
                continue
            return None, self._failure(
                detection_stage,
                "target is not stable or not graspable",
                feedback="target_lost",
                target=target,
                plan=plan,
            )
        return None, self._failure(
            detection_stage,
            "target is not stable or not graspable",
            feedback="target_lost",
            target=last_target,
            plan=plan,
        )

    def _detect_ready_target_after_motion(
        self,
        plan: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[GraspResult]]:
        target, failure = self._detect_ready_target(
            plan,
            allow_unstable_retry=True,
        )
        if target is not None or failure is None or failure.feedback != "target_lost":
            return target, failure
        for recovery_round in range(1, POST_MOTION_TARGET_REACQUIRE_ROUNDS + 1):
            if self._abort_requested():
                return None, self._abort_failure(plan, failure.target)
            plan.append(
                {
                    "stage": "POST_MOTION_TARGET_REACQUIRE",
                    "action": "hold_current_arm_pose",
                    "round": recovery_round,
                    "max_rounds": POST_MOTION_TARGET_REACQUIRE_ROUNDS,
                    "reason": "wait for fresh stable arm-camera frames",
                }
            )
            if POST_MOTION_TARGET_REACQUIRE_SETTLE_SECONDS > 0.0:
                time.sleep(POST_MOTION_TARGET_REACQUIRE_SETTLE_SECONDS)
            self._flush_vision_after_motion()
            target, failure = self._detect_ready_target(
                plan,
                allow_unstable_retry=True,
                retry_attempts=POST_MOTION_TARGET_REACQUIRE_ATTEMPTS,
            )
            if target is not None or failure is None or failure.feedback != "target_lost":
                return target, failure
        return None, failure

    def _initial_red_search_config(self) -> Mapping[str, Any]:
        config = self.reference.get("initial_red_search", {})
        return config if isinstance(config, Mapping) else {}

    def _initial_red_search_value(self, key: str, default: Any) -> Any:
        return self._initial_red_search_config().get(key, default)

    def _initial_red_search_attempts(self, key: str, default: int) -> int:
        value = self._initial_red_search_value(key, default)
        try:
            attempts = int(value)
        except (TypeError, ValueError):
            attempts = default
        return max(POST_MOTION_DETECT_RETRY_ATTEMPTS, attempts)

    def _initial_red_search_b_bounds(self) -> Optional[Tuple[float, float]]:
        if not bool(
            self._initial_red_search_value(
                "use_b_safety_bounds",
                DEFAULT_INITIAL_RED_SEARCH_USE_B_SAFETY_BOUNDS,
            )
        ):
            return None
        try:
            min_b = float(
                self._initial_red_search_value(
                    "b_min_deg",
                    DEFAULT_INITIAL_RED_SEARCH_B_MIN_DEG,
                )
            )
            max_b = float(
                self._initial_red_search_value(
                    "b_max_deg",
                    DEFAULT_INITIAL_RED_SEARCH_B_MAX_DEG,
                )
            )
        except (TypeError, ValueError):
            return None
        if not math.isfinite(min_b) or not math.isfinite(max_b) or min_b >= max_b:
            return None
        return min_b, max_b

    def _initial_red_search_delta_abs_degrees(self) -> float:
        try:
            delta_abs = abs(
                float(
                    self._initial_red_search_value(
                        "delta_deg",
                        DEFAULT_INITIAL_RED_SEARCH_DELTA_DEG,
                    )
                )
            )
        except (TypeError, ValueError):
            delta_abs = DEFAULT_INITIAL_RED_SEARCH_DELTA_DEG
        return delta_abs if math.isfinite(delta_abs) and delta_abs > 0.0 else 0.0

    def _initial_red_search_iteration_limit(self, max_steps: int) -> int:
        bounds = self._initial_red_search_b_bounds()
        delta_abs = self._initial_red_search_delta_abs_degrees()
        if bounds is None or delta_abs <= 0.0:
            iteration_limit = max_steps
        else:
            min_b, max_b = bounds
            boundary_steps = int(math.ceil((max_b - min_b) / delta_abs)) + 2
            iteration_limit = max(max_steps, boundary_steps)
        if not bool(
            self._initial_red_search_value(
                "lower_middle_e_recovery_enabled",
                DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_RECOVERY_ENABLED,
            )
        ):
            return iteration_limit
        try:
            e_delta_abs = abs(
                float(
                    self._initial_red_search_value(
                        "lower_middle_e_delta_deg",
                        DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_DELTA_DEG,
                    )
                )
            )
            e_max_total = abs(
                float(
                    self._initial_red_search_value(
                        "lower_middle_e_max_total_deg",
                        DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_MAX_TOTAL_DEG,
                    )
                )
            )
        except (TypeError, ValueError):
            return iteration_limit
        if (
            not math.isfinite(e_delta_abs)
            or not math.isfinite(e_max_total)
            or e_delta_abs <= 0.0
            or e_max_total <= 0.0
        ):
            return iteration_limit
        e_recovery_steps = int(math.ceil(e_max_total / e_delta_abs))
        return iteration_limit + e_recovery_steps + 1

    def _clip_initial_red_search_step_to_b_bounds(
        self,
        search_step: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        step = dict(search_step)
        if step.get("joint") != "b":
            return step
        bounds = self._initial_red_search_b_bounds()
        if bounds is None:
            return step
        current_b_deg = self._current_joint_degrees("b")
        if current_b_deg is None:
            return step
        min_b, max_b = bounds
        delta_deg = float(step.get("delta_deg", 0.0))
        if delta_deg < 0.0:
            remaining = current_b_deg - min_b
            if remaining <= 1e-9:
                return None
            safe_delta = max(delta_deg, -remaining)
        elif delta_deg > 0.0:
            remaining = max_b - current_b_deg
            if remaining <= 1e-9:
                return None
            safe_delta = min(delta_deg, remaining)
        else:
            return None
        if abs(safe_delta) <= 1e-9:
            return None
        if abs(safe_delta - delta_deg) > 1e-9:
            step["requested_delta_deg"] = delta_deg
        step["delta_deg"] = safe_delta
        step["b_safety_bounds_deg"] = {"min": min_b, "max": max_b}
        return step

    def _loose_red_hint(self) -> Optional[Dict[str, Any]]:
        hint_getter = getattr(self.vision, "loose_red_hint", None)
        if not callable(hint_getter):
            return None
        hint = hint_getter()
        return dict(hint) if isinstance(hint, Mapping) else None

    def _loose_red_search_step(
        self,
        hint: Mapping[str, Any],
        *,
        allow_lower_middle_e: bool = False,
    ) -> Optional[Dict[str, Any]]:
        center = hint.get("center_px")
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            return None
        center_u = float(center[0])
        center_v = float(center[1])
        image_size = hint.get("image_size")
        if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
            default_center_u = float(image_size[0]) / 2.0
            image_height = float(image_size[1])
        else:
            default_center_u = float(self._square_face_target_reference().get("center_px", [640.0])[0])
            image_height = 0.0
        search_center = self._initial_red_search_value("center_px", None)
        if isinstance(search_center, (list, tuple)) and search_center:
            reference_u = float(search_center[0])
        else:
            reference = self._square_face_target_reference()
            reference_center = reference.get("center_px") if isinstance(reference, Mapping) else None
            if isinstance(reference_center, (list, tuple)) and reference_center:
                reference_u = float(reference_center[0])
            else:
                reference_u = default_center_u
        error_u = center_u - reference_u
        tolerance_px = abs(
            float(
                self._initial_red_search_value(
                    "center_tolerance_px",
                    DEFAULT_INITIAL_RED_SEARCH_CENTER_TOLERANCE_PX,
                )
            )
        )
        if abs(error_u) <= tolerance_px:
            if not allow_lower_middle_e or not bool(
                self._initial_red_search_value(
                    "lower_middle_e_recovery_enabled",
                    DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_RECOVERY_ENABLED,
                )
            ):
                return None
            try:
                min_y_ratio = float(
                    self._initial_red_search_value(
                        "lower_middle_min_y_ratio",
                        DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_MIN_Y_RATIO,
                    )
                )
                e_delta_deg = abs(
                    float(
                        self._initial_red_search_value(
                            "lower_middle_e_delta_deg",
                            DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_DELTA_DEG,
                        )
                    )
                )
            except (TypeError, ValueError):
                return None
            if (
                image_height <= 0.0
                or not math.isfinite(min_y_ratio)
                or not 0.0 <= min_y_ratio <= 1.0
                or not math.isfinite(e_delta_deg)
                or e_delta_deg <= 0.0
                or center_v < image_height * min_y_ratio
            ):
                return None
            return {
                "stage": "VISUAL_ALIGN",
                "action": "jog",
                "joint": str(
                    self._initial_red_search_value(
                        "lower_middle_e_joint",
                        DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_JOINT,
                    )
                ),
                "delta_deg": e_delta_deg,
                "reason": "loose red region is centered in the lower image",
                "feedback": "loose_red_lower_middle",
                "target_hint": dict(hint),
            }
        delta_abs = abs(
            float(
                self._initial_red_search_value(
                    "delta_deg",
                    DEFAULT_INITIAL_RED_SEARCH_DELTA_DEG,
                )
            )
        )
        delta_deg = -delta_abs if error_u > 0.0 else delta_abs
        side = "right" if error_u > 0.0 else "left"
        return {
            "stage": "VISUAL_ALIGN",
            "action": "jog",
            "joint": str(
                self._initial_red_search_value(
                    "joint",
                    DEFAULT_INITIAL_RED_SEARCH_JOINT,
                )
            ),
            "delta_deg": max(-delta_abs, min(delta_abs, delta_deg)),
            "reason": f"loose red region is {side} of the search center",
            "feedback": f"loose_red_{side}",
            "target_hint": dict(hint),
        }

    def _try_initial_loose_red_search(
        self,
        plan: List[Dict[str, Any]],
        initial_failure: GraspResult,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[GraspResult]]:
        if not bool(self._initial_red_search_value("enabled", True)):
            return None, initial_failure
        max_steps = max(
            0,
            int(
                self._initial_red_search_value(
                    "max_steps",
                    DEFAULT_INITIAL_RED_SEARCH_MAX_STEPS,
                )
            ),
        )
        last_failure = initial_failure
        lower_middle_e_total_deg = 0.0
        settle_attempts = self._initial_red_search_attempts(
            "settle_attempts",
            DEFAULT_INITIAL_RED_SEARCH_SETTLE_ATTEMPTS,
        )
        centered_settle_attempts = self._initial_red_search_attempts(
            "centered_settle_attempts",
            DEFAULT_INITIAL_RED_SEARCH_CENTERED_SETTLE_ATTEMPTS,
        )
        iteration_limit = self._initial_red_search_iteration_limit(max_steps)
        for _ in range(iteration_limit):
            if self._abort_requested():
                return None, self._abort_failure(plan)
            hint = self._loose_red_hint()
            if hint is None:
                return None, last_failure
            search_step = self._loose_red_search_step(
                hint,
                allow_lower_middle_e=last_failure.target is None,
            )
            if search_step is None:
                target, failure = self._detect_ready_target(
                    plan,
                    allow_unstable_retry=True,
                    retry_attempts=centered_settle_attempts,
                )
                if target is not None:
                    return target, None
                return None, failure if failure is not None else last_failure
            if search_step.get("feedback") == "loose_red_lower_middle":
                try:
                    max_total_deg = abs(
                        float(
                            self._initial_red_search_value(
                                "lower_middle_e_max_total_deg",
                                DEFAULT_INITIAL_RED_SEARCH_LOWER_MIDDLE_E_MAX_TOTAL_DEG,
                            )
                        )
                    )
                except (TypeError, ValueError):
                    max_total_deg = 0.0
                remaining_deg = max_total_deg - lower_middle_e_total_deg
                if (
                    not math.isfinite(remaining_deg)
                    or remaining_deg <= 1e-9
                ):
                    return None, self._failure(
                        "DETECT_RED_STRIP",
                        "red strip target lost at lower-middle elbow recovery limit",
                        feedback="target_lost",
                        target=hint,
                        plan=plan,
                    )
                requested_delta_deg = float(search_step.get("delta_deg", 0.0))
                safe_delta_deg = min(requested_delta_deg, remaining_deg)
                if safe_delta_deg <= 1e-9:
                    return None, self._failure(
                        "DETECT_RED_STRIP",
                        "red strip target lost at lower-middle elbow recovery limit",
                        feedback="target_lost",
                        target=hint,
                        plan=plan,
                    )
                if safe_delta_deg < requested_delta_deg:
                    search_step["requested_delta_deg"] = requested_delta_deg
                search_step["delta_deg"] = safe_delta_deg
                search_step["e_recovery_total_deg"] = (
                    lower_middle_e_total_deg + safe_delta_deg
                )
            search_step = self._clip_initial_red_search_step_to_b_bounds(search_step)
            if search_step is None:
                return None, self._failure(
                    "DETECT_RED_STRIP",
                    "red strip target lost at base safety boundary",
                    feedback="target_lost",
                    target=hint,
                    plan=plan,
                )
            plan.append(search_step)
            if self.dry_run:
                return None, GraspResult(
                    True,
                    "VISUAL_ALIGN",
                    feedback=str(search_step.get("feedback", "target_lost")),
                    object_held=False,
                    target=dict(hint),
                    plan=plan,
                )
            try:
                self._execute_alignment_motion(search_step)
                if search_step.get("feedback") == "loose_red_lower_middle":
                    lower_middle_e_total_deg += float(search_step["delta_deg"])
                self._flush_vision_after_motion()
            except Exception as exc:
                return None, self._failure(
                    "ARM_CONTROL",
                    str(exc),
                    feedback="arm_control_failed",
                    target=hint,
                    plan=plan,
                )
            target, failure = self._detect_ready_target(
                plan,
                allow_unstable_retry=True,
                retry_attempts=settle_attempts,
            )
            if target is not None:
                return target, None
            if failure is not None:
                last_failure = failure
                if failure.feedback != "target_lost":
                    return None, failure
        return None, self._failure(
            "DETECT_RED_STRIP",
            "red strip target lost after loose red base search",
            feedback="target_lost",
            plan=plan,
        )

    def _match_final_grasp_view_image(
        self,
        target: Optional[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        matcher = getattr(self.vision, "match_final_grasp_view", None)
        if not callable(matcher):
            return None
        try:
            result = matcher(self.reference, target_hint=target)
        except TypeError:
            result = matcher(self.reference)
        if result is None:
            return None
        if not isinstance(result, Mapping):
            return {
                "ok": False,
                "feedback": "arm_control_failed",
                "reason": "final image matcher returned a non-object result",
                "target": dict(target) if target else None,
                "metrics": {},
            }
        return dict(result)

    def _record_final_view_image_match(
        self,
        plan: List[Dict[str, Any]],
        match: Mapping[str, Any],
        target: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        matched_target = match.get("target")
        if isinstance(matched_target, Mapping):
            target_payload = dict(matched_target)
        elif target is not None:
            target_payload = dict(target)
        else:
            target_payload = {}
        image_match = {
            "ok": bool(match.get("ok", False)),
            "feedback": str(match.get("feedback", "")),
            "reason": str(match.get("reason", "")),
            "metrics": dict(match.get("metrics", {}))
            if isinstance(match.get("metrics", {}), Mapping)
            else {},
        }
        if plan and plan[-1].get("stage") == "FINAL_VIEW_RECHECK":
            plan[-1]["target"] = target_payload
            plan[-1]["image_match"] = image_match
        else:
            plan.append(
                {
                    "stage": "FINAL_VIEW_RECHECK",
                    "target": target_payload,
                    "image_match": image_match,
                }
            )
        return target_payload
    def _final_view_recheck(
        self,
        plan: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[GraspResult]]:
        target, failure = self._detect_ready_target(
            plan,
            detection_stage="FINAL_VIEW_RECHECK",
        )
        image_match = self._match_final_grasp_view_image(target)
        if image_match is not None:
            matched_target = self._record_final_view_image_match(plan, image_match, target)
            if bool(image_match.get("ok", False)):
                if not matched_target:
                    return None, self._failure(
                        "FINAL_VIEW_RECHECK",
                        "final image matcher accepted the view but did not return target features",
                        feedback="target_lost",
                        target=target,
                        plan=plan,
                    )
                return matched_target, None
            feedback = str(image_match.get("feedback") or "arm_control_failed")
            reason = str(image_match.get("reason") or "final image does not match grasp sample")
            return None, self._failure(
                "FINAL_VIEW_RECHECK",
                reason,
                feedback=feedback,
                target=matched_target or target,
                plan=plan,
            )
        if failure:
            return None, failure
        if self._target_ready_for_closure(target):
            return target, None

        feedback = self._final_view_failure_feedback(target)
        diagnostic_feedback = (
            self._classify_grasp_window(target)
            if self._target_has_final_grasp_flags(target)
            else feedback
        )
        feedback_reason = (
            feedback
            if diagnostic_feedback == feedback
            else f"{feedback}/{diagnostic_feedback}"
        )
        error_u, error_v = self._center_error_px(target)
        reason = (
            "final view does not satisfy closure gate: "
            f"{feedback_reason}; center_error_px=({error_u:.1f},{error_v:.1f}), "
            f"size_ratio={self._size_ratio_to_reference(target):.3f}, "
            f"angle_error_deg={self._angle_error_deg(target):.1f}"
        )
        return None, self._failure(
            "FINAL_VIEW_RECHECK",
            reason,
            feedback=feedback,
            target=target,
            plan=plan,
        )

    def _final_view_correction_config(self) -> Dict[str, Any]:
        config = self.reference.get("final_view_correction", {})
        if not isinstance(config, Mapping):
            config = {}
        return dict(config)

    def _final_s_forward_match_config(self) -> Dict[str, Any]:
        config = self.reference.get("final_s_forward_match", {})
        if not isinstance(config, Mapping):
            config = {}
        return dict(config)

    def _final_s_forward_match_enabled(self) -> bool:
        return bool(self._final_s_forward_match_config().get("enabled", False))

    def _current_joint_degrees(self, joint: str) -> Optional[float]:
        current_pose_degrees = getattr(self.motion, "current_pose_degrees", None)
        if not callable(current_pose_degrees):
            return None
        pose = current_pose_degrees()
        if not isinstance(pose, Mapping) or joint not in pose:
            return None
        value = float(pose[joint])
        return value if math.isfinite(value) else None

    def _final_s_forward_match_feedback_allowed(
        self,
        failure: Optional[GraspResult],
    ) -> bool:
        if failure is None or failure.stage != "FINAL_VIEW_RECHECK":
            return False
        config = self._final_s_forward_match_config()
        allowed = config.get("allowed_feedback", DEFAULT_FINAL_S_FORWARD_ALLOWED_FEEDBACK)
        if not isinstance(allowed, (list, tuple, set)):
            allowed = DEFAULT_FINAL_S_FORWARD_ALLOWED_FEEDBACK
        return failure.feedback in {str(item) for item in allowed}

    def _final_s_forward_match_step(
        self,
        failure: Optional[GraspResult],
        *,
        attempt: int,
        joint: str,
        delta_deg: float,
        current_s_deg: float,
        max_s_deg: float,
        reason: str = "advance shoulder until final image matches manual grasp reference",
        feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if failure is not None and failure.plan:
            image_match = failure.plan[-1].get("image_match", {})
            if isinstance(image_match, Mapping) and isinstance(image_match.get("metrics"), Mapping):
                metrics = dict(image_match["metrics"])
        return {
            "stage": "FINAL_S_FORWARD_MATCH",
            "action": "jog",
            "joint": joint,
            "delta_deg": delta_deg,
            "reason": reason,
            "feedback": feedback if feedback is not None else failure.feedback,
            "attempt": attempt,
            "current_s_deg": current_s_deg,
            "max_s_deg": max_s_deg,
            "metrics": metrics,
        }

    def _final_s_forward_latest_image_match(
        self,
        failure: Optional[GraspResult],
    ) -> Dict[str, Any]:
        if failure is None or not failure.plan:
            return {}
        image_match = failure.plan[-1].get("image_match", {})
        return dict(image_match) if isinstance(image_match, Mapping) else {}

    def _positive_finite_float(self, value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric) or numeric <= 0.0:
            return None
        return numeric

    def _final_s_candidate_far_edge_px(
        self,
        failure: Optional[GraspResult],
    ) -> Optional[float]:
        if failure is None:
            return None
        if isinstance(failure.target, Mapping):
            value = self._positive_finite_float(failure.target.get("far_edge_px"))
            if value is not None:
                return value
        image_match = self._final_s_forward_latest_image_match(failure)
        target = image_match.get("target")
        if isinstance(target, Mapping):
            value = self._positive_finite_float(target.get("far_edge_px"))
            if value is not None:
                return value
        metrics = image_match.get("metrics")
        if isinstance(metrics, Mapping):
            return self._positive_finite_float(metrics.get("candidate_far_edge_px"))
        return None

    def _final_s_reference_far_edge_px(self) -> Optional[float]:
        config = self._final_s_forward_match_config()
        value = self._positive_finite_float(config.get("reference_far_edge_px"))
        if value is not None:
            return value

        final_match_config = self.reference.get("final_view_match", {})
        if not isinstance(final_match_config, Mapping):
            return None
        value = self._positive_finite_float(final_match_config.get("reference_far_edge_px"))
        if value is not None:
            return value

        feature_file = final_match_config.get("reference_features_file")
        if not feature_file:
            return None
        feature_path = Path(str(feature_file))
        if not feature_path.is_absolute():
            feature_path = MODULE_DIR / feature_path
        try:
            payload = json.loads(feature_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        features = None
        if isinstance(payload, Mapping):
            features = payload.get("features")
            if not isinstance(features, list):
                features = payload.get("references")
        if isinstance(features, list):
            for feature in features:
                if isinstance(feature, Mapping):
                    value = self._positive_finite_float(feature.get("far_edge_px"))
                    if value is not None:
                        return value
        return None

    def _final_s_far_edge_gate(
        self,
        failure: Optional[GraspResult],
    ) -> Optional[Dict[str, float]]:
        candidate = self._final_s_candidate_far_edge_px(failure)
        reference = self._final_s_reference_far_edge_px()
        if candidate is None or reference is None:
            return None
        return {
            "candidate_far_edge_px": candidate,
            "reference_far_edge_px": reference,
        }

    def _final_s_forward_limit_accept_step(
        self,
        failure: GraspResult,
        *,
        current_s_deg: float,
        max_s_deg: float,
        far_edge_gate: Mapping[str, float],
    ) -> Dict[str, Any]:
        return {
            "stage": "FINAL_S_FORWARD_LIMIT_ACCEPT",
            "reason": "candidate far edge reached manual reference at shoulder safety limit",
            "feedback": failure.feedback,
            "current_s_deg": current_s_deg,
            "max_s_deg": max_s_deg,
            "candidate_far_edge_px": far_edge_gate["candidate_far_edge_px"],
            "reference_far_edge_px": far_edge_gate["reference_far_edge_px"],
        }

    def _final_s_forward_accept_no_red_at_limit_enabled(self) -> bool:
        config = self._final_s_forward_match_config()
        return bool(config.get("accept_no_red_at_limit", False))

    def _final_s_forward_failure_has_red_target(
        self,
        failure: Optional[GraspResult],
    ) -> bool:
        if failure is None:
            return False
        if isinstance(failure.target, Mapping) and bool(failure.target):
            return True
        image_match = self._final_s_forward_latest_image_match(failure)
        target = image_match.get("target")
        if isinstance(target, Mapping) and bool(target):
            return True
        metrics = image_match.get("metrics")
        if isinstance(metrics, Mapping):
            area = self._positive_finite_float(metrics.get("candidate_area_px"))
            return area is not None
        return False

    def _final_s_forward_can_accept_no_red_at_limit(
        self,
        failure: Optional[GraspResult],
    ) -> bool:
        return (
            self._final_s_forward_accept_no_red_at_limit_enabled()
            and failure is not None
            and failure.stage == "FINAL_VIEW_RECHECK"
            and failure.feedback == "target_lost"
            and not self._final_s_forward_failure_has_red_target(failure)
        )

    def _final_s_forward_limit_no_red_accept_step(
        self,
        failure: GraspResult,
        *,
        current_s_deg: float,
        max_s_deg: float,
    ) -> Dict[str, Any]:
        return {
            "stage": "FINAL_S_FORWARD_LIMIT_NO_RED_ACCEPT",
            "reason": "no red final-view target visible at shoulder safety limit",
            "feedback": failure.feedback,
            "current_s_deg": current_s_deg,
            "max_s_deg": max_s_deg,
        }

    def _final_s_forward_post_recovery_limit_accept_step(
        self,
        failure: GraspResult,
        *,
        current_s_deg: float,
        max_s_deg: float,
    ) -> Dict[str, Any]:
        image_match = self._final_s_forward_latest_image_match(failure)
        metrics = image_match.get("metrics")
        return {
            "stage": "FINAL_S_FORWARD_LIMIT_RECOVERY_ACCEPT",
            "reason": "post-recovery shoulder limit reached; proceed to close gripper",
            "feedback": failure.feedback,
            "current_s_deg": current_s_deg,
            "max_s_deg": max_s_deg,
            "metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
        }

    def _final_s_forward_limit_recovery_config(self) -> Dict[str, Any]:
        config = self._final_s_forward_match_config()
        recovery = config.get("limit_recovery", {})
        if not isinstance(recovery, Mapping):
            recovery = {}
        return dict(recovery)

    def _final_s_forward_limit_recovery_step(
        self,
        failure: GraspResult,
        *,
        current_s_deg: float,
        max_s_deg: float,
        current_e_deg: float,
        far_edge_gate: Mapping[str, float],
    ) -> Optional[Dict[str, Any]]:
        recovery = self._final_s_forward_limit_recovery_config()
        if not bool(recovery.get("enabled", False)):
            return None
        if not math.isfinite(current_s_deg) or not math.isfinite(current_e_deg):
            return None
        configured_delta_deg = float(
            recovery.get(
                "e_delta_deg",
                DEFAULT_FINAL_S_FORWARD_LIMIT_RECOVERY_E_DELTA_DEG,
            )
        )
        if not math.isfinite(configured_delta_deg) or abs(configured_delta_deg) <= 1e-9:
            return None
        configured_delta_deg = -abs(configured_delta_deg)
        target_e_deg = current_e_deg + configured_delta_deg
        actual_delta_deg = configured_delta_deg
        if abs(actual_delta_deg) <= 1e-9:
            return None
        post_recovery_max_s_deg = float(
            recovery.get(
                "post_raise_max_s_deg",
                DEFAULT_FINAL_S_FORWARD_LIMIT_RECOVERY_POST_MAX_S_DEG,
            )
        )
        if not math.isfinite(post_recovery_max_s_deg):
            return None
        target_s_deg = max(current_s_deg, post_recovery_max_s_deg)
        return {
            "stage": "FINAL_S_FORWARD_LIMIT_RAISE_ELBOW",
            "action": "move_joints",
            "joints_deg": {"e": target_e_deg, "s": target_s_deg},
            "delta_deg": actual_delta_deg,
            "configured_delta_deg": configured_delta_deg,
            "s_delta_deg": target_s_deg - current_s_deg,
            "post_recovery_max_s_deg": post_recovery_max_s_deg,
            "reason": "candidate far edge is shorter than manual reference at shoulder safety limit; move elbow and shoulder together to the recovery pose",
            "feedback": failure.feedback,
            "current_s_deg": current_s_deg,
            "max_s_deg": max_s_deg,
            "current_e_deg": current_e_deg,
            "candidate_far_edge_px": far_edge_gate["candidate_far_edge_px"],
            "reference_far_edge_px": far_edge_gate["reference_far_edge_px"],
        }

    @staticmethod
    def _is_nonfatal_final_s_recovery_error(exc: Exception) -> bool:
        message = str(exc)
        lowered = message.lower()
        return (
            ("e关节" in message and "未到达目标" in message)
            or ("elbow" in lowered and "not reached" in lowered)
        )

    def _is_too_much_red_final_view_failure(
        self,
        failure: Optional[GraspResult],
    ) -> bool:
        if failure is None or failure.stage != "FINAL_VIEW_RECHECK":
            return False
        if failure.feedback != "target_too_near":
            return False
        reason = failure.reason.lower()
        return (
            "more red" in reason
            or "contains more red" in reason
            or "visible red exceeds" in reason
            or "aspect ratio" in reason
        )

    def _is_too_small_red_final_view_failure(
        self,
        failure: Optional[GraspResult],
    ) -> bool:
        if failure is None or failure.stage != "FINAL_VIEW_RECHECK":
            return False
        if failure.feedback != "target_too_far":
            return False
        reason = failure.reason.lower()
        return (
            "long side" in reason
            or "too small" in reason
            or "smaller than grasp sample" in reason
        )

    def _is_center_above_final_view_failure(
        self,
        failure: Optional[GraspResult],
    ) -> bool:
        if failure is None or failure.stage != "FINAL_VIEW_RECHECK":
            return False
        reason = failure.reason.lower()
        return "center is above" in reason or "center above" in reason

    def _final_view_correction_step(
        self,
        failure: GraspResult,
        correction_index: int,
    ) -> Optional[Dict[str, Any]]:
        config = self._final_view_correction_config()
        if not bool(config.get("enabled", True)):
            return None
        if self._is_center_above_final_view_failure(failure):
            joint_key = "center_above_joint"
            delta_key = "center_above_delta_deg"
            default_joint = DEFAULT_FINAL_VIEW_TOO_MUCH_RED_JOINT
            default_delta = -abs(DEFAULT_FINAL_VIEW_TOO_MUCH_RED_DELTA_DEG)
            reason = "final view center is above grasp sample"
            feedback = "arm_control_failed"
        elif self._is_too_small_red_final_view_failure(failure):
            joint_key = "too_small_red_joint"
            delta_key = "too_small_red_delta_deg"
            default_joint = DEFAULT_FINAL_VIEW_TOO_SMALL_RED_JOINT
            default_delta = DEFAULT_FINAL_VIEW_TOO_SMALL_RED_DELTA_DEG
            reason = "final view red strip is too small"
            feedback = "target_too_far"
        else:
            joint_key = "too_much_red_joint"
            delta_key = "too_much_red_delta_deg"
            default_joint = DEFAULT_FINAL_VIEW_TOO_MUCH_RED_JOINT
            default_delta = DEFAULT_FINAL_VIEW_TOO_MUCH_RED_DELTA_DEG
            reason = "visible red exceeds grasp sample"
            feedback = "target_too_near"
        joint = str(
            config.get(joint_key, default_joint)
        )
        if joint not in {"b", "s", "e", "w"}:
            return None
        delta_deg = float(
            config.get(
                delta_key,
                default_delta,
            )
        )
        if not math.isfinite(delta_deg) or abs(delta_deg) <= 1e-9:
            return None

        metrics: Dict[str, Any] = {}
        if failure.plan:
            image_match = failure.plan[-1].get("image_match", {})
            if isinstance(image_match, Mapping) and isinstance(image_match.get("metrics"), Mapping):
                metrics = dict(image_match["metrics"])
        return {
            "stage": "FINAL_VIEW_CORRECTION",
            "action": "jog",
            "joint": joint,
            "delta_deg": delta_deg,
            "reason": reason,
            "feedback": feedback,
            "attempt": correction_index + 1,
            "metrics": metrics,
        }

    def _final_view_recheck_with_corrections(
        self,
        plan: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[GraspResult]]:
        target, failure = self._final_view_recheck(plan)
        if failure is None:
            return target, None

        config = self._final_view_correction_config()
        max_steps = int(config.get("max_steps", DEFAULT_FINAL_VIEW_CORRECTION_MAX_STEPS))
        for correction_index in range(max(0, max_steps)):
            if not (
                self._is_too_much_red_final_view_failure(failure)
                or self._is_too_small_red_final_view_failure(failure)
                or self._is_center_above_final_view_failure(failure)
            ):
                return target, failure
            correction_step = self._final_view_correction_step(
                failure,
                correction_index,
            )
            if correction_step is None:
                return target, failure
            plan.append(correction_step)
            try:
                self._execute_alignment_motion(correction_step)
            except Exception as exc:
                return None, self._failure(
                    "FINAL_VIEW_CORRECTION",
                    f"final view correction failed: {exc}",
                    feedback="arm_control_failed",
                    target=target,
                    plan=plan,
                )
            self._flush_vision_after_motion()
            if self._abort_requested():
                return None, self._abort_failure(plan, target)
            target, failure = self._final_view_recheck(plan)
            if failure is None:
                return target, None
        return target, failure

    def _final_view_recheck_with_s_forward_match(
        self,
        plan: List[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[GraspResult]]:
        if not self._final_s_forward_match_enabled():
            return self._final_view_recheck_with_corrections(plan)

        config = self._final_s_forward_match_config()
        joint = str(config.get("joint", DEFAULT_FINAL_S_FORWARD_MATCH_JOINT))
        if joint != DEFAULT_FINAL_S_FORWARD_MATCH_JOINT:
            return None, self._failure(
                "FINAL_S_FORWARD_MATCH",
                "final s-forward match only supports the s joint",
                feedback="arm_control_failed",
                plan=plan,
            )
        max_s_deg = float(config.get("max_s_deg", DEFAULT_FINAL_S_FORWARD_MATCH_MAX_S_DEG))
        min_progress_deg = abs(
            float(
                config.get(
                    "min_progress_deg",
                    DEFAULT_FINAL_S_FORWARD_MIN_PROGRESS_DEG,
                )
            )
        )
        if (
            not math.isfinite(max_s_deg)
            or not math.isfinite(min_progress_deg)
        ):
            return None, self._failure(
                "FINAL_S_FORWARD_MATCH",
                "invalid final s-forward match safety configuration",
                feedback="arm_control_failed",
                plan=plan,
            )

        recovery_config = self._final_s_forward_limit_recovery_config()
        recovery_used = 0
        max_recoveries = int(recovery_config.get("max_recoveries", 1))
        limit_tolerance_deg = max(min_progress_deg, 1e-9)
        attempt = 0
        target: Optional[Dict[str, Any]] = None

        current_s_deg = self._current_joint_degrees(joint)
        if current_s_deg is None:
            return None, self._failure(
                "FINAL_S_FORWARD_MATCH",
                "cannot verify current s joint angle before final s-forward match",
                feedback="arm_control_failed",
                plan=plan,
            )
        if current_s_deg < max_s_deg - limit_tolerance_deg:
            safe_delta_deg = max_s_deg - current_s_deg
            if safe_delta_deg <= 0.0:
                return None, self._failure(
                    "FINAL_S_FORWARD_MATCH",
                    "final s-forward match has no safe shoulder movement left",
                    feedback="arm_control_failed",
                    plan=plan,
                )
            attempt += 1
            search_step = self._final_s_forward_match_step(
                None,
                attempt=attempt,
                joint=joint,
                delta_deg=safe_delta_deg,
                current_s_deg=current_s_deg,
                max_s_deg=max_s_deg,
                reason="advance shoulder directly to first-stage limit before final view recheck",
                feedback="pre_final_view_recheck",
            )
            plan.append(search_step)
            try:
                self._execute_alignment_motion(search_step)
            except Exception as exc:
                return None, self._failure(
                    "FINAL_S_FORWARD_MATCH",
                    f"final s-forward match jog failed: {exc}",
                    feedback="arm_control_failed",
                    plan=plan,
                )
            self._flush_vision_after_motion()
            if self._abort_requested():
                return None, self._abort_failure(plan, target)

        current_s_deg = self._current_joint_degrees(joint)
        if current_s_deg is None:
            return None, self._failure(
                "FINAL_S_FORWARD_MATCH",
                "cannot verify s joint reached the first-stage limit",
                feedback="arm_control_failed",
                plan=plan,
            )
        if current_s_deg < max_s_deg - limit_tolerance_deg:
            return None, self._failure(
                "FINAL_S_FORWARD_MATCH",
                "final s-forward match did not reach shoulder first-stage limit",
                feedback="arm_control_failed",
                plan=plan,
            )

        target, failure = self._final_view_recheck(plan)
        if failure is None:
            return target, None
        if not self._final_s_forward_match_feedback_allowed(failure):
            return target, failure

        if self._final_s_forward_match_feedback_allowed(failure):
            current_s_deg = self._current_joint_degrees(joint)
            if current_s_deg is None:
                return target, self._failure(
                    "FINAL_S_FORWARD_MATCH",
                    "cannot verify current s joint angle before final s-forward match",
                    feedback="arm_control_failed",
                    target=target,
                    plan=plan,
                )
            at_s_limit = current_s_deg >= max_s_deg - limit_tolerance_deg
            if at_s_limit:
                far_edge_gate = self._final_s_far_edge_gate(failure)
                if (
                    far_edge_gate is not None
                    and far_edge_gate["candidate_far_edge_px"]
                    >= far_edge_gate["reference_far_edge_px"]
                ):
                    plan.append(
                        self._final_s_forward_limit_accept_step(
                            failure,
                            current_s_deg=current_s_deg,
                            max_s_deg=max_s_deg,
                            far_edge_gate=far_edge_gate,
                        )
                    )
                    accepted_target = (
                        dict(failure.target)
                        if isinstance(failure.target, Mapping)
                        else dict(target) if isinstance(target, Mapping) else {}
                    )
                    return accepted_target, None
                if self._final_s_forward_can_accept_no_red_at_limit(failure):
                    plan.append(
                        self._final_s_forward_limit_no_red_accept_step(
                            failure,
                            current_s_deg=current_s_deg,
                            max_s_deg=max_s_deg,
                        )
                    )
                    return dict(target) if isinstance(target, Mapping) else {}, None
                if (
                    recovery_used > 0
                    and bool(recovery_config.get("accept_at_post_recovery_limit", True))
                ):
                    plan.append(
                        self._final_s_forward_post_recovery_limit_accept_step(
                            failure,
                            current_s_deg=current_s_deg,
                            max_s_deg=max_s_deg,
                        )
                    )
                    accepted_target = (
                        dict(failure.target)
                        if isinstance(failure.target, Mapping)
                        else dict(target) if isinstance(target, Mapping) else {}
                    )
                    return accepted_target, None
                if far_edge_gate is not None and recovery_used < max(0, max_recoveries):
                    recovery_current_e_deg = self._current_joint_degrees("e")
                    recovery_performed = False
                    if recovery_current_e_deg is not None:
                        recovery_step = self._final_s_forward_limit_recovery_step(
                            failure,
                            current_s_deg=current_s_deg,
                            max_s_deg=max_s_deg,
                            current_e_deg=recovery_current_e_deg,
                            far_edge_gate=far_edge_gate,
                        )
                        if recovery_step is None:
                            recovery_step = None
                    else:
                        recovery_step = None
                    if recovery_step is not None:
                        plan.append(recovery_step)
                        try:
                            self._execute_alignment_motion(recovery_step)
                        except Exception as exc:
                            if self._is_nonfatal_final_s_recovery_error(exc):
                                recovery_step["nonfatal_error"] = str(exc)
                            else:
                                return None, self._failure(
                                    "FINAL_S_FORWARD_MATCH",
                                    f"final s-forward limit elbow recovery failed: {exc}",
                                    feedback="arm_control_failed",
                                    target=target,
                                    plan=plan,
                                )
                        recovery_performed = True
                        self._flush_vision_after_motion()
                        if self._abort_requested():
                            return None, self._abort_failure(plan, target)
                        target, recovery_failure = self._final_view_recheck(plan)
                        if recovery_failure is None:
                            return target, None
                        failure = recovery_failure
                        if not self._final_s_forward_match_feedback_allowed(recovery_failure):
                            recovery_step["stop_reason"] = "feedback_not_allowed"
                            return target, recovery_failure
                    if recovery_performed:
                        recovery_used += 1
                        post_recovery_s_deg = self._current_joint_degrees("s")
                        if post_recovery_s_deg is None:
                            post_recovery_s_deg = float(
                                recovery_step["joints_deg"]["s"]
                            )
                        post_recovery_max_s_deg = float(
                            recovery_step["post_recovery_max_s_deg"]
                        )
                        if bool(
                            recovery_config.get(
                                "accept_at_post_recovery_limit",
                                True,
                            )
                        ):
                            plan.append(
                                self._final_s_forward_post_recovery_limit_accept_step(
                                    failure,
                                    current_s_deg=post_recovery_s_deg,
                                    max_s_deg=post_recovery_max_s_deg,
                                )
                            )
                            accepted_target = (
                                dict(failure.target)
                                if isinstance(failure.target, Mapping)
                                else dict(target) if isinstance(target, Mapping) else {}
                            )
                            return accepted_target, None
                return target, self._failure(
                    "FINAL_S_FORWARD_MATCH",
                    "final s-forward match reached shoulder safety limit",
                    feedback=failure.feedback,
                    target=target,
                    plan=plan,
                )

            return target, self._failure(
                "FINAL_S_FORWARD_MATCH",
                "final s-forward match did not reach shoulder first-stage limit",
                feedback="arm_control_failed",
                target=target,
                plan=plan,
            )
        return target, failure

    def _verify_post_lift_visual_hold(
        self,
        terminal_plan: List[Dict[str, Any]],
        motion_verification: Mapping[str, Any],
    ) -> Dict[str, Any]:
        final_match_config = self.reference.get("final_view_match", {})
        if not (
            isinstance(final_match_config, Mapping)
            and bool(final_match_config.get("enabled", False))
        ):
            return dict(motion_verification)

        self._flush_vision_after_motion()
        image_match = self._match_final_grasp_view_image(None)
        visual_match = {
            "ok": bool(image_match and image_match.get("ok", False)),
            "feedback": str(
                (image_match or {}).get("feedback") or "arm_control_failed"
            ),
            "reason": str(
                (image_match or {}).get("reason")
                or "post-lift final image matcher is unavailable"
            ),
            "metrics": dict((image_match or {}).get("metrics", {}))
            if isinstance((image_match or {}).get("metrics", {}), Mapping)
            else {},
        }
        verification = dict(motion_verification)
        verification["visual_match"] = visual_match
        if not visual_match["ok"]:
            verification.update(
                {
                    "held": False,
                    "reason": (
                        "post-lift visual verification failed: "
                        f"{visual_match['reason']}"
                    ),
                    "verification": "post_lift_visual_rejected",
                }
            )
        elif bool(verification.get("held", False)):
            verification.update(
                {
                    "held": True,
                    "reason": "gripper close and post-lift visual verification passed",
                    "verification": "gripper_and_post_lift_visual",
                }
            )
        terminal_plan[3].update(verification)
        return verification

    def _final_view_failure_feedback(self, target: Mapping[str, Any]) -> str:
        if not self._target_has_final_grasp_flags(target):
            return "target_lost"
        error_u, error_v = self._center_error_px(target)
        if abs(error_u) > FINE_ALIGNMENT_CENTER_TOLERANCE_PX:
            return "target_left" if error_u < 0.0 else "target_right"
        if abs(error_v) > FINE_ALIGNMENT_CENTER_TOLERANCE_PX:
            return "arm_control_failed"
        feedback = self._classify_grasp_window(target)
        if feedback in {"target_above", "target_below", "target_in_grasp_window"}:
            return "arm_control_failed"
        return feedback

    def _classify_grasp_window(self, target: Mapping[str, Any]) -> str:
        error_u, error_v = self._center_error_px(target)
        if error_u < -self.grasp_window_center_tolerance_px:
            return "target_left"
        if error_u > self.grasp_window_center_tolerance_px:
            return "target_right"
        if error_v < -self.grasp_window_center_tolerance_px:
            return "target_above"
        if error_v > self.grasp_window_center_tolerance_px:
            return "target_below"

        ratio = self._size_ratio_to_reference(target)
        if ratio < 1.0 - self.grasp_window_size_ratio_tolerance:
            return "target_too_far"
        if ratio > 1.0 + self.grasp_window_size_ratio_tolerance:
            return "target_too_near"
        return "target_in_grasp_window"

    def _square_face_final_pose_adjustment(
        self,
        target: Mapping[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if not bool(self._square_face_servo_value("final_pose_adjustment_enabled", True)):
            return {}, {}
        reference = self._square_face_target_reference()
        error_u, error_v = self._center_error_px_to_reference(target, reference)
        ratio = self._size_ratio_to_reference_target(target, reference)
        target_angle = float(target.get("angle_deg", 0.0))
        reference_angle = float(reference.get("angle_deg", target_angle))
        angle_error = target_angle - reference_angle
        max_adjust_deg = abs(float(self._square_face_servo_value("final_pose_max_adjust_deg", 2.0)))
        adjustments: Dict[str, float] = {}

        def add_adjustment(joint: str, delta_deg: float) -> None:
            if not joint or not math.isfinite(float(delta_deg)):
                return
            adjustments[joint] = adjustments.get(joint, 0.0) + float(delta_deg)

        add_adjustment(
            str(self._square_face_servo_value("horizontal_joint", "b")),
            error_u
            * float(
                self._square_face_servo_value(
                    "horizontal_gain_deg_per_px",
                    SQUARE_FACE_HORIZONTAL_GAIN_DEG_PER_PX,
                )
            ),
        )
        add_adjustment(
            str(self._square_face_servo_value("vertical_joint", "s")),
            error_v
            * float(
                self._square_face_servo_value(
                    "vertical_gain_deg_per_px",
                    SQUARE_FACE_VERTICAL_GAIN_DEG_PER_PX,
                )
            ),
        )
        add_adjustment(
            str(self._square_face_servo_value("size_joint", "s")),
            (ratio - 1.0)
            * float(
                self._square_face_servo_value(
                    "size_gain_deg_per_ratio",
                    SQUARE_FACE_SIZE_GAIN_DEG_PER_RATIO,
                )
            ),
        )
        if "angle_deg" in reference and bool(target.get("angle_reliable", False)):
            add_adjustment(
                str(self._square_face_servo_value("angle_joint", "w")),
                angle_error
                * float(
                    self._square_face_servo_value(
                        "angle_gain_deg_per_deg",
                        SQUARE_FACE_ANGLE_GAIN_DEG_PER_DEG,
                    )
                ),
            )

        for joint, delta_deg in list(adjustments.items()):
            adjustments[joint] = max(-max_adjust_deg, min(max_adjust_deg, delta_deg))
            if abs(adjustments[joint]) <= 1e-9:
                del adjustments[joint]

        visual_error = {
            "center_px": [error_u, error_v],
            "size_ratio": ratio,
            "angle_deg": angle_error,
        }
        return adjustments, visual_error

    def _final_grasp_pose_step(self, target: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
        approach_sequence = self.reference.get("approach_sequence", {})
        if bool(approach_sequence.get("requires_reteach", False)):
            return None
        pose = approach_sequence.get("grasp_pose_deg", approach_sequence.get("final_grasp_pose_deg"))
        if not pose:
            return None
        reference_pose = {joint: float(value) for joint, value in dict(pose).items()}
        alignment_offset_scale = approach_sequence.get("alignment_offset_scale", 1.0)
        joints_deg, pose_source, current_pose, base_pose = self._pose_relative_to_current_alignment(
            reference_pose,
            alignment_offset_scale,
        )
        expected_reference_pose = dict(reference_pose)
        expected_reference_pose.update(
            {
                joint: float(value)
                for joint, value in dict(
                    approach_sequence.get("grasp_pose_expected_deg", {})
                ).items()
            }
        )
        expected_joints_deg, _, _, _ = self._pose_relative_to_current_alignment(
            expected_reference_pose,
            alignment_offset_scale,
        )
        for joint in FINAL_GRASP_ABSOLUTE_JOINTS:
            if joint in reference_pose:
                joints_deg[joint] = reference_pose[joint]
            if joint in expected_reference_pose:
                expected_joints_deg[joint] = expected_reference_pose[joint]
        visual_adjustment_deg: Dict[str, float] = {}
        visual_error: Dict[str, Any] = {}
        if target is not None:
            visual_adjustment_deg, visual_error = self._square_face_final_pose_adjustment(target)
            for joint, delta_deg in visual_adjustment_deg.items():
                if joint in joints_deg:
                    joints_deg[joint] += delta_deg
                if joint in expected_joints_deg:
                    expected_joints_deg[joint] += delta_deg
        uses_compensated_targets = any(
            abs(expected_joints_deg[joint] - joints_deg[joint]) > 1e-9
            for joint in joints_deg
            if joint in expected_joints_deg
        )
        step = {
            "stage": "MOVE_TO_FINAL_GRASP_POSE",
            "action": "move_joints_with_expected_targets"
            if uses_compensated_targets
            else "move_joints",
            "joints_deg": joints_deg,
            "tolerance_degrees": float(
                approach_sequence.get("pose_tolerance_deg", self._grasp_ready_pose_tolerance_degrees())
            ),
            "pose_source": pose_source,
            "reference_joints_deg": reference_pose,
            "base_joints_deg": base_pose,
            "alignment_offset_scale": alignment_offset_scale,
            "reason": "square face detected; move to saved final grasp offset from current aligned pose"
            if pose_source == "current_aligned_pose"
            else "square face detected; move to saved final grasp view",
        }
        if uses_compensated_targets:
            step["command_joints_deg"] = dict(joints_deg)
            step["expected_joints_deg"] = expected_joints_deg
        if current_pose is not None:
            step["current_joints_deg"] = current_pose
        if target is not None:
            step["visual_error"] = visual_error
            step["visual_adjustment_deg"] = visual_adjustment_deg
        return step

    def _final_pose_confirmation_step(
        self,
        target: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        approach_sequence = self.reference.get("approach_sequence", {})
        if bool(approach_sequence.get("requires_reteach", False)):
            return None
        pose = approach_sequence.get(
            "grasp_pose_deg",
            approach_sequence.get("final_grasp_pose_deg"),
        )
        if not pose:
            return None
        error_u, error_v = self._center_error_px(target)
        return {
            "stage": "MOVE_TO_FINAL_GRASP_POSE",
            "action": "already_at_final_grasp_pose",
            "pose_source": "current_aligned_pose",
            "visual_error": {
                "center_px": [error_u, error_v],
                "size_ratio": self._size_ratio_to_reference(target),
                "angle_deg": self._angle_error_deg(target),
            },
            "visual_adjustment_deg": {},
            "reason": "current view already satisfies the final grasp closure gate",
        }

    def _blocked_alignment_step(self, reason: str, feedback: str) -> Dict[str, Any]:
        return {
            "stage": "VISUAL_ALIGN",
            "action": "blocked",
            "reason": reason,
            "feedback": feedback,
        }

    def _square_face_servo_config(self) -> Mapping[str, Any]:
        visual_servo = self.reference.get("visual_servo", {})
        if not isinstance(visual_servo, Mapping):
            return {}
        square_face = visual_servo.get("square_face", visual_servo)
        return square_face if isinstance(square_face, Mapping) else {}

    def _square_face_servo_value(self, key: str, default: Any) -> Any:
        return self._square_face_servo_config().get(key, default)

    def _target_servo_config(self) -> Mapping[str, Any]:
        visual_servo = self.reference.get("visual_servo", {})
        if not isinstance(visual_servo, Mapping):
            return {}
        target = visual_servo.get("target", {})
        return target if isinstance(target, Mapping) else {}

    def _target_servo_value(self, key: str, default: Any) -> Any:
        return self._target_servo_config().get(key, default)

    def _clamp_jog_delta(self, delta_deg: float, max_delta_deg: Optional[float] = None) -> float:
        if not math.isfinite(float(delta_deg)):
            raise ValueError("jog delta must be finite")
        global_max_delta = abs(float(self.max_jog_deg))
        max_delta = global_max_delta
        if max_delta_deg is not None:
            max_delta = min(global_max_delta, abs(float(max_delta_deg)))
        return max(-max_delta, min(max_delta, float(delta_deg)))

    def _apply_min_jog_delta(
        self,
        delta_deg: float,
        min_delta_deg: Optional[float] = None,
    ) -> float:
        if min_delta_deg is None:
            return float(delta_deg)
        min_delta = abs(float(min_delta_deg))
        delta = float(delta_deg)
        if min_delta <= 0.0 or abs(delta) >= min_delta or abs(delta) <= 1e-9:
            return delta
        return math.copysign(min_delta, delta)

    def _jog_alignment_step(
        self,
        *,
        joint: str,
        delta_deg: float,
        reason: str,
        feedback: Optional[str] = None,
        max_delta_deg: Optional[float] = None,
        min_delta_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        step = {
            "stage": "VISUAL_ALIGN",
            "action": "jog",
            "joint": joint,
            "delta_deg": self._clamp_jog_delta(
                self._apply_min_jog_delta(delta_deg, min_delta_deg=min_delta_deg),
                max_delta_deg=max_delta_deg,
            ),
            "reason": reason,
        }
        if feedback:
            step["feedback"] = feedback
        return step

    def _center_error_px_to_reference(
        self,
        target: Mapping[str, Any],
        reference: Mapping[str, Any],
    ) -> Tuple[float, float]:
        target_center = target.get("center_px", [0.0, 0.0])
        reference_center = reference.get("center_px", [0.0, 0.0])
        return (
            float(target_center[0]) - float(reference_center[0]),
            float(target_center[1]) - float(reference_center[1]),
        )

    def _size_ratio_to_reference_target(
        self,
        target: Mapping[str, Any],
        reference: Mapping[str, Any],
    ) -> float:
        target_size = target.get("size_px", [0.0, 0.0])
        reference_size = reference.get("size_px", [0.0, 0.0])
        target_long = max(float(target_size[0]), float(target_size[1]))
        reference_long = max(float(reference_size[0]), float(reference_size[1]))
        if reference_long <= 0.0:
            return 1.0
        return target_long / reference_long

    def _square_face_alignment_step(self, target: Mapping[str, Any]) -> Dict[str, Any]:
        reference = self._square_face_target_reference()
        error_u, error_v = self._center_error_px_to_reference(target, reference)
        ratio = self._size_ratio_to_reference_target(target, reference)
        target_angle = float(target.get("angle_deg", 0.0))
        reference_angle = float(reference.get("angle_deg", target_angle))
        angle_error = target_angle - reference_angle
        horizontal_tolerance_px = float(
            self._square_face_servo_value("horizontal_tolerance_px", FINE_ALIGNMENT_CENTER_TOLERANCE_PX)
        )
        vertical_tolerance_px = float(
            self._square_face_servo_value("vertical_tolerance_px", FINE_ALIGNMENT_CENTER_TOLERANCE_PX)
        )
        size_ratio_tolerance = float(
            self._square_face_servo_value("size_ratio_tolerance", self.grasp_window_size_ratio_tolerance)
        )
        angle_tolerance_deg = float(
            self._square_face_servo_value("angle_tolerance_deg", FINE_ALIGNMENT_ANGLE_TOLERANCE_DEG)
        )

        def vertical_alignment_step() -> Dict[str, Any]:
            feedback = "square_face_above" if error_v < 0.0 else "square_face_below"
            return self._jog_alignment_step(
                joint=str(self._square_face_servo_value("vertical_joint", "s")),
                delta_deg=error_v
                * float(
                    self._square_face_servo_value(
                        "vertical_gain_deg_per_px",
                        SQUARE_FACE_VERTICAL_GAIN_DEG_PER_PX,
                    )
                ),
                reason="square face center_px vertical error",
                feedback=feedback,
                max_delta_deg=self._square_face_servo_value("vertical_max_jog_deg", None),
                min_delta_deg=self._square_face_servo_value("vertical_min_jog_deg", None),
            )

        vertical_priority_above_px = max(
            vertical_tolerance_px,
            float(self._square_face_servo_value("vertical_priority_above_px", 60.0)),
        )
        if error_v < -vertical_priority_above_px:
            return vertical_alignment_step()

        if abs(error_u) > horizontal_tolerance_px:
            feedback = "square_face_left" if error_u < 0.0 else "square_face_right"
            return self._jog_alignment_step(
                joint=str(self._square_face_servo_value("horizontal_joint", "b")),
                delta_deg=error_u
                * float(
                    self._square_face_servo_value(
                        "horizontal_gain_deg_per_px",
                        SQUARE_FACE_HORIZONTAL_GAIN_DEG_PER_PX,
                    )
                ),
                reason="square face center_px horizontal error",
                feedback=feedback,
                max_delta_deg=self._square_face_servo_value("horizontal_max_jog_deg", None),
                min_delta_deg=self._square_face_servo_value("horizontal_min_jog_deg", None),
            )
        if abs(error_v) > vertical_tolerance_px:
            return vertical_alignment_step()
        if bool(self._square_face_servo_value("size_alignment_enabled", True)):
            if ratio < 1.0 - size_ratio_tolerance:
                return self._jog_alignment_step(
                    joint=str(self._square_face_servo_value("size_joint", "s")),
                    delta_deg=(ratio - 1.0)
                    * float(
                        self._square_face_servo_value(
                            "size_gain_deg_per_ratio",
                            SQUARE_FACE_SIZE_GAIN_DEG_PER_RATIO,
                        )
                    ),
                    reason="square face size error",
                    feedback="square_face_too_far",
                    max_delta_deg=self._square_face_servo_value("size_max_jog_deg", None),
                )
            if ratio > 1.0 + size_ratio_tolerance:
                return self._jog_alignment_step(
                    joint=str(self._square_face_servo_value("size_joint", "s")),
                    delta_deg=(ratio - 1.0)
                    * float(
                        self._square_face_servo_value(
                            "size_gain_deg_per_ratio",
                            SQUARE_FACE_SIZE_GAIN_DEG_PER_RATIO,
                        )
                    ),
                    reason="square face size error",
                    feedback="square_face_too_near",
                    max_delta_deg=self._square_face_servo_value("size_max_jog_deg", None),
                )
        if "angle_deg" in reference and bool(target.get("angle_reliable", False)):
            if abs(angle_error) > angle_tolerance_deg:
                return self._jog_alignment_step(
                    joint=str(self._square_face_servo_value("angle_joint", "w")),
                    delta_deg=angle_error
                    * float(
                        self._square_face_servo_value(
                            "angle_gain_deg_per_deg",
                            SQUARE_FACE_ANGLE_GAIN_DEG_PER_DEG,
                        )
                    ),
                    reason="square face angle_deg error",
                    max_delta_deg=self._square_face_servo_value("angle_max_jog_deg", None),
                )

        final_pose_step = self._final_grasp_pose_step(target)
        if final_pose_step is not None:
            if (
                not self.dry_run
                and final_pose_step.get("pose_source") != "current_aligned_pose"
            ):
                return self._blocked_alignment_step(
                    "current arm pose is unavailable; dynamic final grasp pose cannot be computed",
                    "arm_control_failed",
                )
            return final_pose_step
        return self._blocked_alignment_step(
            "square face matched but no final grasp pose is configured",
            "missing_final_grasp_pose",
        )

    def _alignment_step(
        self,
        target: Mapping[str, Any],
        *,
        allow_final_pose_move: bool = True,
        allow_closure: bool = True,
    ) -> Dict[str, Any]:
        if allow_closure and self._target_ready_for_closure(target):
            return {"stage": "VISUAL_ALIGN", "action": "already_aligned", "reason": "target close to reference"}

        if allow_final_pose_move and self._target_ready_for_square_face_tracking(target):
            return self._square_face_alignment_step(target)

        error_u, error_v = self._center_error_px(target)
        error_angle = self._angle_error_deg(target)

        if abs(error_u) > FINE_ALIGNMENT_CENTER_TOLERANCE_PX:
            return {
                "stage": "VISUAL_ALIGN",
                "action": "jog",
                "joint": "b",
                "delta_deg": max(-self.max_jog_deg, min(self.max_jog_deg, -error_u / 80.0)),
                "reason": "center_px horizontal error",
            }
        if abs(error_v) > FINE_ALIGNMENT_CENTER_TOLERANCE_PX:
            feedback = "target_above" if error_v < 0.0 else "target_below"
            return self._jog_alignment_step(
                joint=str(self._target_servo_value("vertical_joint", "s")),
                delta_deg=error_v
                * float(
                    self._target_servo_value(
                        "vertical_gain_deg_per_px",
                        TARGET_VERTICAL_GAIN_DEG_PER_PX,
                    )
                ),
                reason="target center_px vertical error",
                feedback=feedback,
            )

        feedback = self._classify_grasp_window(target)
        if feedback == "target_too_far":
            ratio = self._size_ratio_to_reference(target)
            return self._jog_alignment_step(
                joint=str(self._target_servo_value("size_joint", "s")),
                delta_deg=(ratio - 1.0)
                * float(
                    self._target_servo_value(
                        "size_gain_deg_per_ratio",
                        SQUARE_FACE_SIZE_GAIN_DEG_PER_RATIO,
                    )
                ),
                reason="target size error",
                feedback=feedback,
            )
        if feedback == "target_too_near":
            ratio = self._size_ratio_to_reference(target)
            return self._jog_alignment_step(
                joint=str(self._target_servo_value("size_joint", "s")),
                delta_deg=(ratio - 1.0)
                * float(
                    self._target_servo_value(
                        "size_gain_deg_per_ratio",
                        SQUARE_FACE_SIZE_GAIN_DEG_PER_RATIO,
                    )
                ),
                reason="target size error",
                feedback=feedback,
            )

        reference_target = self.reference.get("target", {})
        if (
            bool(reference_target.get("require_angle_reliable", True))
            and bool(target.get("angle_reliable", True))
            and abs(error_angle) > FINE_ALIGNMENT_ANGLE_TOLERANCE_DEG
        ):
            return {
                "stage": "VISUAL_ALIGN",
                "action": "jog",
                "joint": "w",
                "delta_deg": max(-self.max_jog_deg, min(self.max_jog_deg, -error_angle / 10.0)),
                "reason": "angle_deg error",
            }
        return self._blocked_alignment_step(
            "target does not satisfy final grasp requirements",
            "target_not_final_grasp_view",
        )

    def _terminal_plan(self) -> List[Dict[str, Any]]:
        start_sequence = self.reference.get("start_sequence", {})
        terminal = self.reference.get("terminal_sequence", {})
        configured_cargo_pose = dict(
            terminal.get(
                "cargo_joints_deg",
                terminal.get(
                    "transport_joints_deg",
                    start_sequence.get(
                        "cargo_pose_deg",
                        start_sequence.get(
                            "transport_pose_deg",
                            dict(DEFAULT_GRASP_READY_POSE_DEG),
                        ),
                    ),
                ),
            )
        )
        cargo_joints = {
            joint: float(configured_cargo_pose.get(joint, DEFAULT_GRASP_READY_POSE_DEG[joint]))
            for joint in ("s", "e")
        }
        return [
            {
                "stage": "CLOSE_GRIPPER",
                "close_gripper_h": float(terminal.get("close_gripper_h", 45.0)),
            },
            {
                "stage": CARGO_POSE_STAGE,
                "pose_name": CARGO_POSE_NAME,
                "joints_deg": cargo_joints,
                "preserve_current_joints": ["b", "w"],
                "keep_gripper_closed": True,
            },
        ]

    def _execute_terminal(self, terminal_plan: List[Dict[str, Any]]) -> None:
        spd, acc = self._final_motion_speed()
        try:
            self.motion.close_gripper(
                angle=terminal_plan[0]["close_gripper_h"],
                spd=spd,
                acc=acc,
            )
        except Exception as exc:
            raise TerminalStepError("CLOSE_GRIPPER", f"close gripper failed: {exc}") from exc

    def _execute_transport_pose(self, terminal_plan: List[Dict[str, Any]]) -> None:
        transport_step = next(
            step for step in terminal_plan
            if step.get("stage") == CARGO_POSE_STAGE
        )
        spd, acc = self._final_motion_speed()
        try:
            if transport_step["joints_deg"]:
                self.motion.move_joints(
                    transport_step["joints_deg"],
                    spd=spd,
                    acc=acc,
                )
        except Exception as exc:
            raise TerminalStepError(
                CARGO_POSE_STAGE,
                f"cargo pose failed: {exc}",
            ) from exc

    @staticmethod
    def _final_s_forward_no_red_fallback_triggered(plan: Iterable[Mapping[str, Any]]) -> bool:
        return any(
            step.get("stage") == "FINAL_S_FORWARD_LIMIT_NO_RED_ACCEPT"
            for step in plan
        )

    def _finish_no_red_fallback_to_cargo(
        self,
        plan: List[Dict[str, Any]],
        target: Optional[Mapping[str, Any]],
        terminal_plan: List[Dict[str, Any]],
    ) -> GraspResult:
        close_step = terminal_plan[0]
        cargo_step = next(
            step for step in terminal_plan
            if step.get("stage") == CARGO_POSE_STAGE
        )
        plan.append(close_step)
        try:
            self.motion.close_gripper(
                angle=close_step["close_gripper_h"],
                spd=self.final_spd,
                acc=self.final_acc,
            )
        except Exception as exc:
            return self._failure(
                "CLOSE_GRIPPER",
                f"close gripper failed: {exc}",
                feedback="arm_control_failed",
                target=target,
                plan=plan,
            )

        plan.append(cargo_step)
        try:
            self._execute_transport_pose(terminal_plan)
        except TerminalStepError as exc:
            return self._failure(
                exc.stage,
                str(exc),
                feedback="arm_control_failed",
                target=target,
                plan=plan,
                object_held=True,
            )

        result_target = dict(target) if isinstance(target, Mapping) else {}
        return GraspResult(
            True,
            "DONE",
            feedback="fallback_cargo_pose",
            object_held=True,
            target=result_target,
            plan=plan,
        )

    def _alignment_preamble(self, plan: List[Dict[str, Any]], alignment_step: Mapping[str, Any]) -> List[Dict[str, Any]]:
        if alignment_step.get("stage") != "MOVE_TO_FINAL_GRASP_POSE":
            return []
        if alignment_step.get("action") not in (
            "move_joints",
            "move_joints_with_expected_targets",
        ):
            return []
        steps: List[Dict[str, Any]] = []
        if not any(
            step.get("stage") in ("OPEN_GRIPPER", "OPEN_GRIPPER_FOR_APPROACH")
            for step in plan
        ):
            open_step = self._approach_open_plan()
            if open_step is not None:
                steps.append(open_step)
        elbow_step = self._elbow_first_final_approach_step(alignment_step)
        if elbow_step is not None:
            steps.append(elbow_step)
        return steps

    def _alignment_postamble(self, alignment_step: Mapping[str, Any]) -> List[Dict[str, Any]]:
        if self._should_stop_after_final_pose(alignment_step):
            return []
        lower_step = self._lower_elbow_after_final_approach_step(alignment_step)
        return [lower_step] if lower_step is not None else []

    def _should_stop_after_final_pose(self, alignment_step: Mapping[str, Any]) -> bool:
        return (
            self.stop_after_final_pose
            and alignment_step.get("stage") == "MOVE_TO_FINAL_GRASP_POSE"
            and alignment_step.get("action")
            in ("move_joints", "move_joints_with_expected_targets")
        )

    def _elbow_first_final_approach_step(
        self,
        final_step: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        current_joints = final_step.get("current_joints_deg")
        if not isinstance(current_joints, Mapping) or "e" not in current_joints:
            return None

        command_joints = final_step.get(
            "command_joints_deg",
            final_step.get("joints_deg", {}),
        )
        expected_joints = final_step.get(
            "expected_joints_deg",
            final_step.get("joints_deg", {}),
        )
        if (
            not isinstance(command_joints, Mapping)
            or not isinstance(expected_joints, Mapping)
            or "e" not in command_joints
            or "e" not in expected_joints
        ):
            return None

        current_e = float(current_joints["e"])
        command_e = float(command_joints["e"])
        expected_e = float(expected_joints["e"])
        if expected_e >= current_e - FINAL_APPROACH_ELBOW_FIRST_MIN_DELTA_DEG:
            return None

        if not isinstance(final_step, dict):
            raise TypeError("final grasp pose step must be mutable")
        for key in ("joints_deg", "command_joints_deg", "expected_joints_deg"):
            values = final_step.get(key)
            if isinstance(values, Mapping):
                remaining = dict(values)
                remaining.pop("e", None)
                final_step[key] = remaining
        final_step["elbow_prepositioned"] = True
        final_step["held_elbow_expected_deg"] = expected_e

        elbow_step: Dict[str, Any] = {
            "stage": "RAISE_ELBOW_FOR_FINAL_APPROACH",
            "action": "move_joints",
            "joints_deg": {"e": expected_e},
            "tolerance_degrees": final_step.get("tolerance_degrees"),
            "reason": "raise elbow to verified final height before shoulder approach",
        }
        if abs(command_e - expected_e) > 1e-9:
            elbow_step.update(
                {
                    "action": "move_joints_with_expected_targets",
                    "joints_deg": {"e": command_e},
                    "command_joints_deg": {"e": command_e},
                    "expected_joints_deg": {"e": expected_e},
                }
            )
        return elbow_step

    def _lower_elbow_after_final_approach_step(
        self,
        final_step: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if final_step.get("stage") != "MOVE_TO_FINAL_GRASP_POSE":
            return None
        if final_step.get("action") not in (
            "move_joints",
            "move_joints_with_expected_targets",
        ):
            return None
        if not bool(final_step.get("elbow_prepositioned", False)):
            return None

        approach_sequence = self.reference.get("approach_sequence", {})
        if not isinstance(approach_sequence, Mapping):
            return None
        lower_delta_deg = float(
            approach_sequence.get("final_grasp_lower_elbow_delta_deg", 0.0)
        )
        if abs(lower_delta_deg) <= 1e-9:
            return None

        held_elbow = final_step.get("held_elbow_expected_deg")
        if held_elbow is None:
            return None
        expected_elbow_deg = float(held_elbow) + lower_delta_deg
        command_elbow_deg = expected_elbow_deg
        if lower_delta_deg < 0.0:
            command_elbow_deg -= GRASP_READY_NEGATIVE_E_COMPENSATION_DEG
        step: Dict[str, Any] = {
            "stage": "LOWER_ELBOW_FOR_FINAL_GRASP",
            "action": "move_joints",
            "joints_deg": {"e": command_elbow_deg},
            "tolerance_degrees": final_step.get("tolerance_degrees"),
            "reason": "lower elbow from clearance height to final grasp height",
            "source_elbow_expected_deg": float(held_elbow),
            "delta_deg": lower_delta_deg,
        }
        if abs(command_elbow_deg - expected_elbow_deg) > 1e-9:
            step.update(
                {
                    "action": "move_joints_with_expected_targets",
                    "command_joints_deg": {"e": command_elbow_deg},
                    "expected_joints_deg": {"e": expected_elbow_deg},
                    "e_negative_compensation_deg": GRASP_READY_NEGATIVE_E_COMPENSATION_DEG,
                }
            )
        return step

    def _execute_alignment_preamble(self, steps: Iterable[Mapping[str, Any]]) -> None:
        for step in steps:
            if step.get("stage") == "OPEN_GRIPPER_FOR_APPROACH":
                self.motion.open_gripper(
                    angle=step["open_gripper_h"],
                    spd=self.spd,
                    acc=self.acc,
                )
            else:
                self._execute_alignment_motion(step)

    def _execute_alignment_postamble(
        self,
        steps: Iterable[Mapping[str, Any]],
        *,
        plan: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        plan = [] if plan is None else plan
        for step in steps:
            if step.get("stage") == "LOWER_ELBOW_FOR_FINAL_GRASP":
                if self._execute_lower_elbow_with_no_red_recovery(step, plan):
                    return True
                continue
            self._execute_alignment_motion(step)
        return False

    def _final_approach_red_visible(self) -> Optional[bool]:
        visibility_check = getattr(self.vision, "red_visible_after_motion", None)
        if not callable(visibility_check):
            return None
        try:
            return visibility_check(max_frames=FINAL_APPROACH_RED_VISIBILITY_FRAMES)
        except Exception:
            return None

    def _final_approach_visibility_plan_step(
        self,
        visible: Optional[bool],
    ) -> Dict[str, Any]:
        details: Dict[str, Any] = {}
        details_getter = getattr(self.vision, "last_red_visibility_check", None)
        if callable(details_getter):
            try:
                reported_details = details_getter()
                if isinstance(reported_details, Mapping):
                    details = dict(reported_details)
            except Exception:
                details = {}
        missing = object()
        if details.get("visible", missing) is not visible:
            details = {}

        max_frames = details.get(
            "max_frames",
            FINAL_APPROACH_RED_VISIBILITY_FRAMES,
        )
        inspected_frames = details.get(
            "inspected_frames",
            0 if visible is None else max_frames,
        )
        result = {
            True: "visible",
            False: "not_visible",
            None: "camera_unknown",
        }[visible]
        return {
            "stage": "FINAL_APPROACH_RED_VISIBILITY_CHECK",
            "result": result,
            "visible": visible,
            "max_frames": max_frames,
            "inspected_frames": inspected_frames,
            "loose_red_hint": details.get("loose_red_hint"),
            "raw_image": details.get("raw_image"),
            "annotated_image": details.get("annotated_image"),
        }

    def _record_final_approach_visibility(
        self,
        plan: List[Dict[str, Any]],
        visible: Optional[bool],
        *,
        before_step: Optional[Mapping[str, Any]] = None,
    ) -> None:
        visibility_step = self._final_approach_visibility_plan_step(visible)
        if before_step is not None:
            for index, planned_step in enumerate(plan):
                if planned_step is before_step:
                    plan.insert(index, visibility_step)
                    return
        plan.append(visibility_step)

    def _execute_lower_elbow_with_no_red_recovery(
        self,
        step: Mapping[str, Any],
        plan: List[Dict[str, Any]],
    ) -> bool:
        if step.get("stage") != "LOWER_ELBOW_FOR_FINAL_GRASP":
            return False

        original_step = dict(step)
        self._flush_vision_after_motion()
        red_visible = self._final_approach_red_visible()
        self._record_final_approach_visibility(
            plan,
            red_visible,
            before_step=step,
        )
        if red_visible is True:
            self._execute_alignment_motion(step)
            return False

        approach_sequence = self.reference.get("approach_sequence", {})
        if not isinstance(approach_sequence, Mapping):
            approach_sequence = {}
        try:
            recovery_delta_deg = abs(
                float(
                    approach_sequence.get(
                        "final_grasp_no_red_e_recovery_delta_deg",
                        FINAL_APPROACH_NO_RED_E_RECOVERY_DELTA_DEG,
                    )
                )
            )
            recovery_max_steps = max(
                0,
                int(
                    approach_sequence.get(
                        "final_grasp_no_red_e_recovery_max_steps",
                        FINAL_APPROACH_NO_RED_E_RECOVERY_MAX_STEPS,
                    )
                ),
            )
        except (TypeError, ValueError):
            recovery_delta_deg = 0.0
            recovery_max_steps = 0
        if not math.isfinite(recovery_delta_deg) or recovery_delta_deg <= 0.0:
            recovery_max_steps = 0
        if recovery_max_steps <= 0:
            self._execute_alignment_motion(step)
            return False

        recovery_total_deg = 0.0
        for attempt in range(1, recovery_max_steps + 1):
            recovery_total_deg += recovery_delta_deg
            recovery_step = {
                "stage": "FINAL_APPROACH_NO_RED_E_RECOVERY",
                "action": "jog",
                "joint": "e",
                "delta_deg": recovery_delta_deg,
                "attempt": attempt,
                "max_steps": recovery_max_steps,
                "recovery_total_deg": recovery_total_deg,
                "reason": "no red target visible after final pose; raise elbow and check fresh frames",
                "feedback": "target_lost",
            }
            if attempt == 1 and isinstance(step, dict):
                step.clear()
                step.update(recovery_step)
            else:
                plan.append(recovery_step)
            self._execute_alignment_motion(recovery_step)
            self._flush_vision_after_motion()
            red_visible = self._final_approach_red_visible()
            self._record_final_approach_visibility(plan, red_visible)
            if red_visible is True:
                lower_step = dict(original_step)
                configured_delta_deg = float(lower_step.get("delta_deg", 0.0))
                lower_step.update(
                    {
                        "configured_delta_deg": configured_delta_deg,
                        "recovery_total_deg": recovery_total_deg,
                        "effective_delta_deg": configured_delta_deg - recovery_total_deg,
                        "recovery_steps": attempt,
                        "reason": (
                            "red target visible after elbow recovery; lower to the original "
                            "final grasp target"
                        ),
                    }
                )
                plan.append(lower_step)
                self._execute_alignment_motion(lower_step)
                return False

        plan.append(
            {
                "stage": "FINAL_APPROACH_NO_RED_RECOVERY_ACCEPT",
                "action": "accept_fallback",
                "reason": (
                    f"no red target visible after {recovery_max_steps} elbow recovery steps; "
                    "close gripper and enter cargo pose"
                ),
                "feedback": "target_lost",
                "recovery_steps": recovery_max_steps,
                "recovery_total_deg": recovery_total_deg,
                "skipped_stage": original_step["stage"],
                "skipped_joints_deg": dict(original_step.get("joints_deg", {})),
            }
        )
        return True

    def _final_motion_speed(self) -> Tuple[float, float]:
        return self.final_spd, self.final_acc

    def _speed_for_alignment_step(self, alignment_step: Mapping[str, Any]) -> Tuple[float, float]:
        if alignment_step.get("stage") in {
            "FINAL_S_FORWARD_MATCH",
            "FINAL_VIEW_CORRECTION",
            "LOWER_ELBOW_FOR_FINAL_GRASP",
            "FINAL_APPROACH_NO_RED_E_RECOVERY",
        }:
            return self._final_motion_speed()
        return self.spd, self.acc

    def _execute_alignment_motion(self, alignment_step: Mapping[str, Any]) -> None:
        action = alignment_step.get("action")
        spd, acc = self._speed_for_alignment_step(alignment_step)
        if action == "jog":
            self.motion.jog(
                alignment_step["joint"],
                alignment_step["delta_deg"],
                spd=spd,
                acc=acc,
            )
            return
        if action == "move_joints":
            self.motion.move_joints(
                alignment_step["joints_deg"],
                spd=spd,
                acc=acc,
                tolerance_degrees=alignment_step.get("tolerance_degrees"),
            )
            return
        if action == "move_joints_with_expected_targets":
            self.motion.move_joints_with_expected_targets(
                alignment_step["command_joints_deg"],
                alignment_step["expected_joints_deg"],
                spd=spd,
                acc=acc,
                tolerance_degrees=alignment_step.get("tolerance_degrees"),
            )
            return
        raise RuntimeError(f"alignment action cannot be executed: {action}")

    def _flush_vision_after_motion(self) -> None:
        flush_after_motion = getattr(self.vision, "flush_after_motion", None)
        if callable(flush_after_motion):
            flush_after_motion()

    def _alignment_feedback(self, target: Mapping[str, Any], alignment_step: Mapping[str, Any]) -> str:
        if alignment_step.get("stage") == "MOVE_TO_FINAL_GRASP_POSE":
            return "target_in_grasp_window"
        return str(alignment_step.get("feedback") or self._classify_grasp_window(target))

    def _public_alignment_feedback(
        self,
        target: Mapping[str, Any],
        alignment_step: Mapping[str, Any],
    ) -> str:
        feedback = self._alignment_feedback(target, alignment_step)
        feedback_map = {
            "square_face_left": "target_left",
            "square_face_right": "target_right",
            "square_face_too_far": "target_too_far",
            "square_face_too_near": "target_too_near",
            "square_face_above": "target_in_grasp_window",
            "square_face_below": "target_in_grasp_window",
            "target_above": "target_in_grasp_window",
            "target_below": "target_in_grasp_window",
            "target_not_final_grasp_view": "arm_control_failed",
        }
        return feedback_map.get(feedback, feedback)

    def _alignment_exhausted_feedback(
        self,
        target: Mapping[str, Any],
        alignment_step: Mapping[str, Any],
    ) -> str:
        feedback = str(alignment_step.get("feedback", ""))
        feedback_map = {
            "square_face_left": "target_left",
            "square_face_right": "target_right",
            "square_face_too_far": "target_too_far",
            "square_face_too_near": "target_too_near",
            "square_face_above": "arm_control_failed",
            "square_face_below": "arm_control_failed",
        }
        if feedback in feedback_map:
            return feedback_map[feedback]
        if feedback in {"target_above", "target_below"}:
            return "arm_control_failed"
        if feedback.startswith("target_") and feedback != "target_in_grasp_window":
            return feedback

        reason = str(alignment_step.get("reason", ""))
        reference = (
            self._square_face_target_reference()
            if reason.startswith("square face")
            else self.reference.get("target", {})
        )
        if "horizontal" in reason:
            error_u, _ = self._center_error_px_to_reference(target, reference)
            return "target_left" if error_u < 0.0 else "target_right"
        if "vertical" in reason:
            return "arm_control_failed"
        if "size" in reason:
            ratio = self._size_ratio_to_reference_target(target, reference)
            return "target_too_far" if ratio < 1.0 else "target_too_near"

        classified = self._classify_grasp_window(target)
        return classified if classified != "target_in_grasp_window" else "arm_control_failed"

    def _alignment_error_value(
        self,
        target: Mapping[str, Any],
        alignment_step: Mapping[str, Any],
    ) -> Optional[float]:
        if alignment_step.get("action") != "jog":
            return None
        reason = str(alignment_step.get("reason", ""))
        reference = (
            self._square_face_target_reference()
            if reason.startswith("square face")
            else self.reference.get("target", {})
        )
        if "horizontal" in reason:
            error_u, _ = self._center_error_px_to_reference(target, reference)
            return abs(error_u)
        if "vertical" in reason:
            _, error_v = self._center_error_px_to_reference(target, reference)
            return abs(error_v)
        if "size" in reason:
            return abs(self._size_ratio_to_reference_target(target, reference) - 1.0)
        if "angle" in reason:
            target_angle = float(target.get("angle_deg", 0.0))
            reference_angle = float(reference.get("angle_deg", target_angle))
            return abs(target_angle - reference_angle)
        return None

    def _alignment_guard_failure(
        self,
        plan: List[Dict[str, Any]],
        target: Mapping[str, Any],
        *,
        guard: str,
        reason: str,
        details: Mapping[str, Any],
    ) -> GraspResult:
        feedback = str(details.get("feedback") or "arm_control_failed")
        detail_payload = {key: value for key, value in details.items() if key != "feedback"}
        plan.append(
            {
                "stage": "VISUAL_ALIGN",
                "action": "blocked",
                "guard": guard,
                "reason": reason,
                "feedback": feedback,
                **detail_payload,
            }
        )
        return self._failure(
            "VISUAL_ALIGN",
            reason,
            feedback=feedback,
            target=target,
            plan=plan,
        )

    def run(self) -> GraspResult:
        plan: List[Dict[str, Any]] = []
        if self._abort_requested():
            return self._abort_failure(plan)

        if self.run_grasp_ready:
            start_failure = self._execute_grasp_ready_pose(plan)
            if start_failure:
                return start_failure
            if self._abort_requested():
                return self._abort_failure(plan)
        else:
            open_failure = self._execute_gripper_open(plan)
            if open_failure:
                return open_failure
            if self._abort_requested():
                return self._abort_failure(plan)

        target, failure = self._detect_ready_target(
            plan,
            allow_unstable_retry=True,
        )
        if failure:
            target, search_result = self._try_initial_loose_red_search(plan, failure)
            if search_result:
                return search_result

        alignment_step = self._alignment_step(
            target,
            allow_closure=not self._target_ready_for_square_face_tracking(target),
        )
        alignment_step["max_align_steps"] = self.max_align_steps
        alignment_preamble = self._alignment_preamble(plan, alignment_step)
        alignment_postamble = self._alignment_postamble(alignment_step)
        plan.extend(alignment_preamble)
        plan.append(alignment_step)
        plan.extend(alignment_postamble)
        if alignment_step.get("action") == "already_aligned":
            final_pose_confirmation = self._final_pose_confirmation_step(target)
            if final_pose_confirmation is not None:
                plan.append(final_pose_confirmation)

        if self.single_step:
            if self._abort_requested():
                return self._abort_failure(plan, target)
            if alignment_step["action"] == "blocked":
                return self._failure(
                    "VISUAL_ALIGN",
                    alignment_step["reason"],
                    feedback=self._public_alignment_feedback(target, alignment_step),
                    target=target,
                    plan=plan,
            )
            try:
                if not self.dry_run and alignment_step["action"] in (
                    "jog",
                    "move_joints",
                    "move_joints_with_expected_targets",
                ):
                    self._execute_alignment_preamble(alignment_preamble)
                    if self._abort_requested():
                        return self._abort_failure(plan, target)
                    self._execute_alignment_motion(alignment_step)
                    if self._abort_requested():
                        return self._abort_failure(plan, target)
                    self._execute_alignment_postamble(
                        alignment_postamble,
                        plan=plan,
                    )
            except Exception as exc:
                return self._failure("ARM_CONTROL", str(exc), feedback="arm_control_failed", target=target, plan=plan)
            return GraspResult(
                True,
                "SINGLE_STEP",
                feedback=self._public_alignment_feedback(target, alignment_step),
                object_held=False,
                target=dict(target),
                plan=plan,
            )

        terminal_plan = self._terminal_plan()
        if self.dry_run:
            if self._should_stop_after_final_pose(alignment_step):
                return GraspResult(
                    True,
                    "STOPPED_AFTER_FINAL_POSE",
                    feedback=self._public_alignment_feedback(target, alignment_step),
                    object_held=False,
                    target=dict(target),
                    plan=plan,
                )
            if alignment_step["action"] != "already_aligned":
                return GraspResult(
                    True,
                    "VISUAL_ALIGN",
                    feedback=self._public_alignment_feedback(target, alignment_step),
                    object_held=False,
                    target=dict(target),
                    plan=plan,
                )
            plan.append(
                {
                    "stage": "FINAL_VIEW_RECHECK",
                    "target": dict(target),
                    "dry_run": True,
                }
            )
            return GraspResult(
                True,
                "DONE",
                feedback="target_in_grasp_window",
                object_held=False,
                target=dict(target),
                    plan=plan + terminal_plan,
                )

        final_pose_move_used = False
        stagnant_steps = 0
        direction_reversals = 0
        try:
            for step_index in range(self.max_align_steps + 1):
                if self._abort_requested():
                    return self._abort_failure(plan, target)
                if alignment_step["action"] == "already_aligned":
                    if self._abort_requested():
                        return self._abort_failure(plan, target)
                    target, final_view_failure = self._final_view_recheck_with_s_forward_match(plan)
                    if final_view_failure:
                        return final_view_failure
                    if self._final_s_forward_no_red_fallback_triggered(plan):
                        return self._finish_no_red_fallback_to_cargo(
                            plan,
                            target,
                            terminal_plan,
                        )
                    plan.extend(terminal_plan)
                    try:
                        self._execute_terminal(terminal_plan)
                    except TerminalStepError as exc:
                        return self._failure(
                            exc.stage,
                            str(exc),
                            feedback="arm_control_failed",
                            target=target,
                            plan=plan,
                        )
                    try:
                        self._execute_transport_pose(terminal_plan)
                    except TerminalStepError as exc:
                        return self._failure(
                            exc.stage,
                            str(exc),
                            feedback="arm_control_failed",
                            target=target,
                            plan=plan,
                            object_held=True,
                        )
                    return GraspResult(
                        True,
                        "DONE",
                        feedback="target_in_grasp_window",
                        object_held=True,
                        target=dict(target),
                        plan=plan,
                    )
                if alignment_step["action"] == "blocked":
                    return self._failure(
                        "VISUAL_ALIGN",
                        alignment_step["reason"],
                        feedback=self._public_alignment_feedback(target, alignment_step),
                        target=target,
                        plan=plan,
                    )
                if step_index >= self.max_align_steps:
                    alignment_feedback = self._alignment_feedback(target, alignment_step)
                    exhausted_feedback = self._alignment_exhausted_feedback(
                        target,
                        alignment_step,
                    )
                    plan.append(
                        {
                            "stage": "VISUAL_ALIGN",
                            "action": "blocked",
                            "guard": "max_steps",
                            "reason": "alignment steps exhausted",
                            "feedback": exhausted_feedback,
                            "alignment_feedback": alignment_feedback,
                            "max_align_steps": self.max_align_steps,
                        }
                    )
                    return self._failure(
                        "VISUAL_ALIGN",
                        "alignment steps exhausted",
                        feedback=exhausted_feedback,
                        target=target,
                        plan=plan,
                    )
                if self._abort_requested():
                    return self._abort_failure(plan, target)
                target_before_motion = dict(target)
                executed_alignment_step = dict(alignment_step)
                self._execute_alignment_preamble(alignment_preamble)
                if self._abort_requested():
                    return self._abort_failure(plan, target)
                self._execute_alignment_motion(alignment_step)
                if self._abort_requested():
                    return self._abort_failure(plan, target)
                final_approach_no_red_fallback = self._execute_alignment_postamble(
                    alignment_postamble,
                    plan=plan,
                )
                if final_approach_no_red_fallback:
                    return self._finish_no_red_fallback_to_cargo(
                        plan,
                        target,
                        terminal_plan,
                    )
                self._flush_vision_after_motion()
                if self._abort_requested():
                    return self._abort_failure(plan, target)
                if alignment_step["action"] in (
                    "move_joints",
                    "move_joints_with_expected_targets",
                ):
                    final_pose_move_used = True
                    if self._should_stop_after_final_pose(executed_alignment_step):
                        return GraspResult(
                            True,
                            "STOPPED_AFTER_FINAL_POSE",
                            feedback=self._public_alignment_feedback(
                                target,
                                executed_alignment_step,
                            ),
                            object_held=False,
                            target=dict(target),
                            plan=plan,
                        )
                    target, final_view_failure = self._final_view_recheck_with_s_forward_match(plan)
                    if final_view_failure:
                        return final_view_failure
                    if self._final_s_forward_no_red_fallback_triggered(plan):
                        return self._finish_no_red_fallback_to_cargo(
                            plan,
                            target,
                            terminal_plan,
                        )
                    plan.extend(terminal_plan)
                    try:
                        self._execute_terminal(terminal_plan)
                    except TerminalStepError as exc:
                        return self._failure(
                            exc.stage,
                            str(exc),
                            feedback="arm_control_failed",
                            target=target,
                            plan=plan,
                        )
                    try:
                        self._execute_transport_pose(terminal_plan)
                    except TerminalStepError as exc:
                        return self._failure(
                            exc.stage,
                            str(exc),
                            feedback="arm_control_failed",
                            target=target,
                            plan=plan,
                            object_held=True,
                        )
                    return GraspResult(
                        True,
                        "DONE",
                        feedback="target_in_grasp_window",
                        object_held=True,
                        target=dict(target),
                        plan=plan,
                    )
                target, failure = self._detect_ready_target_after_motion(plan)
                if failure:
                    return failure
                next_alignment_step = self._alignment_step(
                    target,
                    allow_final_pose_move=not final_pose_move_used,
                    allow_closure=not self._target_ready_for_square_face_tracking(target),
                )
                if executed_alignment_step.get("action") == "jog":
                    previous_error = self._alignment_error_value(
                        target_before_motion,
                        executed_alignment_step,
                    )
                    current_error = self._alignment_error_value(
                        target,
                        executed_alignment_step,
                    )
                    if previous_error is not None and current_error is not None:
                        minimum_progress = previous_error * ALIGNMENT_MIN_PROGRESS_RATIO
                        if current_error <= previous_error - minimum_progress:
                            stagnant_steps = 0
                        else:
                            stagnant_steps += 1

                    same_joint = (
                        next_alignment_step.get("action") == "jog"
                        and next_alignment_step.get("joint") == executed_alignment_step.get("joint")
                    )
                    reversed_direction = (
                        same_joint
                        and float(next_alignment_step.get("delta_deg", 0.0))
                        * float(executed_alignment_step.get("delta_deg", 0.0))
                        < 0.0
                    )
                    direction_reversals = direction_reversals + 1 if reversed_direction else 0
                    if direction_reversals >= ALIGNMENT_MAX_DIRECTION_REVERSALS:
                        joint = str(executed_alignment_step.get("joint", "unknown"))
                        guard_feedback = "arm_control_failed"
                        for candidate_step in (executed_alignment_step, next_alignment_step):
                            if str(candidate_step.get("feedback", "")) in {
                                "square_face_too_far",
                                "square_face_too_near",
                            }:
                                guard_feedback = self._alignment_exhausted_feedback(
                                    target,
                                    candidate_step,
                                )
                                break
                        guard_reason = f"alignment oscillation detected on {joint} joint"
                        if guard_feedback in {"target_too_far", "target_too_near"}:
                            guard_reason = (
                                f"{guard_reason}: size alignment conflict ({guard_feedback})"
                            )
                        return self._alignment_guard_failure(
                            plan,
                            target,
                            guard="oscillation",
                            reason=guard_reason,
                            details={
                                "feedback": guard_feedback,
                                "joint": joint,
                                "direction_reversals": direction_reversals,
                            },
                        )
                    if stagnant_steps >= ALIGNMENT_MAX_STAGNANT_STEPS:
                        guard_feedback = "arm_control_failed"
                        for candidate_step in (executed_alignment_step, next_alignment_step):
                            if str(candidate_step.get("feedback", "")) in {
                                "square_face_too_far",
                                "square_face_too_near",
                            }:
                                guard_feedback = self._alignment_exhausted_feedback(
                                    target,
                                    candidate_step,
                                )
                                break
                        return self._alignment_guard_failure(
                            plan,
                            target,
                            guard="no_progress",
                            reason="alignment visual error did not improve",
                            details={
                                "feedback": guard_feedback,
                                "stagnant_steps": stagnant_steps,
                                "previous_error": previous_error,
                                "current_error": current_error,
                            },
                        )
                else:
                    stagnant_steps = 0
                    direction_reversals = 0

                alignment_step = next_alignment_step
                alignment_step["max_align_steps"] = self.max_align_steps
                alignment_preamble = self._alignment_preamble(plan, alignment_step)
                alignment_postamble = self._alignment_postamble(alignment_step)
                plan.extend(alignment_preamble)
                plan.append(alignment_step)
                plan.extend(alignment_postamble)
        except Exception as exc:
            return self._failure("ARM_CONTROL", str(exc), feedback="arm_control_failed", target=target, plan=plan)

        return self._failure("VISUAL_ALIGN", "unreachable alignment state", feedback="arm_control_failed", target=target, plan=plan)


class StaticVision:
    def __init__(self, target: Optional[Mapping[str, Any]] = None):
        self.target = target or {
            "track_id": 0,
            "color": "red",
            "center_px": [640.0, 520.0],
            "angle_deg": -90.0,
            "angle_reliable": True,
            "size_px": [124.0, 73.0],
            "area_px": 8200.0,
            "confidence": 1.0,
            "stable_frames": 1,
            "stable": True,
            "grasp_candidate": True,
        }

    def detect(self) -> Mapping[str, Any]:
        return self.target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conservative red-strip grasp state machine")
    parser.add_argument("--device", help="camera device, for example /dev/video0")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--single-step", action="store_true")
    parser.add_argument("--stop-after-final-pose", action="store_true")
    parser.add_argument("--static-target", action="store_true", help="use built-in target instead of camera")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--frames-per-detect", type=int, default=5)
    parser.add_argument("--max-align-steps", type=int, default=5)
    parser.add_argument("--max-jog-deg", type=float, default=1.0)
    parser.add_argument("--spd", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--acc", type=float, default=DEFAULT_ACCELERATION)
    parser.add_argument("--json-result", action="store_true")
    return parser


def _build_cli_vision(args: argparse.Namespace):
    if args.static_target or not args.device:
        return StaticVision(), None
    vision = open_strip_camera_vision(
        device=args.device,
        config_path=args.config,
        calibration_path=args.calibration,
        width=args.width,
        height=args.height,
        fps=args.fps,
        frames_per_detect=args.frames_per_detect,
    )
    return vision, vision


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    reference = load_grasp_reference(args.reference) if args.reference else default_grasp_reference()
    vision, closeable = _build_cli_vision(args)
    try:
        machine = ArmGraspStateMachine(
            vision=vision,
            motion=None,
            reference=reference,
            dry_run=True,
            single_step=args.single_step,
            stop_after_final_pose=args.stop_after_final_pose,
            max_align_steps=args.max_align_steps,
            max_jog_deg=args.max_jog_deg,
            spd=args.spd,
            acc=args.acc,
        )
        result = machine.run().to_dict()
    finally:
        if closeable is not None:
            closeable.close()
    if args.json_result:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    else:
        print(result["stage"], result["reason"], result.get("feedback", ""))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
