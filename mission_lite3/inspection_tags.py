from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .wide_camera import load_wide_calibration


@dataclass(frozen=True)
class InspectionTagObservation:
    tag_id: int
    center_x_px: float
    center_y_px: float
    edge_px: float
    distance_m: Optional[float]
    corners: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class InspectionTagTarget:
    tag_id: int
    center_x_px: float
    center_y_px: float
    edge_px: float


@dataclass(frozen=True)
class InspectionTagCorrection:
    kind: str
    reason: str
    distance_m: float = 0.0
    center_error_px: float = 0.0
    edge_error_px: float = 0.0


def _resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent.parent / path


def _camera_focal_x_px(config: Mapping[str, Any]) -> Optional[float]:
    camera = config.get("camera")
    inspection = config.get("inspection")
    if not isinstance(camera, Mapping) or not isinstance(inspection, Mapping):
        return None
    calibration_path = camera.get("wide_calibration")
    if not isinstance(calibration_path, str) or not calibration_path.strip():
        return None
    try:
        calibration = load_wide_calibration(_resolve_project_path(calibration_path))
        matrix_key = (
            "new_camera_matrix"
            if bool(inspection.get("use_wide_undistortion", False))
            else "camera_matrix"
        )
        focal_x = float(calibration[matrix_key][0][0])
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return focal_x if math.isfinite(focal_x) and focal_x > 0.0 else None


def station_tag_target(
    tag_config: Mapping[str, Any],
    stop_name: str,
) -> Optional[InspectionTagTarget]:
    station_ids = tag_config.get("station_tag_ids")
    targets = tag_config.get("targets")
    if not isinstance(station_ids, Mapping) or not isinstance(targets, Mapping):
        return None
    raw_id = station_ids.get(stop_name)
    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
        return None
    raw_target = targets.get(str(raw_id), targets.get(raw_id))
    if not isinstance(raw_target, Mapping):
        return None
    try:
        return InspectionTagTarget(
            tag_id=int(raw_id),
            center_x_px=float(raw_target["center_x_px"]),
            center_y_px=float(raw_target["center_y_px"]),
            edge_px=float(raw_target["edge_px"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


class InspectionTagDetector:
    def __init__(self, config: Mapping[str, Any]) -> None:
        inspection = config.get("inspection")
        tag_config = (
            inspection.get("tag_localization", {})
            if isinstance(inspection, Mapping)
            else {}
        )
        self.config = dict(tag_config) if isinstance(tag_config, Mapping) else {}
        self.enabled = bool(self.config.get("enabled", False))
        self.tag_size_m = float(self.config.get("tag_size_m", 0.08))
        self.min_edge_px = float(self.config.get("min_edge_px", 24.0))
        self.mask_margin_px = int(self.config.get("mask_margin_px", 4))
        self.mask_for_recognition = bool(
            self.config.get("mask_for_recognition", True)
        )
        self.focal_x_px = _camera_focal_x_px(config)
        self.available = False
        self.unavailable_reason = "disabled"
        self._cv2 = None
        self._detector = None
        self._dictionary = None
        self._parameters = None
        if self.enabled:
            self._initialize_backend()

    def _initialize_backend(self) -> None:
        try:
            import cv2

            aruco = cv2.aruco
            family_name = str(
                self.config.get("family", "DICT_APRILTAG_36h11")
            )
            if not family_name.startswith("DICT_"):
                family_name = f"DICT_APRILTAG_{family_name}"
            family_value = getattr(aruco, family_name)
            dictionary = aruco.getPredefinedDictionary(family_value)
            parameters = aruco.DetectorParameters()
            parameters.markerBorderBits = int(
                self.config.get("marker_border_bits", 2)
            )
            if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
                parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            detector = (
                aruco.ArucoDetector(dictionary, parameters)
                if hasattr(aruco, "ArucoDetector")
                else None
            )
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            self.unavailable_reason = f"backend_unavailable:{exc}"
            return
        self._cv2 = cv2
        self._dictionary = dictionary
        self._parameters = parameters
        self._detector = detector
        self.available = True
        self.unavailable_reason = ""

    def detect(self, frame: Any) -> list[InspectionTagObservation]:
        if not self.available or frame is None or not hasattr(frame, "shape"):
            return []
        assert self._cv2 is not None
        gray = (
            frame
            if len(frame.shape) == 2
            else self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        )
        if self._detector is not None:
            corners, ids, _rejected = self._detector.detectMarkers(gray)
        else:
            corners, ids, _rejected = self._cv2.aruco.detectMarkers(
                gray,
                self._dictionary,
                parameters=self._parameters,
            )
        if ids is None:
            return []
        observations: list[InspectionTagObservation] = []
        for raw_corners, raw_id in zip(corners, ids.reshape(-1)):
            points = np.asarray(raw_corners, dtype=np.float64).reshape(4, 2)
            edges = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
            edge_px = float(np.mean(edges))
            if not math.isfinite(edge_px) or edge_px < self.min_edge_px:
                continue
            center = np.mean(points, axis=0)
            distance_m = None
            if self.focal_x_px is not None:
                distance_m = self.focal_x_px * self.tag_size_m / edge_px
            observations.append(
                InspectionTagObservation(
                    tag_id=int(raw_id),
                    center_x_px=float(center[0]),
                    center_y_px=float(center[1]),
                    edge_px=edge_px,
                    distance_m=distance_m,
                    corners=tuple(
                        (float(point[0]), float(point[1])) for point in points
                    ),
                )
            )
        observations.sort(key=lambda item: (-item.edge_px, item.tag_id))
        return observations

    def mask(self, frame: Any, observations: Sequence[InspectionTagObservation]):
        if frame is None or not observations:
            return frame
        assert self._cv2 is not None
        output = frame.copy()
        height, width = output.shape[:2]
        margin = max(0, self.mask_margin_px)
        for observation in observations:
            points = np.asarray(observation.corners, dtype=np.float64)
            x0 = max(0, int(math.floor(float(np.min(points[:, 0])))) - margin)
            y0 = max(0, int(math.floor(float(np.min(points[:, 1])))) - margin)
            x1 = min(
                width - 1,
                int(math.ceil(float(np.max(points[:, 0])))) + margin,
            )
            y1 = min(
                height - 1,
                int(math.ceil(float(np.max(points[:, 1])))) + margin,
            )
            self._cv2.rectangle(output, (x0, y0), (x1, y1), (255, 255, 255), -1)
        return output

    @staticmethod
    def recommended_letter_min_height_px(
        observations: Sequence[InspectionTagObservation],
    ) -> Optional[int]:
        if not observations:
            return None
        edge_px = max(item.edge_px for item in observations)
        return max(18, min(24, int(round(edge_px * 0.48))))


def median_tag_observation(
    observations: Iterable[InspectionTagObservation],
) -> Optional[InspectionTagObservation]:
    values = list(observations)
    if not values:
        return None
    tag_ids = {item.tag_id for item in values}
    if len(tag_ids) != 1:
        raise ValueError("median tag observation requires one tag id")
    ordered = sorted(values, key=lambda item: item.center_x_px)
    representative = ordered[len(ordered) // 2]
    distances = [item.distance_m for item in values if item.distance_m is not None]
    return InspectionTagObservation(
        tag_id=representative.tag_id,
        center_x_px=float(median(item.center_x_px for item in values)),
        center_y_px=float(median(item.center_y_px for item in values)),
        edge_px=float(median(item.edge_px for item in values)),
        distance_m=float(median(distances)) if distances else None,
        corners=representative.corners,
    )


def plan_inspection_tag_correction(
    observation: InspectionTagObservation,
    target: InspectionTagTarget,
    config: Mapping[str, Any],
    *,
    focal_x_px: Optional[float],
) -> InspectionTagCorrection:
    if observation.tag_id != target.tag_id:
        return InspectionTagCorrection("fail", "unexpected_tag_id")
    center_error = observation.center_x_px - target.center_x_px
    edge_error = observation.edge_px - target.edge_px
    center_tolerance = float(config.get("center_tolerance_px", 12.0))
    edge_tolerance = float(config.get("edge_tolerance_px", 2.0))
    if abs(center_error) <= center_tolerance and abs(edge_error) <= edge_tolerance:
        return InspectionTagCorrection(
            "complete",
            "within_tolerance",
            center_error_px=center_error,
            edge_error_px=edge_error,
        )
    tag_size_m = float(config.get("tag_size_m", 0.08))
    if abs(center_error) > center_tolerance:
        if focal_x_px is None or focal_x_px <= 0.0:
            return InspectionTagCorrection(
                "fail",
                "focal_length_unavailable",
                center_error_px=center_error,
                edge_error_px=edge_error,
            )
        current_distance = focal_x_px * tag_size_m / observation.edge_px
        sign = int(config.get("positive_error_strafe_sign", -1))
        requested = sign * center_error * current_distance / focal_x_px
        limit = float(config.get("max_strafe_step_m", 0.10))
        requested = math.copysign(min(abs(requested), limit), requested)
        return InspectionTagCorrection(
            "strafe",
            "horizontal_center_error",
            distance_m=requested,
            center_error_px=center_error,
            edge_error_px=edge_error,
        )
    if (
        abs(edge_error) > edge_tolerance
        and focal_x_px is not None
        and focal_x_px > 0.0
    ):
        current_distance = focal_x_px * tag_size_m / observation.edge_px
        target_distance = focal_x_px * tag_size_m / target.edge_px
        requested = current_distance - target_distance
        limit = float(config.get("max_forward_step_m", 0.10))
        requested = math.copysign(min(abs(requested), limit), requested)
        return InspectionTagCorrection(
            "forward",
            "edge_size_error",
            distance_m=requested,
            center_error_px=center_error,
            edge_error_px=edge_error,
        )
    return InspectionTagCorrection(
        "complete",
        "center_within_tolerance",
        center_error_px=center_error,
        edge_error_px=edge_error,
    )
