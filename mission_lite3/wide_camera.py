from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


CARDBOARD_HSV_LOWER = (8, 18, 70)
CARDBOARD_HSV_UPPER = (38, 140, 255)
CARDBOARD_LAB_B_RANGE = (132, 180)


@dataclass(frozen=True)
class BoxParallelResult:
    ok: bool
    reason: str
    top_angle_deg: float | None = None
    seam_angle_deg: float | None = None
    parallel_error_deg: float | None = None
    confidence: float = 0.0
    box_x_range: tuple[int, int] | None = None
    top_line: tuple[int, int, int, int] | None = None
    seam_line: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_wide_calibration(path: str | Path) -> dict[str, Any]:
    calibration_path = Path(path)
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != 1:
        raise ValueError("unsupported wide-camera calibration schema")
    if data.get("model") != "pinhole":
        raise ValueError("wide-camera calibration model must be pinhole")
    image_size = data.get("image_size")
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(int(value) <= 0 for value in image_size)
    ):
        raise ValueError("wide-camera calibration image_size is invalid")
    camera_matrix = np.asarray(data.get("camera_matrix"), dtype=np.float64)
    new_camera_matrix = np.asarray(
        data.get("new_camera_matrix"),
        dtype=np.float64,
    )
    distortion = np.asarray(
        data.get("distortion_coefficients"),
        dtype=np.float64,
    ).reshape(-1)
    if camera_matrix.shape != (3, 3) or new_camera_matrix.shape != (3, 3):
        raise ValueError("wide-camera calibration matrices must be 3x3")
    if distortion.size not in {4, 5, 8, 12, 14}:
        raise ValueError("wide-camera distortion coefficient count is unsupported")
    if not (
        np.isfinite(camera_matrix).all()
        and np.isfinite(new_camera_matrix).all()
        and np.isfinite(distortion).all()
    ):
        raise ValueError("wide-camera calibration contains non-finite values")
    if not bool(data.get("validated_for_undistortion", False)):
        raise ValueError("wide-camera calibration is not validated for undistortion")
    data["image_size"] = [int(image_size[0]), int(image_size[1])]
    return data


class WideCameraUndistorter:
    def __init__(self, calibration: Mapping[str, Any]) -> None:
        self.calibration = dict(calibration)
        width, height = self.calibration["image_size"]
        self.image_size = (int(width), int(height))
        camera_matrix = np.asarray(
            self.calibration["camera_matrix"],
            dtype=np.float64,
        )
        new_camera_matrix = np.asarray(
            self.calibration["new_camera_matrix"],
            dtype=np.float64,
        )
        distortion = np.asarray(
            self.calibration["distortion_coefficients"],
            dtype=np.float64,
        )
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            None,
            new_camera_matrix,
            self.image_size,
            cv2.CV_32FC1,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "WideCameraUndistorter":
        return cls(load_wide_calibration(path))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.ndim < 2:
            raise ValueError("wide-camera frame is invalid")
        actual_size = (int(frame.shape[1]), int(frame.shape[0]))
        if actual_size != self.image_size:
            raise ValueError(
                "wide-camera frame size mismatch: "
                f"calibrated={self.image_size} actual={actual_size}"
            )
        return cv2.remap(
            frame,
            self.map_x,
            self.map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )


def cardboard_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv_mask = cv2.inRange(
        hsv,
        np.asarray(CARDBOARD_HSV_LOWER, dtype=np.uint8),
        np.asarray(CARDBOARD_HSV_UPPER, dtype=np.uint8),
    )
    lab_mask = cv2.inRange(
        lab[:, :, 2],
        CARDBOARD_LAB_B_RANGE[0],
        CARDBOARD_LAB_B_RANGE[1],
    )
    return cv2.bitwise_and(hsv_mask, lab_mask)


def _fit_line(points: np.ndarray) -> tuple[float, float, float, float]:
    vx, vy, x0, y0 = cv2.fitLine(
        points.astype(np.float32),
        cv2.DIST_HUBER,
        0,
        0.01,
        0.01,
    ).reshape(-1)
    return float(vx), float(vy), float(x0), float(y0)


def _line_y(vx: float, vy: float, x0: float, y0: float, x: float) -> float:
    if abs(vx) < 1e-9:
        return y0
    return y0 + (x - x0) * vy / vx


def _largest_contiguous_x_run(
    points: list[tuple[int, int]],
    *,
    max_gap: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    ordered = sorted(points, key=lambda point: point[0])
    runs: list[list[tuple[int, int]]] = [[ordered[0]]]
    for point in ordered[1:]:
        if point[0] - runs[-1][-1][0] <= max_gap:
            runs[-1].append(point)
        else:
            runs.append([point])
    return max(
        runs,
        key=lambda run: (run[-1][0] - run[0][0], len(run)),
    )


def _select_widest_line_cluster(
    candidates: list[tuple[float, float, tuple[int, int, int, int], float]],
    *,
    max_reference_y_gap: float,
) -> list[tuple[float, float, tuple[int, int, int, int], float]]:
    if not candidates:
        return []

    clusters = [
        [
            candidate
            for candidate in candidates
            if abs(candidate[3] - anchor[3]) <= max_reference_y_gap
        ]
        for anchor in candidates
    ]

    def score(cluster):
        x_values = [
            x
            for _length, _angle, (x1, _y1, x2, _y2), _reference_y in cluster
            for x in (x1, x2)
        ]
        span = max(x_values) - min(x_values)
        total_length = sum(candidate[0] for candidate in cluster)
        return span, total_length

    return max(clusters, key=score)


def detect_box_parallel(frame: np.ndarray) -> BoxParallelResult:
    if frame is None or frame.ndim != 3:
        return BoxParallelResult(False, "invalid_frame")
    height, width = frame.shape[:2]
    mask = cardboard_mask(frame)
    boundary_points: list[tuple[int, int]] = []
    search_y0 = int(round(0.45 * height))
    search_y1 = int(round(0.72 * height))
    occupancy_depth = max(40, int(round(0.19 * height)))
    for x in range(int(round(0.10 * width)), int(round(0.90 * width))):
        candidates = np.flatnonzero(mask[search_y0:search_y1, x])
        if candidates.size == 0:
            continue
        y = search_y0 + int(candidates[0])
        bottom = min(height, y + occupancy_depth)
        occupancy = float(np.count_nonzero(mask[y:bottom, x])) / max(1, bottom - y)
        if occupancy >= 0.65:
            boundary_points.append((x, y))
    if len(boundary_points) < max(120, int(0.12 * width)):
        return BoxParallelResult(False, "cardboard_top_not_found")

    # Warm-coloured furniture behind the box can create a second short run at
    # almost the same height.  Including that disconnected run stretches the
    # top line and lets the seam search escape from the cardboard face.  Keep
    # the widest continuous cardboard run before fitting either line.
    boundary_points = _largest_contiguous_x_run(
        boundary_points,
        max_gap=max(8, int(round(0.008 * width))),
    )
    if len(boundary_points) < max(120, int(0.12 * width)):
        return BoxParallelResult(False, "cardboard_contiguous_span_too_small")

    boundary = np.asarray(boundary_points, dtype=np.float64)
    median_y = float(np.median(boundary[:, 1]))
    boundary = boundary[np.abs(boundary[:, 1] - median_y) < 45.0]
    if len(boundary) < max(100, int(0.10 * width)):
        return BoxParallelResult(False, "cardboard_top_inconsistent")
    vx, vy, x0, y0 = _fit_line(boundary)
    top_angle = math.degrees(math.atan2(vy, vx))
    x_min = int(round(float(np.min(boundary[:, 0]))))
    x_max = int(round(float(np.max(boundary[:, 0]))))
    box_span = x_max - x_min
    if box_span < int(0.25 * width):
        return BoxParallelResult(False, "cardboard_span_too_small")
    top_y_min = int(round(_line_y(vx, vy, x0, y0, x_min)))
    top_y_max = int(round(_line_y(vx, vy, x0, y0, x_max)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    region = np.zeros_like(edges)
    seam_roi_y0 = int(round(median_y + 0.14 * height))
    # At the final pre-grasp distance the front face can fill the lower half
    # of the image and the horizontal cardboard seam sits very close to the
    # bottom edge.  The previous 0.42-height limit cut that real seam out of
    # the Hough ROI entirely (for example, y~=675 in a 720 px frame with the
    # top at y~=324).  Keep the upper bound tied to the detected box top, but
    # allow the search to reach the bottom for close boxes.  The horizontal
    # angle and 75%-of-box-span checks below still reject short background
    # edges.
    seam_roi_y1 = min(height, int(round(median_y + 0.58 * height)))
    # The box face expands laterally below its top edge.  At the final
    # approach distance the seam can begin near the image edge even though
    # the detected top starts well inside it, so a fixed 20 px pad clips a
    # large part of the real line.
    seam_x_pad = max(20, int(round(0.12 * width)))
    seam_roi_x0 = max(0, x_min - seam_x_pad)
    seam_roi_x1 = min(width - 1, x_max + seam_x_pad)
    cv2.rectangle(
        region,
        (seam_roi_x0, max(0, seam_roi_y0)),
        (seam_roi_x1, min(height - 1, seam_roi_y1)),
        255,
        -1,
    )
    edges = cv2.bitwise_and(edges, region)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=40,
        # Close-up cardboard seams are often interrupted by tape and glare.
        # Accept shorter Hough fragments here and enforce coverage only after
        # compatible fragments have been projected and merged below.
        minLineLength=max(80, int(0.12 * box_span)),
        maxLineGap=45,
    )
    candidates: list[tuple[float, float, tuple[int, int, int, int], float]] = []
    box_center_x = 0.5 * float(x_min + x_max)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) <= 12.0 and length >= 0.12 * box_span:
                # Compare fragments at a common x coordinate.  Comparing
                # their raw centre-y values splits one genuinely collinear
                # seam when it is a few degrees off horizontal: fragments
                # from opposite sides of a wide close-up box can be tens of
                # pixels apart vertically even though they belong to the
                # same line.
                reference_y = 0.5 * (float(y1) + float(y2))
                if abs(dx) > 1e-9:
                    reference_y += (
                        box_center_x - 0.5 * float(x1 + x2)
                    ) * dy / dx
                candidates.append(
                    (
                        length,
                        angle,
                        (int(x1), int(y1), int(x2), int(y2)),
                        reference_y,
                    )
                )
    if not candidates:
        return BoxParallelResult(
            False,
            "cardboard_seam_not_found",
            top_angle_deg=top_angle,
            box_x_range=(x_min, x_max),
            top_line=(x_min, top_y_min, x_max, top_y_max),
        )

    cluster = _select_widest_line_cluster(
        candidates,
        max_reference_y_gap=14.0,
    )
    seam_points: list[tuple[float, float]] = []
    for length, _angle, (x1, y1, x2, y2), _center_y in cluster:
        sample_count = max(2, int(round(length / 18.0)))
        for ratio in np.linspace(0.0, 1.0, sample_count):
            seam_points.append(
                (
                    float(x1) + ratio * float(x2 - x1),
                    float(y1) + ratio * float(y2 - y1),
                )
            )
    seam_array = np.asarray(seam_points, dtype=np.float64)
    seam_vx, seam_vy, seam_x0, seam_y0 = _fit_line(seam_array)
    seam_angle = math.degrees(math.atan2(seam_vy, seam_vx))
    seam_x_min = int(round(float(np.min(seam_array[:, 0]))))
    seam_x_max = int(round(float(np.max(seam_array[:, 0]))))
    seam_span = seam_x_max - seam_x_min
    seam_line = (
        seam_x_min,
        int(round(_line_y(seam_vx, seam_vy, seam_x0, seam_y0, seam_x_min))),
        seam_x_max,
        int(round(_line_y(seam_vx, seam_vy, seam_x0, seam_y0, seam_x_max))),
    )
    # Measure coverage against the part of the fitted line that can actually
    # be visible inside the search ROI.  A sloped seam close to the camera can
    # hit the bottom of the image long before it reaches the far side of the
    # box; comparing only with the top-edge span incorrectly rejects that
    # complete, boundary-clipped observation.  Retain an absolute half-box
    # requirement so a short incidental edge near the image boundary cannot
    # validate itself merely by extrapolating out of frame.
    visible_x0 = float(seam_roi_x0)
    visible_x1 = float(seam_roi_x1)
    slope = seam_vy / seam_vx if abs(seam_vx) >= 1e-9 else math.inf
    if math.isfinite(slope) and abs(slope) >= 1e-9:
        intercept = seam_y0 - slope * seam_x0
        y_low = float(max(0, seam_roi_y0))
        y_high = float(min(height - 1, seam_roi_y1))
        at_low = (y_low - intercept) / slope
        at_high = (y_high - intercept) / slope
        visible_x0 = max(visible_x0, min(at_low, at_high))
        visible_x1 = min(visible_x1, max(at_low, at_high))
    visible_span = min(float(box_span), max(0.0, visible_x1 - visible_x0))
    required_span = max(0.50 * box_span, 0.75 * visible_span)
    if seam_span < required_span:
        return BoxParallelResult(
            False,
            "cardboard_seam_span_too_small",
            top_angle_deg=top_angle,
            seam_angle_deg=seam_angle,
            box_x_range=(x_min, x_max),
            top_line=(x_min, top_y_min, x_max, top_y_max),
            seam_line=seam_line,
        )
    parallel_error = seam_angle - top_angle
    top_coverage = min(1.0, box_span / (0.40 * width))
    seam_coverage = min(
        1.0,
        seam_span / (0.30 * width),
    )
    confidence = min(top_coverage, seam_coverage)
    return BoxParallelResult(
        True,
        "",
        top_angle_deg=top_angle,
        seam_angle_deg=seam_angle,
        parallel_error_deg=parallel_error,
        confidence=confidence,
        box_x_range=(x_min, x_max),
        top_line=(x_min, top_y_min, x_max, top_y_max),
        seam_line=seam_line,
    )


def annotate_box_parallel(
    frame: np.ndarray,
    result: BoxParallelResult,
) -> np.ndarray:
    output = frame.copy()
    if result.top_line is not None:
        x1, y1, x2, y2 = result.top_line
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 3)
    if result.seam_line is not None:
        x1, y1, x2, y2 = result.seam_line
        cv2.line(output, (x1, y1), (x2, y2), (255, 255, 0), 3)
    text = (
        f"parallel_error={result.parallel_error_deg:+.2f}deg "
        f"confidence={result.confidence:.2f}"
        if result.ok and result.parallel_error_deg is not None
        else f"box parallel unavailable: {result.reason}"
    )
    cv2.putText(
        output,
        text,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output
