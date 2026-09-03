import copy
import json
import math
import numbers
from dataclasses import dataclass, replace
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np


REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "camera",
    "colors",
    "morphology",
    "geometry",
    "tracker",
    "roi",
    "output",
}


_SPLIT_CORE_THRESHOLD_RATIO = 0.75
_SPLIT_MIN_CORE_AREA_PX = 16
_SPLIT_MIN_CORE_AREA_FRACTION_OF_MIN_AREA = 0.1
_SPLIT_MIN_FILL_RATIO = 0.50
_SPLIT_PIXEL_OPERATION_BUDGET = 1_000_000


@dataclass(frozen=True)
class StripCandidate:
    color: str
    center_px: tuple
    angle_deg: float
    long_side_px: float
    short_side_px: float
    area_px: float
    fill_ratio: float
    solidity: float
    confidence: float
    box: tuple


@dataclass(frozen=True)
class TrackedStrip:
    track_id: int
    color: str
    center_px: tuple
    angle_deg: float
    angle_unwrapped_deg: float
    size_px: tuple
    area_px: float
    confidence: float
    stable_frames: int
    stable: bool
    grasp_candidate: bool
    box: tuple
    angle_reliable: bool = True


@dataclass
class _TrackRecord:
    strip: TrackedStrip
    missed_frames: int = 0


def _require_fields(mapping, fields, name):
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be an object")
    missing = set(fields) - set(mapping)
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(sorted(missing))}")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_finite_number(value, name, minimum=None, maximum=None):
    if not _is_number(value) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def _require_integer(value, name, minimum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _validate_hsv_triplet(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three integers")
    limits = (179, 255, 255)
    for index, (channel, maximum) in enumerate(zip(value, limits)):
        if (
            not isinstance(channel, int)
            or isinstance(channel, bool)
            or not 0 <= channel <= maximum
        ):
            raise ValueError(f"{name}[{index}] must be in the valid HSV range")


def _validate_colors(colors):
    _require_fields(colors, ("red", "green"), "colors")
    if set(colors) != {"red", "green"}:
        raise ValueError("colors must contain exactly red and green")
    for color in ("red", "green"):
        ranges = colors[color]
        if not isinstance(ranges, list) or not ranges:
            raise ValueError(f"colors.{color} must contain at least one range")
        for index, hsv_range in enumerate(ranges):
            name = f"colors.{color}[{index}]"
            _require_fields(hsv_range, ("lower", "upper"), name)
            _validate_hsv_triplet(hsv_range["lower"], f"{name}.lower")
            _validate_hsv_triplet(hsv_range["upper"], f"{name}.upper")
            if any(
                lower > upper
                for lower, upper in zip(hsv_range["lower"], hsv_range["upper"])
            ):
                raise ValueError(f"{name}.lower must not exceed upper")


def _validate_camera(camera):
    fields = ("device", "width", "height", "fps", "fourcc", "opencv_threads")
    _require_fields(camera, fields, "camera")
    device = camera["device"]
    valid_device = (
        (isinstance(device, str) and bool(device))
        or (
            isinstance(device, int)
            and not isinstance(device, bool)
            and device >= 0
        )
    )
    if not valid_device:
        raise ValueError(
            "camera.device must be a non-empty string or non-negative integer"
        )
    for field in ("width", "height", "fps", "opencv_threads"):
        _require_integer(camera[field], f"camera.{field}", minimum=1)
    if not isinstance(camera["fourcc"], str) or len(camera["fourcc"]) != 4:
        raise ValueError("camera.fourcc must be a four-character string")


def _validate_morphology(morphology):
    fields = (
        "open_kernel",
        "close_kernel",
        "open_iterations",
        "close_iterations",
    )
    _require_fields(morphology, fields, "morphology")
    for field in ("open_kernel", "close_kernel"):
        value = morphology[field]
        _require_integer(value, f"morphology.{field}", minimum=1)
        if value % 2 == 0:
            raise ValueError(f"morphology.{field} must be odd")
    for field in ("open_iterations", "close_iterations"):
        _require_integer(morphology[field], f"morphology.{field}", minimum=0)


def _validate_geometry(geometry):
    fields = (
        "min_area_px",
        "max_area_ratio",
        "min_long_side_px",
        "min_short_side_px",
        "min_aspect_ratio",
        "max_aspect_ratio",
        "ideal_aspect_ratio",
        "min_fill_ratio",
        "min_solidity",
        "border_margin_px",
        "min_confidence",
    )
    _require_fields(geometry, fields, "geometry")
    for field in ("min_area_px", "min_long_side_px", "min_short_side_px"):
        _require_finite_number(geometry[field], f"geometry.{field}", minimum=0)
        if geometry[field] == 0:
            raise ValueError(f"geometry.{field} must be positive")
    for field in ("max_area_ratio", "min_fill_ratio", "min_solidity"):
        _require_finite_number(
            geometry[field], f"geometry.{field}", minimum=0, maximum=1
        )
        if geometry[field] == 0:
            raise ValueError(f"geometry.{field} must be positive")
    _require_finite_number(
        geometry["min_confidence"],
        "geometry.min_confidence",
        minimum=0,
        maximum=1,
    )
    _require_integer(
        geometry["border_margin_px"], "geometry.border_margin_px", minimum=0
    )
    for field in (
        "min_aspect_ratio",
        "ideal_aspect_ratio",
        "max_aspect_ratio",
    ):
        _require_finite_number(geometry[field], f"geometry.{field}", minimum=0)
        if geometry[field] == 0:
            raise ValueError(f"geometry.{field} must be positive")
    if not (
        geometry["min_aspect_ratio"]
        <= geometry["ideal_aspect_ratio"]
        <= geometry["max_aspect_ratio"]
    ):
        raise ValueError("geometry aspect ratios must be ordered min <= ideal <= max")


def _validate_tracker(tracker):
    fields = (
        "max_center_distance_px",
        "max_area_change_ratio",
        "max_missed_frames",
        "stable_frames",
        "angle_alpha",
        "min_orientation_aspect_ratio",
    )
    _require_fields(tracker, fields, "tracker")
    _require_finite_number(
        tracker["max_center_distance_px"],
        "tracker.max_center_distance_px",
        minimum=0,
    )
    if tracker["max_center_distance_px"] == 0:
        raise ValueError("tracker.max_center_distance_px must be positive")
    _require_finite_number(
        tracker["max_area_change_ratio"],
        "tracker.max_area_change_ratio",
        minimum=0,
        maximum=1,
    )
    _require_integer(
        tracker["max_missed_frames"], "tracker.max_missed_frames", minimum=0
    )
    _require_integer(tracker["stable_frames"], "tracker.stable_frames", minimum=1)
    _require_finite_number(
        tracker["angle_alpha"], "tracker.angle_alpha", minimum=0, maximum=1
    )
    if tracker["angle_alpha"] == 0:
        raise ValueError("tracker.angle_alpha must be positive")
    _require_finite_number(
        tracker["min_orientation_aspect_ratio"],
        "tracker.min_orientation_aspect_ratio",
        minimum=1,
    )


def _validate_roi(roi):
    if roi is None:
        return
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError("roi must be null or four finite numbers")
    for index, value in enumerate(roi):
        _require_finite_number(value, f"roi[{index}]")
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        raise ValueError("roi must satisfy x2 > x1 and y2 > y1")


def _validate_output(output):
    fields = (
        "frame_id",
        "print_interval_seconds",
        "metrics_interval_seconds",
        "debug_directory",
    )
    _require_fields(output, fields, "output")
    for field in ("frame_id", "debug_directory"):
        if not isinstance(output[field], str) or not output[field]:
            raise ValueError(f"output.{field} must be a non-empty string")
    for field in ("print_interval_seconds", "metrics_interval_seconds"):
        _require_finite_number(output[field], f"output.{field}", minimum=0)
        if output[field] == 0:
            raise ValueError(f"output.{field} must be positive")


def validate_config(config):
    _require_fields(config, REQUIRED_TOP_LEVEL_FIELDS, "config")
    schema_version = config["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ValueError("schema_version must be the integer 1")
    _validate_camera(config["camera"])
    _validate_colors(config["colors"])
    _validate_morphology(config["morphology"])
    _validate_geometry(config["geometry"])
    _validate_tracker(config["tracker"])
    _validate_roi(config["roi"])
    _validate_output(config["output"])


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    validate_config(config)
    return config


def clean_mask(mask, morphology):
    result = mask
    operations = (
        ("open_kernel", "open_iterations", cv2.MORPH_OPEN),
        ("close_kernel", "close_iterations", cv2.MORPH_CLOSE),
    )
    for kernel_field, iterations_field, operation in operations:
        iterations = morphology[iterations_field]
        if iterations:
            size = morphology[kernel_field]
            kernel = np.ones((size, size), dtype=np.uint8)
            result = cv2.morphologyEx(
                result, operation, kernel, iterations=iterations
            )
    return result


def _roi_bounds(roi, width, height):
    x1, y1, x2, y2 = roi
    # Four values in [0, 1] are normalized; every other valid ROI uses pixels.
    if all(0 <= value <= 1 for value in roi):
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    left = max(0, min(width, math.floor(x1)))
    top = max(0, min(height, math.floor(y1)))
    right = max(0, min(width, math.ceil(x2)))
    bottom = max(0, min(height, math.ceil(y2)))
    return left, top, right, bottom


def _validate_frame(frame_bgr):
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.size == 0
        or frame_bgr.dtype != np.uint8
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
    ):
        raise ValueError(
            "frame_bgr must be a non-empty uint8 ndarray with shape (H, W, 3)"
        )


def _is_public_real(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_public_integer(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _validate_public_real(value, name, *, minimum=None, maximum=None, positive=False):
    if not _is_public_real(value) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if positive and numeric <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return numeric


def _validate_public_integer(value, name, *, minimum=None):
    if not _is_public_integer(value):
        raise ValueError(f"{name} must be an integer")
    numeric = int(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return numeric


def _validate_color_name(color, name):
    if not isinstance(color, str) or color not in {"red", "green"}:
        raise ValueError(f"{name} must be 'red' or 'green'")
    return color


def _validate_point_pair(point, name):
    if not isinstance(point, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must contain two numbers")
    if len(point) != 2:
        raise ValueError(f"{name} must contain two numbers")
    x, y = point
    return (
        _validate_public_real(x, f"{name}[0]"),
        _validate_public_real(y, f"{name}[1]"),
    )


def _validate_box_points(box, name):
    if not isinstance(box, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} must contain four 2D points")
    if len(box) != 4:
        raise ValueError(f"{name} must contain four 2D points")
    normalized = []
    for index, point in enumerate(box):
        normalized.append(_validate_point_pair(point, f"{name}[{index}]"))
    return tuple(normalized)


def _validate_image_size(image_size):
    if not isinstance(image_size, (list, tuple, np.ndarray)):
        raise ValueError("image_size must contain two integers")
    if len(image_size) != 2:
        raise ValueError("image_size must contain two integers")
    width = _validate_public_integer(image_size[0], "image_size[0]", minimum=1)
    height = _validate_public_integer(image_size[1], "image_size[1]", minimum=1)
    return width, height


def _validate_public_candidate(candidate):
    _validate_color_name(candidate.color, "candidate.color")
    _validate_point_pair(candidate.center_px, "candidate.center_px")
    _validate_public_real(candidate.angle_deg, "candidate.angle_deg")
    _validate_public_real(candidate.long_side_px, "candidate.long_side_px", positive=True)
    _validate_public_real(candidate.short_side_px, "candidate.short_side_px", positive=True)
    _validate_public_real(candidate.area_px, "candidate.area_px", positive=True)
    _validate_public_real(
        candidate.fill_ratio, "candidate.fill_ratio", minimum=0.0, maximum=1.0
    )
    _validate_public_real(
        candidate.solidity, "candidate.solidity", minimum=0.0, maximum=1.0
    )
    _validate_public_real(
        candidate.confidence, "candidate.confidence", minimum=0.0, maximum=1.0
    )
    _validate_box_points(candidate.box, "candidate.box")


def _validate_public_tracked_strip(tracked_strip):
    _validate_public_integer(tracked_strip.track_id, "tracked_strip.track_id", minimum=1)
    _validate_color_name(tracked_strip.color, "tracked_strip.color")
    _validate_point_pair(tracked_strip.center_px, "tracked_strip.center_px")
    _validate_public_real(tracked_strip.angle_deg, "tracked_strip.angle_deg")
    _validate_public_real(
        tracked_strip.angle_unwrapped_deg, "tracked_strip.angle_unwrapped_deg"
    )
    _validate_point_pair(tracked_strip.size_px, "tracked_strip.size_px")
    if tracked_strip.size_px[0] <= 0 or tracked_strip.size_px[1] <= 0:
        raise ValueError("tracked_strip.size_px must be positive")
    _validate_public_real(tracked_strip.area_px, "tracked_strip.area_px", positive=True)
    _validate_public_real(
        tracked_strip.confidence, "tracked_strip.confidence", minimum=0.0, maximum=1.0
    )
    _validate_public_integer(
        tracked_strip.stable_frames, "tracked_strip.stable_frames", minimum=0
    )
    if not isinstance(tracked_strip.stable, bool):
        raise ValueError("tracked_strip.stable must be a boolean")
    if not isinstance(tracked_strip.grasp_candidate, bool):
        raise ValueError("tracked_strip.grasp_candidate must be a boolean")
    if not isinstance(tracked_strip.angle_reliable, bool):
        raise ValueError("tracked_strip.angle_reliable must be a boolean")
    _validate_box_points(tracked_strip.box, "tracked_strip.box")


def build_color_masks(frame_bgr, config):
    validate_config(config)
    _validate_frame(frame_bgr)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    masks = {}
    for color, ranges in config["colors"].items():
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for hsv_range in ranges:
            lower = np.asarray(hsv_range["lower"], dtype=np.uint8)
            upper = np.asarray(hsv_range["upper"], dtype=np.uint8)
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))
        masks[color] = clean_mask(combined, config["morphology"])

    if config["roi"] is not None:
        height, width = hsv.shape[:2]
        left, top, right, bottom = _roi_bounds(config["roi"], width, height)
        roi_mask = np.zeros((height, width), dtype=np.uint8)
        roi_mask[top:bottom, left:right] = 255
        for color in masks:
            masks[color] = cv2.bitwise_and(masks[color], roi_mask)
    return masks


def normalize_axis_angle(angle):
    return (float(angle) + 90.0) % 180.0 - 90.0


def axial_angle_difference(a, b):
    return abs(normalize_axis_angle(float(a) - float(b)))


def unwrap_axis_angle(angle, reference=None):
    base = normalize_axis_angle(angle)
    if reference is None:
        return base
    reference = float(reference)
    wrapped = base + 180.0 * round((reference - base) / 180.0)
    return float(wrapped)


def normalize_min_area_rect(rect):
    center, (width, height), cv_angle = rect
    width = float(width)
    height = float(height)
    if width >= height:
        long_side = width
        short_side = height
        angle_deg = -float(cv_angle)
    else:
        long_side = height
        short_side = width
        angle_deg = -(float(cv_angle) + 90.0)
    normalized_center = (float(center[0]), float(center[1]))
    return (
        normalized_center,
        long_side,
        short_side,
        normalize_axis_angle(angle_deg),
    )


def _clamp_unit(value):
    return max(0.0, min(1.0, float(value)))


def _threshold_score(value, minimum, maximum=1.0):
    if maximum <= minimum:
        return 1.0
    return _clamp_unit((value - minimum) / (maximum - minimum))


def _aspect_score(aspect, ideal):
    if aspect <= 0 or ideal <= 0:
        return 0.0
    return _clamp_unit(min(aspect, ideal) / max(aspect, ideal))


def _boundary_score(box, frame_width, frame_height, margin):
    points = np.asarray(box, dtype=np.float32)
    clearance = min(
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(frame_width - 1 - points[:, 0].max()),
        float(frame_height - 1 - points[:, 1].max()),
    )
    return _clamp_unit((clearance - margin) / 20.0)


def _candidate_confidence(
    area,
    fill_ratio,
    solidity,
    aspect,
    box,
    frame_width,
    frame_height,
    geometry,
):
    fill_score = _threshold_score(
        fill_ratio, geometry["min_fill_ratio"]
    )
    solidity_score = _threshold_score(
        solidity, geometry["min_solidity"]
    )
    area_score = _threshold_score(
        area,
        geometry["min_area_px"],
        geometry["min_area_px"] * 4.0,
    )
    aspect_score = _aspect_score(
        aspect, geometry["ideal_aspect_ratio"]
    )
    boundary_score = _boundary_score(
        box,
        frame_width,
        frame_height,
        geometry["border_margin_px"],
    )
    confidence = (
        0.25 * fill_score
        + 0.20 * solidity_score
        + 0.20 * area_score
        + 0.25 * aspect_score
        + 0.10 * boundary_score
    )
    return _clamp_unit(confidence)


def _touches_border(box, bounds, margin):
    points = np.asarray(box, dtype=np.float32)
    left, top, right, bottom = bounds
    min_y_limit = top + margin
    if top <= 0:
        min_y_limit = top - margin
    return bool(
        points[:, 0].min() <= left + margin
        or points[:, 1].min() <= min_y_limit
        or points[:, 0].max() >= right - 1 - margin
        or points[:, 1].max() >= bottom - 1 - margin
    )


def _candidate_rect(candidate):
    if hasattr(candidate, "long_side_px") and hasattr(candidate, "short_side_px"):
        size = (candidate.long_side_px, candidate.short_side_px)
    else:
        size = candidate.size_px
    return (
        candidate.center_px,
        size,
        -candidate.angle_deg,
    )


def _rotated_box_iou(first, second):
    _, intersection = cv2.rotatedRectangleIntersection(
        _candidate_rect(first), _candidate_rect(second)
    )
    if intersection is None:
        return 0.0
    intersection_area = abs(float(cv2.contourArea(intersection)))
    if intersection_area <= 0:
        return 0.0
    first_size = first.size_px if hasattr(first, "size_px") else (
        first.long_side_px,
        first.short_side_px,
    )
    second_size = second.size_px if hasattr(second, "size_px") else (
        second.long_side_px,
        second.short_side_px,
    )
    first_area = float(first_size[0] * first_size[1])
    second_area = float(second_size[0] * second_size[1])
    union_area = first_area + second_area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def _overlap_over_smaller_box(first, second):
    _, intersection = cv2.rotatedRectangleIntersection(
        _candidate_rect(first), _candidate_rect(second)
    )
    if intersection is None:
        return 0.0
    intersection_area = abs(float(cv2.contourArea(intersection)))
    first_size = first.size_px if hasattr(first, "size_px") else (
        first.long_side_px,
        first.short_side_px,
    )
    second_size = second.size_px if hasattr(second, "size_px") else (
        second.long_side_px,
        second.short_side_px,
    )
    smaller_area = min(
        first_size[0] * first_size[1],
        second_size[0] * second_size[1],
    )
    if smaller_area <= 0:
        return 0.0
    return intersection_area / smaller_area


def _deduplicate_candidates(candidates):
    kept = []
    ordered = sorted(
        candidates,
        key=lambda candidate: candidate.confidence,
        reverse=True,
    )
    for candidate in ordered:
        duplicate = any(
            candidate.color == existing.color
            and _overlap_over_smaller_box(candidate, existing) >= 0.65
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def _unpack_find_contours_result(result):
    item_count = len(result)
    if item_count == 2:
        return result[0]
    if item_count == 3:
        return result[1]
    raise RuntimeError(
        f"cv2.findContours returned {item_count} items; expected 2 or 3"
    )


def _candidate_from_contour(
    contour,
    *,
    color,
    frame_width,
    frame_height,
    frame_area,
    geometry,
    detection_bounds,
):
    area = float(cv2.contourArea(contour))
    if (
        area < geometry["min_area_px"]
        or area > geometry["max_area_ratio"] * frame_area
    ):
        return None

    rect = cv2.minAreaRect(contour)
    center, long_side, short_side, angle_deg = normalize_min_area_rect(rect)
    box_array = cv2.boxPoints(rect)
    box = tuple(
        (float(point[0]), float(point[1])) for point in box_array
    )
    box_area = long_side * short_side
    if box_area <= 0 or short_side <= 0:
        return None

    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    fill_ratio = area / box_area
    solidity = area / hull_area if hull_area > 0 else 0.0
    aspect = long_side / short_side
    if (
        long_side < geometry["min_long_side_px"]
        or short_side < geometry["min_short_side_px"]
        or aspect < geometry["min_aspect_ratio"]
        or aspect > geometry["max_aspect_ratio"]
        or fill_ratio < geometry["min_fill_ratio"]
        or solidity < geometry["min_solidity"]
        or _touches_border(
            box,
            detection_bounds,
            geometry["border_margin_px"],
        )
    ):
        return None

    confidence = _candidate_confidence(
        area,
        fill_ratio,
        solidity,
        aspect,
        box,
        frame_width,
        frame_height,
        geometry,
    )
    if confidence < geometry["min_confidence"]:
        return None
    return StripCandidate(
        color=color,
        center_px=center,
        angle_deg=angle_deg,
        long_side_px=long_side,
        short_side_px=short_side,
        area_px=area,
        fill_ratio=fill_ratio,
        solidity=solidity,
        confidence=confidence,
        box=box,
    )


def _contour_local_mask(contour):
    x, y, width, height = cv2.boundingRect(contour)
    origin_x = x - 1
    origin_y = y - 1
    local_contour = contour - np.array(
        [[[origin_x, origin_y]]], dtype=contour.dtype
    )
    mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.drawContours(mask, [local_contour], -1, 255, thickness=-1)
    return mask, origin_x, origin_y


def _significant_core_components(core_mask, max_components, min_area_px):
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(core_mask)
    components = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= _SPLIT_MIN_CORE_AREA_PX:
            components.append((label, area))
    if len(components) < 2:
        return labels, []

    minimum_area = max(
        _SPLIT_MIN_CORE_AREA_PX,
        int(
            math.ceil(
                float(min_area_px)
                * _SPLIT_MIN_CORE_AREA_FRACTION_OF_MIN_AREA
            )
        ),
    )
    filtered = [
        (label, area)
        for label, area in components
        if area >= minimum_area
    ]
    filtered.sort(key=lambda item: item[1], reverse=True)
    if len(filtered) > max_components:
        return labels, []
    return labels, filtered


def _split_contour_candidates(
    contour,
    *,
    color,
    frame_width,
    frame_height,
    frame_area,
    geometry,
    detection_bounds,
):
    contour_area = float(cv2.contourArea(contour))
    if contour_area > geometry["max_area_ratio"] * frame_area:
        return None
    max_components = int(contour_area // geometry["min_area_px"])
    if max_components < 2:
        return None
    local_mask, offset_x, offset_y = _contour_local_mask(contour)
    distance = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if not math.isfinite(max_distance) or max_distance <= 0:
        return None

    core_mask = np.where(
        distance >= max_distance * _SPLIT_CORE_THRESHOLD_RATIO,
        255,
        0,
    ).astype(np.uint8)
    labels, significant = _significant_core_components(
        core_mask,
        max_components,
        geometry["min_area_px"],
    )
    if len(significant) < 2:
        return None
    if local_mask.size * len(significant) > _SPLIT_PIXEL_OPERATION_BUDGET:
        return None

    foreground_pixels = np.column_stack(np.where(local_mask > 0))
    if foreground_pixels.size == 0:
        return None

    foreground_mask = local_mask > 0
    best_distance = np.full(local_mask.shape, np.inf, dtype=np.float32)
    assignments = np.full(local_mask.shape, -1, dtype=np.int16)
    for component_index, (label, _) in enumerate(significant):
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        distance_input = np.where(component_mask > 0, 0, 255).astype(np.uint8)
        distance_map = cv2.distanceTransform(distance_input, cv2.DIST_L2, 5)
        better = foreground_mask & (distance_map < best_distance)
        best_distance[better] = distance_map[better]
        assignments[better] = component_index

    if np.any(assignments[foreground_mask] < 0):
        return []

    split_geometry = dict(geometry)
    split_geometry["min_fill_ratio"] = min(
        float(geometry["min_fill_ratio"]),
        _SPLIT_MIN_FILL_RATIO,
    )
    valid_candidates = []
    for component_index in range(len(significant)):
        assigned_mask = np.zeros_like(local_mask)
        assigned_pixels = foreground_pixels[
            assignments[foreground_pixels[:, 0], foreground_pixels[:, 1]]
            == component_index
        ]
        if len(assigned_pixels) == 0:
            continue
        assigned_mask[
            assigned_pixels[:, 0],
            assigned_pixels[:, 1],
        ] = 255
        contours = _unpack_find_contours_result(
            cv2.findContours(
                assigned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        )
        if not contours:
            continue
        local_contour = max(contours, key=cv2.contourArea)
        global_contour = local_contour + np.array(
            [[[offset_x, offset_y]]], dtype=local_contour.dtype
        )
        candidate = _candidate_from_contour(
            global_contour,
            color=color,
            frame_width=frame_width,
            frame_height=frame_height,
            frame_area=frame_area,
            geometry=split_geometry,
            detection_bounds=detection_bounds,
        )
        if candidate is not None:
            valid_candidates.append(candidate)

    return valid_candidates


def detect_candidates(frame_bgr, config):
    masks = build_color_masks(frame_bgr, config)
    frame_height, frame_width = frame_bgr.shape[:2]
    frame_area = float(frame_height * frame_width)
    geometry = config["geometry"]
    detection_bounds = (
        _roi_bounds(config["roi"], frame_width, frame_height)
        if config["roi"] is not None
        else (0, 0, frame_width, frame_height)
    )
    candidates = []

    for color, mask in masks.items():
        contours = _unpack_find_contours_result(
            cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
        )
        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            if contour_area > geometry["max_area_ratio"] * frame_area:
                continue
            original_candidate = _candidate_from_contour(
                contour,
                color=color,
                frame_width=frame_width,
                frame_height=frame_height,
                frame_area=frame_area,
                geometry=geometry,
                detection_bounds=detection_bounds,
            )
            split_candidates = _split_contour_candidates(
                contour,
                color=color,
                frame_width=frame_width,
                frame_height=frame_height,
                frame_area=frame_area,
                geometry=geometry,
                detection_bounds=detection_bounds,
            )
            if split_candidates is None:
                if original_candidate is not None:
                    candidates.append(original_candidate)
                continue

            if len(split_candidates) >= 1:
                candidates.extend(split_candidates)
                continue
            if original_candidate is not None:
                candidates.append(original_candidate)

    candidates.sort(
        key=lambda candidate: (
            -candidate.confidence,
            candidate.color,
            candidate.center_px[0],
            candidate.center_px[1],
            candidate.angle_deg,
        )
    )
    return _deduplicate_candidates(candidates), masks


def _as_float_pair(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two numbers")
    x, y = value
    if not _is_number(x) or not _is_number(y):
        raise ValueError(f"{name} must contain two numbers")
    return float(x), float(y)


def _candidate_sort_key(candidate):
    return (
        candidate.color,
        float(candidate.center_px[0]),
        float(candidate.center_px[1]),
        float(candidate.area_px),
        float(candidate.angle_deg),
        float(candidate.confidence),
        tuple(float(value) for point in candidate.box for value in point),
    )


def _pair_score(track, candidate, tracker_config):
    if track.color != candidate.color:
        return None
    max_center_distance = float(tracker_config["max_center_distance_px"])
    center_distance = math.hypot(
        float(candidate.center_px[0]) - float(track.center_px[0]),
        float(candidate.center_px[1]) - float(track.center_px[1]),
    )
    if center_distance > max_center_distance:
        return None
    track_area = float(track.area_px)
    candidate_area = float(candidate.area_px)
    area_denominator = max(track_area, candidate_area)
    if area_denominator <= 0:
        return None
    area_change_ratio = abs(candidate_area - track_area) / area_denominator
    if area_change_ratio > float(tracker_config["max_area_change_ratio"]):
        return None
    box_iou = _rotated_box_iou(track, candidate)
    center_score = 1.0 - center_distance / max_center_distance
    if float(tracker_config["max_area_change_ratio"]) > 0:
        area_score = 1.0 - area_change_ratio / float(
            tracker_config["max_area_change_ratio"]
        )
    else:
        area_score = 1.0 if area_change_ratio == 0 else 0.0
    candidate_aspect = float(candidate.long_side_px) / float(
        candidate.short_side_px
    )
    candidate_angle_reliable = candidate_aspect >= float(
        tracker_config["min_orientation_aspect_ratio"]
    )
    if track.angle_reliable and candidate_angle_reliable:
        angle_difference = axial_angle_difference(
            candidate.angle_deg, track.angle_unwrapped_deg
        )
        angle_score = 1.0 - min(angle_difference, 90.0) / 90.0
        return (
            0.42 * center_score
            + 0.22 * area_score
            + 0.18 * angle_score
            + 0.18 * box_iou
        )
    return 0.51 * center_score + 0.27 * area_score + 0.22 * box_iou


def _best_assignment(tracks, candidates, tracker_config):
    if not tracks or not candidates:
        return tuple(len(candidates) for _ in tracks)

    unmatched_index = len(candidates)
    candidate_keys = [_candidate_sort_key(candidate) for candidate in candidates]
    edges_by_track = []
    for track in tracks:
        edges = []
        for candidate_index, candidate in enumerate(candidates):
            score = _pair_score(track, candidate, tracker_config)
            if score is not None:
                edges.append((candidate_index, float(score)))
        edges.sort(
            key=lambda item: (
                -item[1],
                candidate_keys[item[0]],
                item[0],
            )
        )
        edges_by_track.append(edges)

    track_order = sorted(
        range(len(tracks)),
        key=lambda index: (len(edges_by_track[index]), tracks[index].track_id),
    )
    matched_track_to_candidate = [unmatched_index] * len(tracks)
    matched_candidate_to_track = {}

    def augment(track_index, seen_candidates):
        for candidate_index, _score in edges_by_track[track_index]:
            if candidate_index in seen_candidates:
                continue
            seen_candidates.add(candidate_index)
            current_track_index = matched_candidate_to_track.get(candidate_index)
            if current_track_index is None or augment(
                current_track_index, seen_candidates
            ):
                matched_candidate_to_track[candidate_index] = track_index
                matched_track_to_candidate[track_index] = candidate_index
                return True
        return False

    for track_index in track_order:
        augment(track_index, set())

    return tuple(matched_track_to_candidate)


def _validate_candidates(candidates):
    if candidates is None or isinstance(candidates, (str, bytes, dict)):
        raise ValueError("candidates must be an iterable of StripCandidate")
    if not isinstance(candidates, Iterable):
        raise ValueError("candidates must be an iterable of StripCandidate")
    validated = list(candidates)
    for candidate in validated:
        if not isinstance(candidate, StripCandidate):
            raise ValueError("candidates must contain StripCandidate instances")
        _validate_public_candidate(candidate)
    return validated


def _validate_tracked_strips(detections):
    if detections is None or isinstance(detections, (str, bytes, dict)):
        raise ValueError("detections must be an iterable of TrackedStrip")
    if not isinstance(detections, Iterable):
        raise ValueError("detections must be an iterable of TrackedStrip")
    validated = list(detections)
    for detection in validated:
        if not isinstance(detection, TrackedStrip):
            raise ValueError("detections must contain TrackedStrip instances")
        _validate_public_tracked_strip(detection)
    return validated


def _tracked_strip_from_candidate(
    track_id,
    candidate,
    stable_frames,
    stable,
    angle_unwrapped,
    angle_reliable,
):
    return TrackedStrip(
        track_id=int(track_id),
        color=candidate.color,
        center_px=(
            float(candidate.center_px[0]),
            float(candidate.center_px[1]),
        ),
        angle_deg=normalize_axis_angle(angle_unwrapped),
        angle_unwrapped_deg=float(angle_unwrapped),
        size_px=(
            float(candidate.long_side_px),
            float(candidate.short_side_px),
        ),
        area_px=float(candidate.area_px),
        confidence=float(candidate.confidence),
        stable_frames=int(stable_frames),
        stable=bool(stable),
        grasp_candidate=bool(stable and candidate.color == "red"),
        box=tuple(
            (float(point[0]), float(point[1])) for point in candidate.box
        ),
        angle_reliable=bool(angle_reliable),
    )


def tracked_strip_to_dict(tracked_strip):
    if not isinstance(tracked_strip, TrackedStrip):
        raise ValueError("tracked_strip must be a TrackedStrip")
    _validate_public_tracked_strip(tracked_strip)
    return {
        "track_id": int(tracked_strip.track_id),
        "color": tracked_strip.color,
        "center_px": [float(tracked_strip.center_px[0]), float(tracked_strip.center_px[1])],
        "angle_deg": float(tracked_strip.angle_deg),
        "angle_unwrapped_deg": float(tracked_strip.angle_unwrapped_deg),
        "angle_reliable": bool(tracked_strip.angle_reliable),
        "size_px": [float(tracked_strip.size_px[0]), float(tracked_strip.size_px[1])],
        "area_px": float(tracked_strip.area_px),
        "confidence": float(tracked_strip.confidence),
        "stable_frames": int(tracked_strip.stable_frames),
        "stable": bool(tracked_strip.stable),
        "grasp_candidate": bool(tracked_strip.grasp_candidate),
    }


def build_frame_payload(
    *,
    timestamp_ns,
    frame_id,
    frame_seq,
    image_size,
    detections,
):
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("frame_id must be a non-empty string")
    _validate_public_integer(timestamp_ns, "timestamp_ns", minimum=0)
    _validate_public_integer(frame_seq, "frame_seq", minimum=0)
    width, height = _validate_image_size(image_size)
    validated_detections = _validate_tracked_strips(detections)
    return {
        "schema_version": 1,
        "timestamp_ns": int(timestamp_ns),
        "frame_id": frame_id,
        "frame_seq": int(frame_seq),
        "image_size": [int(width), int(height)],
        "detections": [tracked_strip_to_dict(strip) for strip in validated_detections],
    }


class StripTracker:
    def __init__(self, config):
        validate_config(config)
        self._config = copy.deepcopy(config)
        self._records = []
        self._next_track_id = 1

    def _update_matched_record(self, record, candidate):
        tracker_config = self._config["tracker"]
        if record.missed_frames > 0:
            stable_frames = 1
        else:
            stable_frames = record.strip.stable_frames + 1
        reference_angle = record.strip.angle_unwrapped_deg
        candidate_aspect = float(candidate.long_side_px) / float(
            candidate.short_side_px
        )
        candidate_angle_reliable = candidate_aspect >= float(
            tracker_config["min_orientation_aspect_ratio"]
        )
        if candidate_angle_reliable:
            if record.strip.angle_reliable:
                measured_angle = unwrap_axis_angle(
                    candidate.angle_deg,
                    reference_angle,
                )
                angle_unwrapped = (
                    (1.0 - float(tracker_config["angle_alpha"]))
                    * float(reference_angle)
                    + float(tracker_config["angle_alpha"])
                    * float(measured_angle)
                )
            else:
                angle_unwrapped = normalize_axis_angle(candidate.angle_deg)
            angle_reliable = True
        else:
            angle_unwrapped = float(reference_angle)
            angle_reliable = False
        stable = stable_frames >= int(tracker_config["stable_frames"])
        record.strip = _tracked_strip_from_candidate(
            record.strip.track_id,
            candidate,
            stable_frames,
            stable,
            angle_unwrapped,
            angle_reliable,
        )
        record.missed_frames = 0

    def _mark_missed_record(self, record):
        record.missed_frames += 1
        record.strip = replace(
            record.strip,
            stable_frames=0,
            stable=False,
            grasp_candidate=False,
        )

    def _create_record(self, candidate):
        stable_frames = 1
        stable = stable_frames >= int(self._config["tracker"]["stable_frames"])
        candidate_aspect = float(candidate.long_side_px) / float(
            candidate.short_side_px
        )
        angle_reliable = candidate_aspect >= float(
            self._config["tracker"]["min_orientation_aspect_ratio"]
        )
        strip = _tracked_strip_from_candidate(
            self._next_track_id,
            candidate,
            stable_frames,
            stable,
            normalize_axis_angle(candidate.angle_deg),
            angle_reliable,
        )
        self._next_track_id += 1
        return _TrackRecord(strip=strip, missed_frames=0)

    def update(self, candidates):
        validated_candidates = _validate_candidates(candidates)
        active_records = [record for record in self._records]

        by_color_tracks = {}
        by_color_candidates = {}
        for record in active_records:
            by_color_tracks.setdefault(record.strip.color, []).append(record)
        for candidate in validated_candidates:
            by_color_candidates.setdefault(candidate.color, []).append(candidate)

        matched_record_ids = set()
        matched_candidate_ids = set()

        for color in sorted(set(by_color_tracks) | set(by_color_candidates)):
            records = sorted(
                by_color_tracks.get(color, []),
                key=lambda record: record.strip.track_id,
            )
            color_candidates = sorted(
                by_color_candidates.get(color, []),
                key=_candidate_sort_key,
            )
            assignment = _best_assignment(
                [record.strip for record in records],
                color_candidates,
                self._config["tracker"],
            )
            for record, candidate_index in zip(records, assignment):
                if candidate_index == len(color_candidates):
                    continue
                candidate = color_candidates[candidate_index]
                self._update_matched_record(record, candidate)
                matched_record_ids.add(record.strip.track_id)
                matched_candidate_ids.add(id(candidate))

        for record in self._records:
            if record.strip.track_id not in matched_record_ids:
                self._mark_missed_record(record)

        self._records = [
            record
            for record in self._records
            if record.missed_frames <= self._config["tracker"]["max_missed_frames"]
        ]

        for candidate in validated_candidates:
            if id(candidate) not in matched_candidate_ids:
                self._records.append(self._create_record(candidate))

        visible = [
            replace(record.strip)
            for record in self._records
            if record.missed_frames == 0
        ]
        visible.sort(key=lambda strip: strip.track_id)
        return visible
