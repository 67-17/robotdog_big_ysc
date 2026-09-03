import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


DEFAULT_CENTER_TOLERANCE_PX = (90.0, 90.0)
DEFAULT_SIZE_RATIO_TOLERANCE = 0.45
DEFAULT_AREA_RATIO_TOLERANCE = 0.65
DEFAULT_ANGLE_TOLERANCE_DEG = 15.0
DEFAULT_MIN_AREA_PX = 5000.0
DEFAULT_MIN_VISIBLE_LONG_SIDE_RATIO = 0.0
DEFAULT_MAX_VISIBLE_DEPTH_RATIO = float("inf")
DEFAULT_MAX_LONG_OVER_SHORT_RATIO = float("inf")
DEFAULT_MIN_LONG_OVER_SHORT_RATIO = 0.0
DEFAULT_MIN_RED_AREA_RATIO = 0.0
DEFAULT_MIN_FAR_EDGE_RATIO = 0.0


def _as_pair(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        scalar = float(value)
        return scalar, scalar
    return default


def _feature_pair(feature: Mapping[str, Any], key: str) -> Tuple[float, float]:
    value = feature.get(key, [0.0, 0.0])
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"feature {key} must contain two numbers")
    return float(value[0]), float(value[1])


def _left_edge_x(feature: Mapping[str, Any]) -> float:
    bbox = feature.get("bbox_px")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return float(bbox[0])
    points = feature.get("quad_points_px")
    if isinstance(points, (list, tuple)):
        xs = [
            float(point[0])
            for point in points
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if xs:
            return min(xs)
    center = _feature_pair(feature, "center_px")
    width, _height = _feature_pair(feature, "size_px")
    return center[0] - abs(width) / 2.0


def _long_side(feature: Mapping[str, Any]) -> float:
    width, height = _feature_pair(feature, "size_px")
    return max(abs(width), abs(height))


def _short_side(feature: Mapping[str, Any]) -> float:
    width, height = _feature_pair(feature, "size_px")
    return min(abs(width), abs(height))


def _bbox_aspect(feature: Mapping[str, Any]) -> float:
    short_side = _short_side(feature)
    if short_side <= 0.0:
        return 1.0
    return _long_side(feature) / short_side


def _positive_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _far_edge(feature: Mapping[str, Any]) -> Optional[float]:
    return _positive_float(feature.get("far_edge_px"))


def _has_far_edge_depth_ratio(feature: Mapping[str, Any]) -> bool:
    return _far_edge_over_depth(feature, fallback_to_bbox=False) is not None


def _far_edge_over_depth(
    feature: Mapping[str, Any],
    *,
    fallback_to_bbox: bool = True,
) -> Optional[float]:
    explicit = _positive_float(feature.get("far_edge_over_depth"))
    if explicit is not None:
        return explicit
    far_edge = _positive_float(feature.get("far_edge_px"))
    depth_edge = _positive_float(feature.get("visible_depth_edge_px"))
    if far_edge is not None and depth_edge is not None:
        return far_edge / depth_edge
    if fallback_to_bbox:
        return _bbox_aspect(feature)
    return None


def _aspect(feature: Mapping[str, Any]) -> float:
    value = _far_edge_over_depth(feature)
    return 1.0 if value is None else value


def _visible_depth(feature: Mapping[str, Any]) -> float:
    far_edge = _positive_float(feature.get("far_edge_px"))
    depth_edge = _positive_float(feature.get("visible_depth_edge_px"))
    if far_edge is not None and depth_edge is not None:
        return depth_edge / far_edge
    long_side = _long_side(feature)
    if long_side <= 0.0:
        return 0.0
    return _short_side(feature) / long_side


def _bbox_visible_depth(feature: Mapping[str, Any]) -> float:
    long_side = _long_side(feature)
    if long_side <= 0.0:
        return 0.0
    return _short_side(feature) / long_side


def _aspect_for_match(feature: Mapping[str, Any], use_far_edge_depth_ratio: bool) -> float:
    if use_far_edge_depth_ratio:
        value = _far_edge_over_depth(feature, fallback_to_bbox=False)
        if value is not None:
            return value
    return _bbox_aspect(feature)


def _visible_depth_for_match(
    feature: Mapping[str, Any],
    use_far_edge_depth_ratio: bool,
) -> float:
    if use_far_edge_depth_ratio:
        far_edge = _positive_float(feature.get("far_edge_px"))
        depth_edge = _positive_float(feature.get("visible_depth_edge_px"))
        if far_edge is not None and depth_edge is not None:
            return depth_edge / far_edge
    return _bbox_visible_depth(feature)


def _area(feature: Mapping[str, Any]) -> float:
    return float(feature.get("area_px", 0.0))


def _ratio(candidate: float, reference: float) -> float:
    if reference <= 0.0:
        return 1.0
    return candidate / reference


def _target_from_feature(feature: Mapping[str, Any]) -> Dict[str, Any]:
    center = _feature_pair(feature, "center_px")
    size = _feature_pair(feature, "size_px")
    target = {
        "track_id": int(feature.get("track_id", 0)),
        "color": "red",
        "center_px": [center[0], center[1]],
        "angle_deg": float(feature.get("angle_deg", 0.0)),
        "angle_reliable": bool(feature.get("angle_reliable", False)),
        "size_px": [size[0], size[1]],
        "area_px": float(feature.get("area_px", 0.0)),
        "confidence": float(feature.get("confidence", 1.0)),
        "stable_frames": int(feature.get("stable_frames", 1)),
        "stable": True,
        "grasp_candidate": True,
    }
    for key in (
        "bbox_px",
        "quad_points_px",
        "far_edge_px",
        "side_edge_px",
        "visible_depth_edge_px",
        "far_edge_over_depth",
    ):
        if key in feature:
            target[key] = feature[key]
    return target


def _distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def _ordered_quad_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(points) != 4:
        raise ValueError("quad must contain exactly four points")
    by_y = sorted(points, key=lambda item: (item[1], item[0]))
    top = sorted(by_y[:2], key=lambda item: item[0])
    bottom = sorted(by_y[2:], key=lambda item: item[0])
    top_left, top_right = top
    bottom_left, bottom_right = bottom
    return [top_left, top_right, bottom_right, bottom_left]


def _quad_edge_feature(points: List[Tuple[float, float]]) -> Dict[str, Any]:
    try:
        top_left, top_right, bottom_right, bottom_left = _ordered_quad_points(points)
    except ValueError:
        return {}
    far_edge = _distance(top_left, top_right)
    left_depth = _distance(top_left, bottom_left)
    right_depth = _distance(top_right, bottom_right)
    depth_edges = [value for value in (left_depth, right_depth) if value > 1.0]
    if far_edge <= 1.0 or not depth_edges:
        return {}
    visible_depth = min(depth_edges)
    ordered = [top_left, top_right, bottom_right, bottom_left]
    return {
        "quad_points_px": [[float(x), float(y)] for x, y in ordered],
        "far_edge_px": float(far_edge),
        "side_edge_px": [float(left_depth), float(right_depth)],
        "visible_depth_edge_px": float(visible_depth),
        "far_edge_over_depth": float(far_edge / visible_depth),
    }


def _valid_reference_features(reference_set: Mapping[str, Any]) -> List[Dict[str, Any]]:
    features = reference_set.get("features", [])
    if not isinstance(features, list):
        raise ValueError("reference feature set must contain a features list")
    valid = [
        dict(feature)
        for feature in features
        if isinstance(feature, Mapping) and not bool(feature.get("needs_review", False))
    ]
    if not valid:
        raise ValueError("reference feature set has no accepted features")
    return valid


def _nearest_reference_feature(
    features: List[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> Mapping[str, Any]:
    candidate_center = _feature_pair(candidate, "center_px")
    candidate_long_side = _long_side(candidate)

    def score(feature: Mapping[str, Any]) -> float:
        reference_center = _feature_pair(feature, "center_px")
        center_distance = math.hypot(
            candidate_center[0] - reference_center[0],
            candidate_center[1] - reference_center[1],
        )
        long_side_distance = abs(candidate_long_side - _long_side(feature))
        return center_distance + 0.25 * long_side_distance

    return min(features, key=score)


def _reference_feature_bounds(
    features: List[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    use_far_edge_depth_ratio: bool = True,
) -> Dict[str, float]:
    max_visible_depth_ratio = float(config.get("max_visible_depth_ratio", 1.0))
    max_long_over_short_ratio = float(config.get("max_long_over_short_ratio", 1.0))
    min_long_over_short_ratio = float(
        config.get(
            "min_long_over_short_ratio",
            DEFAULT_MIN_LONG_OVER_SHORT_RATIO,
        )
    )
    min_visible_long_side_ratio = float(config.get("min_visible_long_side_ratio", 0.0))
    min_red_area_ratio = float(config.get("min_red_area_ratio", DEFAULT_MIN_RED_AREA_RATIO))
    min_far_edge_ratio = float(
        config.get("min_far_edge_ratio", DEFAULT_MIN_FAR_EDGE_RATIO)
    )
    visible_depths = [
        _visible_depth_for_match(feature, use_far_edge_depth_ratio)
        for feature in features
    ]
    long_over_short_values = [
        _aspect_for_match(feature, use_far_edge_depth_ratio)
        for feature in features
    ]
    long_sides = [_long_side(feature) for feature in features]
    areas = [_area(feature) for feature in features]
    far_edges = [
        edge
        for edge in (_far_edge(feature) for feature in features)
        if edge is not None
    ]
    centers = [_feature_pair(feature, "center_px") for feature in features]
    left_edges = [_left_edge_x(feature) for feature in features]
    min_center_x = min(center[0] for center in centers)
    min_center_y = min(center[1] for center in centers)
    max_center_x = max(center[0] for center in centers)
    max_center_y = max(center[1] for center in centers)
    return {
        "max_visible_depth": max(visible_depths) * max_visible_depth_ratio,
        "max_long_over_short": max(long_over_short_values) * max_long_over_short_ratio,
        "min_long_over_short": min(long_over_short_values) * min_long_over_short_ratio,
        "min_long_side": min(long_sides) * min_visible_long_side_ratio,
        "min_area": min(areas) * min_red_area_ratio,
        "min_far_edge": (
            min(far_edges) * min_far_edge_ratio
            if far_edges and min_far_edge_ratio > 0.0
            else 0.0
        ),
        "min_center_x": min_center_x,
        "min_center_y": min_center_y,
        "max_center_x": max_center_x,
        "max_center_y": max_center_y,
        "min_left_edge_x": min(left_edges),
    }


def compare_final_view_features(
    reference: Mapping[str, Any],
    candidate: Optional[Mapping[str, Any]],
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    config = dict(config or {})
    if candidate is None:
        return {
            "ok": False,
            "feedback": "target_lost",
            "reason": "no red final-view component found",
            "metrics": {},
            "target": None,
        }

    center_tolerance = _as_pair(
        config.get("center_tolerance_px"),
        DEFAULT_CENTER_TOLERANCE_PX,
    )
    size_ratio_tolerance = float(
        config.get("size_ratio_tolerance", DEFAULT_SIZE_RATIO_TOLERANCE)
    )
    area_ratio_tolerance = float(
        config.get("area_ratio_tolerance", DEFAULT_AREA_RATIO_TOLERANCE)
    )
    angle_tolerance_deg = float(
        config.get("angle_tolerance_deg", DEFAULT_ANGLE_TOLERANCE_DEG)
    )

    ref_center = _feature_pair(reference, "center_px")
    cand_center = _feature_pair(candidate, "center_px")
    center_error = (cand_center[0] - ref_center[0], cand_center[1] - ref_center[1])

    long_ratio = _ratio(_long_side(candidate), _long_side(reference))
    short_ratio = _ratio(_short_side(candidate), _short_side(reference))
    area_ratio = _ratio(
        float(candidate.get("area_px", 0.0)),
        float(reference.get("area_px", 0.0)),
    )
    require_far_edge_depth_ratio = bool(config.get("require_far_edge_depth_ratio", False))
    use_far_edge_depth_ratio = (
        _has_far_edge_depth_ratio(reference) and _has_far_edge_depth_ratio(candidate)
    )
    candidate_aspect = _aspect_for_match(candidate, use_far_edge_depth_ratio)
    reference_aspect = _aspect_for_match(reference, use_far_edge_depth_ratio)
    aspect_ratio = _ratio(candidate_aspect, reference_aspect)
    candidate_bbox_aspect = _bbox_aspect(candidate)
    reference_bbox_aspect = _bbox_aspect(reference)
    candidate_visible_depth = _visible_depth_for_match(
        candidate,
        use_far_edge_depth_ratio,
    )
    reference_visible_depth = _visible_depth_for_match(
        reference,
        use_far_edge_depth_ratio,
    )
    visible_depth_ratio = _ratio(candidate_visible_depth, reference_visible_depth)
    reference_far_edge = _far_edge(reference)
    candidate_far_edge = _far_edge(candidate)
    size_ratio = max(long_ratio, short_ratio)
    if long_ratio < 1.0 and short_ratio < 1.0:
        size_ratio = min(long_ratio, short_ratio)

    angle_error = float(candidate.get("angle_deg", 0.0)) - float(reference.get("angle_deg", 0.0))
    while angle_error > 90.0:
        angle_error -= 180.0
    while angle_error < -90.0:
        angle_error += 180.0

    metrics = {
        "center_error_px": [center_error[0], center_error[1]],
        "size_ratio": size_ratio,
        "long_side_ratio": long_ratio,
        "short_side_ratio": short_ratio,
        "area_ratio": area_ratio,
        "candidate_aspect": candidate_aspect,
        "reference_aspect": reference_aspect,
        "aspect_ratio": aspect_ratio,
        "candidate_long_over_short": candidate_aspect,
        "reference_long_over_short": reference_aspect,
        "candidate_bbox_long_over_short": candidate_bbox_aspect,
        "reference_bbox_long_over_short": reference_bbox_aspect,
        "candidate_visible_depth": candidate_visible_depth,
        "reference_visible_depth": reference_visible_depth,
        "visible_depth_ratio": visible_depth_ratio,
        "angle_error_deg": angle_error,
    }
    target = _target_from_feature(candidate)
    allow_center_right_outside = bool(config.get("allow_center_right_outside", False))
    if allow_center_right_outside:
        reference_left_edge_x = _left_edge_x(reference)
        metrics["reference_left_edge_x"] = reference_left_edge_x
    if require_far_edge_depth_ratio and not use_far_edge_depth_ratio:
        return {
            "ok": False,
            "feedback": "target_lost",
            "reason": "final image red quadrilateral far-edge-over-depth is unavailable",
            "metrics": metrics,
            "target": target,
        }

    if allow_center_right_outside:
        if cand_center[0] < metrics["reference_left_edge_x"]:
            return {
                "ok": False,
                "feedback": "target_left",
                "reason": "final image center is left of reference red left edge",
                "metrics": metrics,
                "target": target,
            }
    elif abs(center_error[0]) > center_tolerance[0]:
        return {
            "ok": False,
            "feedback": "target_left" if center_error[0] < 0.0 else "target_right",
            "reason": "final image center differs from grasp sample",
            "metrics": metrics,
            "target": target,
        }
    if abs(center_error[1]) > center_tolerance[1]:
        return {
            "ok": False,
            "feedback": "arm_control_failed",
            "reason": "final image vertical center differs from grasp sample",
            "metrics": metrics,
            "target": target,
        }
    minimum_size_ratio = 1.0 - size_ratio_tolerance
    allow_less_visible_red = bool(config.get("allow_less_visible_red", False))
    min_visible_long_side_ratio = float(
        config.get(
            "min_visible_long_side_ratio",
            DEFAULT_MIN_VISIBLE_LONG_SIDE_RATIO,
        )
    )
    max_visible_depth_ratio = float(
        config.get(
            "max_visible_depth_ratio",
            DEFAULT_MAX_VISIBLE_DEPTH_RATIO,
        )
    )
    min_long_over_short_ratio = float(
        config.get(
            "min_long_over_short_ratio",
            DEFAULT_MIN_LONG_OVER_SHORT_RATIO,
        )
    )
    max_long_over_short_ratio = float(
        config.get(
            "max_long_over_short_ratio",
            DEFAULT_MAX_LONG_OVER_SHORT_RATIO,
        )
    )
    min_red_area_ratio = float(
        config.get(
            "min_red_area_ratio",
            DEFAULT_MIN_RED_AREA_RATIO,
        )
    )
    min_far_edge_ratio = float(
        config.get(
            "min_far_edge_ratio",
            DEFAULT_MIN_FAR_EDGE_RATIO,
        )
    )
    min_long_over_short = reference_aspect * min_long_over_short_ratio
    max_long_over_short = reference_aspect * max_long_over_short_ratio
    min_red_area = _area(reference) * min_red_area_ratio
    min_far_edge = (
        reference_far_edge * min_far_edge_ratio
        if reference_far_edge is not None and min_far_edge_ratio > 0.0
        else 0.0
    )
    too_small = (
        long_ratio < minimum_size_ratio
        or short_ratio < minimum_size_ratio
        or area_ratio < 1.0 - area_ratio_tolerance
    )
    visible_depth_too_large = visible_depth_ratio > max_visible_depth_ratio
    if min_long_over_short > 0.0 and candidate_aspect < min_long_over_short:
        metrics["reference_min_long_over_short"] = min_long_over_short
        return {
            "ok": False,
            "feedback": "target_too_far",
            "reason": "final image red far-edge-over-depth is below grasp sample",
            "metrics": metrics,
            "target": target,
        }
    if min_red_area > 0.0 and _area(candidate) < min_red_area:
        metrics["reference_min_area_px"] = min_red_area
        return {
            "ok": False,
            "feedback": "target_too_far",
            "reason": "final image red area is below grasp sample",
            "metrics": metrics,
            "target": target,
        }
    if min_far_edge > 0.0:
        metrics["reference_min_far_edge_px"] = min_far_edge
        if candidate_far_edge is None:
            return {
                "ok": False,
                "feedback": "target_lost",
                "reason": "final image red far edge is unavailable",
                "metrics": metrics,
                "target": target,
            }
        metrics["candidate_far_edge_px"] = candidate_far_edge
        if candidate_far_edge < min_far_edge:
            return {
                "ok": False,
                "feedback": "target_too_far",
                "reason": "final image red far edge is below grasp sample",
                "metrics": metrics,
                "target": target,
            }
    if math.isfinite(max_long_over_short) and candidate_aspect > max_long_over_short:
        metrics["reference_max_long_over_short"] = max_long_over_short
        return {
            "ok": False,
            "feedback": "target_too_near",
            "reason": "final image red far-edge-over-depth exceeds grasp sample",
            "metrics": metrics,
            "target": target,
        }
    if visible_depth_too_large:
        return {
            "ok": False,
            "feedback": "target_too_near",
            "reason": "final image red visible depth exceeds grasp sample",
            "metrics": metrics,
            "target": target,
        }
    if min_visible_long_side_ratio > 0.0 and long_ratio < min_visible_long_side_ratio:
        return {
            "ok": False,
            "feedback": "target_too_far",
            "reason": "final image red long side is too small compared with grasp sample",
            "metrics": metrics,
            "target": target,
        }
    if too_small and not allow_less_visible_red:
        return {
            "ok": False,
            "feedback": "target_too_far",
            "reason": "final image target is smaller than grasp sample",
            "metrics": metrics,
            "target": target,
        }
    if abs(angle_error) > angle_tolerance_deg:
        return {
            "ok": False,
            "feedback": "arm_control_failed",
            "reason": "final image angle differs from grasp sample",
            "metrics": metrics,
            "target": target,
        }
    return {
        "ok": True,
        "feedback": "target_in_grasp_window",
        "reason": "final image matches grasp sample",
        "metrics": metrics,
        "target": target,
    }


def compare_final_view_feature_set(
    reference_set: Mapping[str, Any],
    candidate: Optional[Mapping[str, Any]],
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    config = dict(config or {})
    if candidate is None:
        return {
            "ok": False,
            "feedback": "target_lost",
            "reason": "no red final-view component found",
            "metrics": {},
            "target": None,
        }

    features = _valid_reference_features(reference_set)
    reference = _nearest_reference_feature(features, candidate)
    require_far_edge_depth_ratio = bool(config.get("require_far_edge_depth_ratio", False))
    base_config = {
        key: value
        for key, value in config.items()
        if key
        not in {
            "center_tolerance_px",
            "max_visible_depth_ratio",
            "max_long_over_short_ratio",
            "min_visible_long_side_ratio",
            "min_red_area_ratio",
            "min_far_edge_ratio",
        }
    }
    base_config["center_tolerance_px"] = [1.0e9, 1.0e9]
    base_result = compare_final_view_features(reference, candidate, base_config)
    metrics = dict(base_result.get("metrics", {}))
    use_far_edge_depth_ratio = (
        all(_has_far_edge_depth_ratio(feature) for feature in features)
        and _has_far_edge_depth_ratio(candidate)
    )
    bounds = _reference_feature_bounds(
        features,
        config,
        use_far_edge_depth_ratio=use_far_edge_depth_ratio,
    )
    candidate_visible_depth = _visible_depth_for_match(
        candidate,
        use_far_edge_depth_ratio,
    )
    candidate_long_over_short = _aspect_for_match(candidate, use_far_edge_depth_ratio)
    candidate_bbox_long_over_short = _bbox_aspect(candidate)
    candidate_long_side = _long_side(candidate)
    candidate_area = _area(candidate)
    candidate_far_edge = _far_edge(candidate)
    candidate_center = _feature_pair(candidate, "center_px")
    center_tolerance = _as_pair(
        config.get("center_tolerance_px"),
        DEFAULT_CENTER_TOLERANCE_PX,
    )
    metrics.update(
        {
            "candidate_visible_depth": candidate_visible_depth,
            "reference_max_visible_depth": bounds["max_visible_depth"],
            "candidate_long_over_short": candidate_long_over_short,
            "reference_max_long_over_short": bounds["max_long_over_short"],
            "reference_min_long_over_short": bounds["min_long_over_short"],
            "candidate_bbox_long_over_short": candidate_bbox_long_over_short,
            "reference_min_long_side_px": bounds["min_long_side"],
            "candidate_area_px": candidate_area,
            "reference_min_area_px": bounds["min_area"],
            "candidate_far_edge_px": candidate_far_edge,
            "reference_min_far_edge_px": bounds["min_far_edge"],
            "reference_center_min_px": [bounds["min_center_x"], bounds["min_center_y"]],
            "reference_center_max_px": [bounds["max_center_x"], bounds["max_center_y"]],
            "reference_min_left_edge_x": bounds["min_left_edge_x"],
        }
    )
    base_result["metrics"] = metrics
    target = base_result.get("target") or _target_from_feature(candidate)
    allow_center_right_outside = bool(config.get("allow_center_right_outside", False))

    if require_far_edge_depth_ratio and not use_far_edge_depth_ratio:
        return {
            "ok": False,
            "feedback": "target_lost",
            "reason": "final image red quadrilateral far-edge-over-depth is unavailable",
            "metrics": metrics,
            "target": target,
        }
    if not base_result.get("ok", False):
        return base_result
    if allow_center_right_outside:
        if candidate_center[0] < bounds["min_left_edge_x"]:
            return {
                "ok": False,
                "feedback": "target_left",
                "reason": "final image center is left of reference red left edge",
                "metrics": metrics,
                "target": target,
            }
    else:
        if candidate_center[0] < bounds["min_center_x"] - center_tolerance[0]:
            return {
                "ok": False,
                "feedback": "target_left",
                "reason": "final image center is left of multi-reference grasp window",
                "metrics": metrics,
                "target": target,
            }
        if candidate_center[0] > bounds["max_center_x"] + center_tolerance[0]:
            return {
                "ok": False,
                "feedback": "target_right",
                "reason": "final image center is right of multi-reference grasp window",
                "metrics": metrics,
                "target": target,
            }
    if candidate_center[1] < bounds["min_center_y"] - center_tolerance[1]:
        return {
            "ok": False,
            "feedback": "arm_control_failed",
            "reason": "final image center is above multi-reference grasp window",
            "metrics": metrics,
            "target": target,
        }
    if candidate_center[1] > bounds["max_center_y"] + center_tolerance[1]:
        return {
            "ok": False,
            "feedback": "arm_control_failed",
            "reason": "final image center is below multi-reference grasp window",
            "metrics": metrics,
            "target": target,
        }
    if (
        bounds["min_long_over_short"] > 0.0
        and candidate_long_over_short < bounds["min_long_over_short"]
    ):
        return {
            "ok": False,
            "feedback": "target_too_far",
            "reason": "final image red far-edge-over-depth is below multi-reference grasp window",
            "metrics": metrics,
            "target": target,
        }
    if bounds["min_area"] > 0.0 and candidate_area < bounds["min_area"]:
        return {
            "ok": False,
            "feedback": "target_too_far",
            "reason": "final image red area is below multi-reference grasp window",
            "metrics": metrics,
            "target": target,
        }
    if bounds["min_far_edge"] > 0.0:
        if candidate_far_edge is None:
            return {
                "ok": False,
                "feedback": "target_lost",
                "reason": "final image red far edge is unavailable",
                "metrics": metrics,
                "target": target,
            }
        if candidate_far_edge < bounds["min_far_edge"]:
            return {
                "ok": False,
                "feedback": "target_too_far",
                "reason": "final image red far edge is below multi-reference grasp window",
                "metrics": metrics,
                "target": target,
            }
    if candidate_long_over_short > bounds["max_long_over_short"]:
        return {
            "ok": False,
            "feedback": "target_too_near",
            "reason": "final image red far-edge-over-depth exceeds multi-reference grasp window",
            "metrics": metrics,
            "target": target,
        }
    if candidate_visible_depth > bounds["max_visible_depth"]:
        return {
            "ok": False,
            "feedback": "target_too_near",
            "reason": "final image visible red exceeds multi-reference grasp window",
            "metrics": metrics,
            "target": target,
        }
    if (
        bounds["min_long_side"] > 0.0
        and candidate_long_side < bounds["min_long_side"]
    ):
        return {
            "ok": False,
            "feedback": "target_too_far",
            "reason": "final image red long side is below multi-reference grasp window",
            "metrics": metrics,
            "target": target,
        }
    return {
        "ok": True,
        "feedback": "target_in_grasp_window",
        "reason": "final image matches multi-reference grasp window",
        "metrics": metrics,
        "target": target,
    }


def _roi_bounds(roi: Any, width: int, height: int) -> Tuple[int, int, int, int]:
    if roi is None:
        return 0, 0, width, height
    x1, y1, x2, y2 = roi
    if all(0 <= float(value) <= 1 for value in roi):
        x1, x2 = float(x1) * width, float(x2) * width
        y1, y2 = float(y1) * height, float(y2) * height
    return (
        max(0, min(width, math.floor(float(x1)))),
        max(0, min(height, math.floor(float(y1)))),
        max(0, min(width, math.ceil(float(x2)))),
        max(0, min(height, math.ceil(float(y2)))),
    )


def _build_red_mask(frame_bgr: Any, detector_config: Mapping[str, Any]):
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for hsv_range in detector_config["colors"]["red"]:
        lower = np.asarray(hsv_range["lower"], dtype=np.uint8)
        upper = np.asarray(hsv_range["upper"], dtype=np.uint8)
        combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lower, upper))

    morphology = detector_config.get("morphology", {})
    open_kernel = int(morphology.get("open_kernel", 3))
    close_kernel = int(morphology.get("close_kernel", 5))
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel, open_kernel))
        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_OPEN,
            kernel,
            iterations=int(morphology.get("open_iterations", 1)),
        )
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=int(morphology.get("close_iterations", 1)),
        )

    height, width = combined.shape[:2]
    left, top, right, bottom = _roi_bounds(detector_config.get("roi"), width, height)
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    roi_mask[top:bottom, left:right] = 255
    return cv2.bitwise_and(combined, roi_mask)


def _component_quad_feature(component_mask: Any) -> Dict[str, Any]:
    import cv2

    contours, _hierarchy = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return {}
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 1.0:
        return {}
    for epsilon_ratio in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1):
        approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        points = [
            (float(point[0][0]), float(point[0][1]))
            for point in approx
        ]
        feature = _quad_edge_feature(points)
        if feature:
            return feature
    return {}


def _nearest_reference_score(
    feature: Mapping[str, Any],
    references: List[Mapping[str, Any]],
) -> float:
    center = _feature_pair(feature, "center_px")
    long_side = _long_side(feature)

    def score(reference: Mapping[str, Any]) -> float:
        reference_center = _feature_pair(reference, "center_px")
        center_distance = math.hypot(
            center[0] - reference_center[0],
            center[1] - reference_center[1],
        )
        long_side_distance = abs(long_side - _long_side(reference))
        return center_distance + 0.25 * long_side_distance

    return min(score(reference) for reference in references)


def extract_red_component_feature(
    frame_bgr: Any,
    detector_config: Mapping[str, Any],
    match_config: Optional[Mapping[str, Any]] = None,
    reference_feature: Optional[Mapping[str, Any]] = None,
    *,
    reference_features: Optional[List[Mapping[str, Any]]] = None,
    target_hint: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    import cv2
    import numpy as np

    match_config = dict(match_config or {})
    mask = _build_red_mask(frame_bgr, detector_config)
    min_area = float(match_config.get("min_area_px", DEFAULT_MIN_AREA_PX))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = float(stats[label, cv2.CC_STAT_LEFT])
        y = float(stats[label, cv2.CC_STAT_TOP])
        w = float(stats[label, cv2.CC_STAT_WIDTH])
        h = float(stats[label, cv2.CC_STAT_HEIGHT])
        ys, xs = np.where(labels == label)
        points = np.column_stack([xs, ys]).astype("float32")
        angle = 0.0
        if len(points) >= 5:
            (_center, (rect_w, rect_h), rect_angle) = cv2.minAreaRect(points)
            angle = float(rect_angle)
            if rect_w < rect_h:
                angle += 90.0
        feature = {
            "track_id": int(label),
            "center_px": [float(centroids[label][0]), float(centroids[label][1])],
            "size_px": [w, h],
            "bbox_px": [x, y, w, h],
            "area_px": area,
            "angle_deg": angle,
            "angle_reliable": False,
            "confidence": 1.0,
        }
        component_mask = np.zeros(mask.shape, dtype=np.uint8)
        component_mask[labels == label] = 255
        feature.update(_component_quad_feature(component_mask))
        candidates.append(feature)
    if not candidates:
        return None
    if target_hint is not None:
        hint_center = _feature_pair(target_hint, "center_px")
        hint_long_side = _long_side(target_hint)

        def hint_score(feature: Mapping[str, Any]) -> float:
            center = _feature_pair(feature, "center_px")
            center_distance = math.hypot(
                center[0] - hint_center[0],
                center[1] - hint_center[1],
            )
            size_error = abs(_long_side(feature) - hint_long_side)
            return center_distance + 0.15 * size_error

        return min(candidates, key=hint_score)
    if reference_features:
        accepted = [
            feature
            for feature in reference_features
            if isinstance(feature, Mapping) and not bool(feature.get("needs_review", False))
        ]
        if accepted:
            return min(candidates, key=lambda feature: _nearest_reference_score(feature, accepted))
    if reference_feature is None:
        return max(candidates, key=lambda item: float(item["area_px"]))
    return min(candidates, key=lambda feature: _nearest_reference_score(feature, [reference_feature]))


def _resolve_path(path_value: Any, base_dir: Optional[Path]) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return Path(base_dir or Path.cwd()) / path


def _load_reference_feature(
    detector_config: Mapping[str, Any],
    match_config: Mapping[str, Any],
    base_dir: Optional[Path],
) -> Optional[Dict[str, Any]]:
    if isinstance(match_config.get("reference_features"), Mapping):
        return dict(match_config["reference_features"])
    image_path = match_config.get("reference_image")
    if not image_path:
        return None
    import cv2

    resolved = _resolve_path(image_path, base_dir)
    frame = cv2.imread(str(resolved))
    if frame is None:
        raise FileNotFoundError(f"cannot read final grasp reference image: {resolved}")
    return extract_red_component_feature(frame, detector_config, match_config)


def _load_reference_feature_set(
    match_config: Mapping[str, Any],
    base_dir: Optional[Path],
) -> Optional[Dict[str, Any]]:
    features_file = match_config.get("reference_features_file")
    if not features_file:
        return None
    resolved = _resolve_path(features_file, base_dir)
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"cannot read final grasp reference features: {resolved}"
        ) from exc
    if not isinstance(data, Mapping):
        raise ValueError("final grasp reference features must be a JSON object")
    _valid_reference_features(data)
    return dict(data)


def match_final_grasp_view(
    frame_bgr: Any,
    detector_config: Mapping[str, Any],
    match_config: Mapping[str, Any],
    *,
    base_dir: Optional[Path] = None,
    target_hint: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not bool(match_config.get("enabled", False)):
        return None
    if frame_bgr is None:
        return {
            "ok": False,
            "feedback": "target_lost",
            "reason": "no final-view camera frame is available",
            "metrics": {},
            "target": None,
        }
    try:
        reference_set = _load_reference_feature_set(match_config, base_dir)
        if reference_set is not None:
            reference_features = _valid_reference_features(reference_set)
            candidate = extract_red_component_feature(
                frame_bgr,
                detector_config,
                match_config,
                reference_features=reference_features,
                target_hint=target_hint,
            )
            return compare_final_view_feature_set(reference_set, candidate, match_config)
        reference = _load_reference_feature(detector_config, match_config, base_dir)
        if reference is None:
            return None
        candidate = extract_red_component_feature(
            frame_bgr,
            detector_config,
            match_config,
            reference,
            target_hint=target_hint,
        )
        return compare_final_view_features(reference, candidate, match_config)
    except Exception as exc:
        return {
            "ok": False,
            "feedback": "arm_control_failed",
            "reason": f"final image matcher failed: {exc}",
            "metrics": {},
            "target": None,
        }
