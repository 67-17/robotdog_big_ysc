"""Adapt meter recognition results to inspection task states."""


STATUS_MAP = {
    "偏低": {
        "state": "abnormal",
        "description": "偏低",
        "status_text": "状态异常",
    },
    "正常": {
        "state": "normal",
        "description": "正常",
        "status_text": "状态正常",
    },
    "偏高": {
        "state": "abnormal",
        "description": "偏高",
        "status_text": "状态异常",
    },
    "未知": {
        "state": "unknown",
        "description": "无法确认",
        "status_text": "状态无法确认",
    },
}


def normalize_meter_status(source_status):
    """Map meter_recognition status text to inspection status fields."""
    status = str(source_status or "未知")
    normalized = dict(STATUS_MAP.get(status, STATUS_MAP["未知"]))
    normalized["source_status"] = status
    return normalized


def _point_to_list(point):
    x, y = point
    return [float(x), float(y)]


def _bbox_from_center_radius(center, radius):
    cx, cy = center
    radius = float(radius)
    return [
        float(cx) - radius,
        float(cy) - radius,
        float(cx) + radius,
        float(cy) + radius,
    ]


def adapt_meter_result(raw_result, meter_id="meter_01"):
    """Convert a meter_recognition result dict to inspection result fields."""
    meter_found = bool(raw_result.get("meter_found", True))
    pointer_found = bool(raw_result.get("pointer_found", meter_found))
    source_status = raw_result.get("status", "未知")
    normalized = normalize_meter_status(source_status)
    state_reason = f"meter_status_recognition:{normalized['source_status']}"

    if not meter_found:
        normalized = normalize_meter_status("未知")
        state_reason = "meter_not_found"
        pointer_found = False
    elif not pointer_found:
        normalized = normalize_meter_status("未知")
        state_reason = "pointer_not_found"

    center = raw_result.get("center", (0.0, 0.0))
    radius = float(raw_result.get("radius", 0.0))
    tip_point = raw_result.get("tip_point", center)

    return {
        "meter_id": meter_id,
        "state": normalized["state"],
        "description": normalized["description"],
        "status_text": normalized["status_text"],
        "state_text": normalized["source_status"],
        "state_reason": state_reason,
        "center_px": _point_to_list(center),
        "radius_px": radius,
        "pointer_tip_px": _point_to_list(tip_point),
        "bbox_xyxy": _bbox_from_center_radius(center, radius),
        "meter_found": meter_found,
        "pointer_found": pointer_found,
        "pointer_support": raw_result.get("pointer_support"),
        "pointer_method": raw_result.get("pointer_method"),
        "pointer_line": raw_result.get("pointer_line"),
        "status_evidence": raw_result.get("status_evidence"),
        "raw_color_counts": dict(raw_result.get("color_counts", {})),
    }
