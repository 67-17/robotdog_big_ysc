"""Aggregate inspection detections into abnormal area output."""


VALID_AREAS = ("A", "B", "C", "D")


def _meter_by_id(meters):
    return {meter["meter_id"]: meter for meter in meters}


def _count_check(areas):
    if any(area["state"] == "unknown" for area in areas):
        return "incomplete"
    abnormal_count = sum(1 for area in areas if area["state"] == "abnormal")
    normal_count = sum(1 for area in areas if area["state"] == "normal")
    return "pass" if abnormal_count == 2 and normal_count == 2 else "fail"


def _unknown_area(binding, reason):
    return {
        "area": binding["area"],
        "state": "unknown",
        "description": "无法确认",
        "meter_id": binding.get("meter_id"),
        "binding_status": binding.get("binding_status", "ambiguous"),
        "reason": reason,
    }


def _area_from_binding(binding, meters_by_id):
    area = str(binding["area"]).upper()
    if area not in VALID_AREAS:
        return _unknown_area({"area": area, "meter_id": None}, "invalid_area")

    if binding.get("binding_status") != "ok":
        return _unknown_area(binding, binding.get("reason", "binding_not_ok"))

    meter_id = binding.get("meter_id")
    meter = meters_by_id.get(meter_id)
    if meter is None:
        return _unknown_area(binding, "meter_not_found")

    state = meter.get("state", "unknown")
    if state not in ("normal", "abnormal"):
        return {
            "area": area,
            "state": "unknown",
            "description": meter.get("description", "无法确认"),
            "meter_id": meter_id,
            "binding_status": "ok",
            "reason": meter.get("state_reason", "meter_state_unknown"),
        }

    return {
        "area": area,
        "state": state,
        "description": meter.get("description", ""),
        "meter_id": meter_id,
        "binding_status": "ok",
        "reason": meter.get("state_reason", ""),
    }


def build_inspection_result(bindings, meters):
    """Build the final inspection result from bindings and adapted meters."""
    meters_by_id = _meter_by_id(meters)
    areas = [_area_from_binding(binding, meters_by_id) for binding in bindings]
    areas.sort(key=lambda item: item["area"])

    abnormal_areas = [
        area["area"] for area in areas
        if area["state"] == "abnormal" and area["area"] in VALID_AREAS
    ]
    unknown_areas = [
        area["area"] for area in areas
        if area["state"] == "unknown" and area["area"] in VALID_AREAS
    ]

    return {
        "ok": True,
        "areas": areas,
        "abnormal_areas": abnormal_areas,
        "unknown_areas": unknown_areas,
        "count_check": _count_check(areas),
    }
