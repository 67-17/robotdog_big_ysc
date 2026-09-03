"""Bind detected A/B/C/D letters to detected meters."""

import math


def _center_px(item):
    if "center_px" in item:
        x, y = item["center_px"]
        return float(x), float(y)

    x0, y0, x1, y1 = item["bbox_xyxy"]
    return (float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0


def _distance_px(a, b):
    ax, ay = _center_px(a)
    bx, by = _center_px(b)
    return math.hypot(ax - bx, ay - by)


def _ambiguous(area, reason, distance_px=None):
    return {
        "area": area,
        "meter_id": None,
        "distance_px": distance_px,
        "binding_confidence": 0.0,
        "binding_status": "ambiguous",
        "reason": reason,
    }


def bind_letters_to_meters(letters, meters, max_distance_px=180.0):
    """Bind each letter to its nearest available meter.

    This first version deliberately avoids guessing a second-best meter when
    the nearest one is already taken. That conflict should trigger re-shooting
    or a better view instead of silently binding the wrong area.
    """
    bound_meter_ids = set()
    bindings = []

    for letter in letters:
        area = str(letter["label"]).upper()
        if not meters:
            bindings.append(_ambiguous(area, "no_meter_detected"))
            continue

        candidates = sorted(
            (
                (_distance_px(letter, meter), meter)
                for meter in meters
            ),
            key=lambda item: item[0],
        )
        nearest_distance, nearest_meter = candidates[0]
        meter_id = nearest_meter["meter_id"]

        if nearest_distance > max_distance_px:
            bindings.append(
                _ambiguous(area, "nearest_meter_too_far", round(nearest_distance, 3))
            )
            continue

        if meter_id in bound_meter_ids:
            bindings.append(
                _ambiguous(area, "meter_already_bound", round(nearest_distance, 3))
            )
            continue

        bound_meter_ids.add(meter_id)
        confidence = max(0.0, 1.0 - nearest_distance / float(max_distance_px))
        bindings.append(
            {
                "area": area,
                "meter_id": meter_id,
                "distance_px": round(nearest_distance, 3),
                "binding_confidence": round(confidence, 3),
                "binding_status": "ok",
                "reason": "nearest_meter",
            }
        )

    return bindings
