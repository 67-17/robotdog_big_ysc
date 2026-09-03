"""Single-frame inspection pipeline for real camera input."""

import io
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_DIR = Path(__file__).resolve().parent
TASK_DIR = MODULE_DIR.parent


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


layout = _load_module(MODULE_DIR / "layout.py", "layout")
letter_recognition = _load_module(MODULE_DIR / "letter_recognition.py", "letter_recognition")
meter_status_adapter = _load_module(MODULE_DIR / "meter_status_adapter.py", "meter_status_adapter")
result_aggregation = _load_module(MODULE_DIR / "result_aggregation.py", "result_aggregation")
meter_status_recognition = _load_module(
    TASK_DIR / "meter_recognition" / "scripts" / "meter_status_recognition.py",
    "meter_status_recognition",
)


LETTER_ANCHOR_MIN_CONFIDENCE = 0.72
LETTER_SEARCH_X_RANGE = (0.18, 0.86)
LETTER_SEARCH_Y_RANGE = (0.32, 0.76)
METER_ROI_HALF_WIDTH_GLYPH_RATIO = 3.2
METER_ROI_TOP_GLYPH_RATIO = 1.2
METER_ROI_BOTTOM_GLYPH_RATIO = 7.0
METER_EXPECTED_CENTER_Y_GLYPH_RATIO = 3.9
METER_MIN_RADIUS_GLYPH_RATIO = 0.8
METER_MAX_RADIUS_GLYPH_RATIO = 2.2
METER_MAX_CENTER_DISTANCE_GLYPH_RATIO = 1.6
METER_CLEAR_SUPPORT_SCORE = 0.9


def warm_up_recognizers():
    """Build cached templates before the first live camera observation."""
    templates = letter_recognition._templates()
    return sum(len(variants) for variants in templates.values())


def _crop(image, bbox_xyxy):
    clipped = _clip_bbox_to_image(bbox_xyxy, image.size)
    if clipped is None:
        return None
    x0, y0, x1, y1 = [int(round(value)) for value in clipped]
    return image.crop((x0, y0, x1, y1))


def _inset_bbox(bbox_xyxy, ratio=0.04):
    x0, y0, x1, y1 = [float(value) for value in bbox_xyxy]
    width = x1 - x0
    height = y1 - y0
    dx = width * ratio
    dy = height * ratio
    return [x0 + dx, y0 + dy, x1 - dx, y1 - dy]


def _bbox_has_area(bbox_xyxy, min_size=2.0):
    if bbox_xyxy is None:
        return False
    x0, y0, x1, y1 = [float(value) for value in bbox_xyxy]
    return (x1 - x0) >= min_size and (y1 - y0) >= min_size


def _clip_bbox_to_image(bbox_xyxy, image_size):
    width, height = image_size
    clipped = _clamp_bbox(bbox_xyxy, width, height)
    return clipped if _bbox_has_area(clipped) else None


def _roi_error(reason, rois, geometry_source):
    return {
        "ok": False,
        "reason": reason,
        "geometry_source": geometry_source,
        "rois": rois,
        "abnormal_areas": [],
        "unknown_areas": [],
    }


def _is_clear_meter_result(raw_result):
    return (
        raw_result.get("meter_found", True)
        and raw_result.get("pointer_found", True)
        and raw_result.get("status") not in (None, "未知")
    )


def _jpeg_smoothed_image(image, quality=88):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _offset_bbox(bbox_xyxy, dx, dy):
    if bbox_xyxy is None:
        return None
    x0, y0, x1, y1 = bbox_xyxy
    return [x0 + dx, y0 + dy, x1 + dx, y1 + dy]


def _offset_point(point, dx, dy):
    if point is None:
        return None
    x, y = point
    return [x + dx, y + dy]


def _offset_letter_detection(letter_detection, roi_bbox):
    x0, y0, _, _ = roi_bbox
    shifted = dict(letter_detection)
    shifted["center_px"] = _offset_point(shifted.get("center_px"), x0, y0)
    shifted["bbox_xyxy"] = _offset_bbox(shifted.get("bbox_xyxy"), x0, y0)
    return shifted


def _locate_letter_anchor(image, min_height_px=None):
    """Find the large A/B/C/D glyph on its bright paper before searching for a meter."""
    cv2 = meter_status_recognition.import_cv2_optional()
    if cv2 is None:
        return None
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark_mask = (gray < 105).astype(np.uint8) * 255
    count, _labels, stats, centers = cv2.connectedComponentsWithStats(
        dark_mask,
        8,
    )
    image_area = float(width * height)
    default_min_height = max(24, int(round(height * 0.035)))
    if min_height_px is None:
        min_height = default_min_height
    else:
        min_height = max(
            8,
            min(default_min_height, int(round(float(min_height_px)))),
        )
    max_height = max(min_height + 1, int(round(height * 0.18)))
    min_width = max(8, int(round(width * 0.008)))
    max_width = max(min_width + 1, int(round(width * 0.12)))
    min_area = max(80, int(round(image_area * 0.00008)))
    max_area = max(min_area + 1, int(round(image_area * 0.006)))
    candidates = []
    for index in range(1, count):
        x, y, component_width, component_height, area = [
            int(value) for value in stats[index]
        ]
        center_x, center_y = [float(value) for value in centers[index]]
        if not (
            width * LETTER_SEARCH_X_RANGE[0]
            <= center_x
            <= width * LETTER_SEARCH_X_RANGE[1]
            and height * LETTER_SEARCH_Y_RANGE[0]
            <= center_y
            <= height * LETTER_SEARCH_Y_RANGE[1]
        ):
            continue
        if not (
            min_height <= component_height <= max_height
            and min_width <= component_width <= max_width
            and min_area <= area <= max_area
        ):
            continue

        # Preserve enough of the white tag around oblique/small glyphs. Tight
        # crops made a clear distant A score just below the anchor threshold.
        pad_x = int(round(component_height * 1.4))
        pad_y = int(round(component_height * 1.2))
        recognition_bbox = _clamp_bbox(
            [
                x - pad_x,
                y - pad_y,
                x + component_width + pad_x,
                y + component_height + pad_y,
            ],
            width,
            height,
        )
        patch = _crop(image, recognition_bbox)
        if patch is None:
            continue
        patch_gray = np.asarray(patch.convert("L"), dtype=np.uint8)
        white_ratio = float(np.mean(patch_gray > 175))
        if white_ratio < 0.55:
            continue
        detection = letter_recognition.recognize_letter_roi(patch)
        confidence = float(detection.get("confidence", 0.0) or 0.0)
        if not detection.get("label") or confidence < LETTER_ANCHOR_MIN_CONFIDENCE:
            continue
        detection = _offset_letter_detection(detection, recognition_bbox)
        detection["anchor_white_ratio"] = round(white_ratio, 3)
        detection["component_bbox_xyxy"] = [
            float(x),
            float(y),
            float(x + component_width),
            float(y + component_height),
        ]
        horizontal_distance = abs(center_x - width / 2.0) / max(1.0, width)
        score = confidence + white_ratio * 0.08 - horizontal_distance * 0.02
        candidates.append(
            {
                "score": score,
                "detection": detection,
                "recognition_bbox_xyxy": recognition_bbox,
                "glyph_height": float(component_height),
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])


def _letter_anchored_rois(image_size, anchor):
    width, height = image_size
    detection = anchor["detection"]
    center_x, center_y = detection["center_px"]
    glyph_height = float(anchor["glyph_height"])
    meter_roi = _clamp_bbox(
        [
            center_x - METER_ROI_HALF_WIDTH_GLYPH_RATIO * glyph_height,
            center_y + METER_ROI_TOP_GLYPH_RATIO * glyph_height,
            center_x + METER_ROI_HALF_WIDTH_GLYPH_RATIO * glyph_height,
            center_y + METER_ROI_BOTTOM_GLYPH_RATIO * glyph_height,
        ],
        width,
        height,
    )
    letter_roi = list(anchor["recognition_bbox_xyxy"])
    content_bbox = _clamp_bbox(
        [
            min(letter_roi[0], meter_roi[0]),
            letter_roi[1],
            max(letter_roi[2], meter_roi[2]),
            meter_roi[3],
        ],
        width,
        height,
    )
    return {
        "panel_bbox_xyxy": content_bbox,
        "content_bbox_xyxy": content_bbox,
        "letter_roi_xyxy": letter_roi,
        "letter_recognition_bbox_xyxy": letter_roi,
        "letter_component_bbox_xyxy": detection["component_bbox_xyxy"],
        "meter_roi_xyxy": meter_roi,
        "meter_expected_center_px": [
            center_x,
            center_y + METER_EXPECTED_CENTER_Y_GLYPH_RATIO * glyph_height,
        ],
        "letter_glyph_height_px": glyph_height,
    }


def _meter_result_support_score(result):
    if not _is_clear_meter_result(result):
        return -1.0
    support = result.get("pointer_support") or {}
    return float(support.get("hit_ratio", 0.0)) + float(
        support.get("longest_run_ratio", 0.0)
    )


def _analyze_letter_anchored_meter(meter_roi, rois):
    meter_x0, meter_y0, _, _ = rois["meter_roi_xyxy"]
    expected_x, expected_y = rois["meter_expected_center_px"]
    expected_center = (expected_x - meter_x0, expected_y - meter_y0)
    glyph_height = float(rois["letter_glyph_height_px"])
    located = meter_status_recognition.locate_meter_circle(
        meter_roi,
        expected_center=expected_center,
        min_radius=METER_MIN_RADIUS_GLYPH_RATIO * glyph_height,
        max_radius=METER_MAX_RADIUS_GLYPH_RATIO * glyph_height,
        max_center_distance=METER_MAX_CENTER_DISTANCE_GLYPH_RATIO * glyph_height,
    )
    if located is None:
        fallback_result = meter_status_recognition.analyze_meter_rgb_image(
            meter_roi,
            center_hint=expected_center,
            radius_hint=max(5.0, glyph_height * 1.4),
        )
        fallback_result = dict(fallback_result)
        fallback_result["letter_anchor_expected_center"] = expected_center
        fallback_result["letter_anchor_support_score"] = _meter_result_support_score(
            fallback_result
        )
        fallback_result["circle_source"] = "letter_layout_fallback"
        return fallback_result

    center = tuple(float(value) for value in located["center"])
    radius = float(located["radius"])
    result = meter_status_recognition.analyze_meter_rgb_image(
        meter_roi,
        center_hint=center,
        radius_hint=radius,
    )
    best_result = result
    best_score = _meter_result_support_score(result)
    if best_score < METER_CLEAR_SUPPORT_SCORE:
        refinements = (
            (expected_center, radius),
            ((center[0] - 0.10 * radius, center[1] - 0.10 * radius), radius * 1.05),
            ((center[0], center[1] - 0.10 * radius), radius * 0.95),
            ((center[0] + 0.10 * radius, center[1] - 0.10 * radius), radius * 1.05),
            ((center[0] - 0.10 * radius, center[1]), radius * 1.05),
        )
        for candidate_center, candidate_radius in refinements:
            candidate = meter_status_recognition.analyze_meter_rgb_image(
                meter_roi,
                center_hint=candidate_center,
                radius_hint=candidate_radius,
            )
            score = _meter_result_support_score(candidate)
            if score > best_score:
                best_result = candidate
                best_score = score
            if best_score >= METER_CLEAR_SUPPORT_SCORE:
                break
    best_result = dict(best_result)
    best_result["letter_anchor_expected_center"] = expected_center
    best_result["letter_anchor_support_score"] = best_score
    best_result["circle_source"] = "detected_circle"
    return best_result


def _meter_anchor_from_result(raw_meter_result, meter_roi_bbox):
    if not raw_meter_result.get("meter_found", True):
        return None
    center = raw_meter_result.get("center")
    radius = raw_meter_result.get("radius")
    if center is None or radius is None:
        return None
    try:
        local_x, local_y = center
        radius = float(radius)
    except (TypeError, ValueError):
        return None
    if radius <= 0:
        return None
    meter_x0, meter_y0, _, _ = meter_roi_bbox
    return (float(meter_x0) + float(local_x), float(meter_y0) + float(local_y)), radius


def _recognize_letter_from_meter_result(image, raw_meter_result, meter_roi_bbox):
    anchor = _meter_anchor_from_result(raw_meter_result, meter_roi_bbox)
    if anchor is None:
        return None
    refined_rois = _meter_anchored_rois(image.size, anchor[0], anchor[1])
    letter_bbox = _inset_bbox(refined_rois["letter_roi_xyxy"], ratio=0.0)
    if not _bbox_has_area(_clip_bbox_to_image(letter_bbox, image.size)):
        return None
    letter_roi = _crop(image, letter_bbox)
    if letter_roi is None:
        return None
    refined_rois["letter_recognition_bbox_xyxy"] = letter_bbox
    letter_detection = letter_recognition.recognize_letter_roi(letter_roi)
    return _offset_letter_detection(letter_detection, letter_bbox), refined_rois


def _clamp_bbox(bbox_xyxy, width, height):
    x0, y0, x1, y1 = bbox_xyxy
    return [
        max(0.0, min(float(width), float(x0))),
        max(0.0, min(float(height), float(y0))),
        max(0.0, min(float(width), float(x1))),
        max(0.0, min(float(height), float(y1))),
    ]


def _meter_anchored_rois(image_size, center, radius):
    width, height = image_size
    cx, cy = center
    letter_roi = _clamp_bbox(
        [cx - 1.40 * radius, cy - 2.95 * radius, cx + 1.40 * radius, cy - 1.05 * radius],
        width,
        height,
    )
    meter_roi = _clamp_bbox(
        [cx - 1.08 * radius, cy - 1.08 * radius, cx + 1.08 * radius, cy + 1.08 * radius],
        width,
        height,
    )
    content_bbox = _clamp_bbox(
        [
            min(letter_roi[0], meter_roi[0]),
            letter_roi[1],
            max(letter_roi[2], meter_roi[2]),
            meter_roi[3],
        ],
        width,
        height,
    )
    return {
        "panel_bbox_xyxy": content_bbox,
        "content_bbox_xyxy": content_bbox,
        "letter_roi_xyxy": letter_roi,
        "meter_roi_xyxy": meter_roi,
    }


def analyze_inspection_frame(
    image,
    panel_bbox_xyxy=None,
    meter_analyzer=None,
    letter_anchor_min_height_px=None,
):
    """Analyze one camera frame containing one fixed-layout inspection tag."""
    pil_image = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
    meter_hint = None
    letter_anchor = None
    if panel_bbox_xyxy is None:
        letter_anchor = _locate_letter_anchor(
            pil_image,
            min_height_px=letter_anchor_min_height_px,
        )
        if letter_anchor is not None:
            rois = _letter_anchored_rois(pil_image.size, letter_anchor)
            geometry_source = "letter_anchor"
        else:
            located_meter = meter_status_recognition.locate_meter_circle(pil_image)
            if located_meter is None:
                return {
                    "ok": False,
                    "reason": "meter_not_found",
                    "geometry_source": "meter_anchor",
                    "abnormal_areas": [],
                    "unknown_areas": [],
                }
            meter_hint = (located_meter["center"], located_meter["radius"])
            rois = _meter_anchored_rois(pil_image.size, *meter_hint)
            geometry_source = "meter_anchor"
        inset_ratio = 0.02
    else:
        rois = layout.split_inspection_tag_rois(panel_bbox_xyxy)
        geometry_source = "manual_panel"
        inset_ratio = 0.04

    if geometry_source in {"meter_anchor", "letter_anchor"}:
        letter_recognition_bbox = _inset_bbox(rois["letter_roi_xyxy"], ratio=0.0)
    else:
        letter_recognition_bbox = _inset_bbox(rois["letter_roi_xyxy"], ratio=inset_ratio)
    rois["letter_recognition_bbox_xyxy"] = letter_recognition_bbox
    if not _bbox_has_area(_clip_bbox_to_image(letter_recognition_bbox, pil_image.size)):
        return _roi_error("letter_roi_out_of_frame", rois, geometry_source)
    if not _bbox_has_area(_clip_bbox_to_image(rois["meter_roi_xyxy"], pil_image.size)):
        return _roi_error("meter_roi_out_of_frame", rois, geometry_source)

    letter_roi = _crop(pil_image, letter_recognition_bbox)
    meter_roi = _crop(pil_image, rois["meter_roi_xyxy"])
    if letter_roi is None:
        return _roi_error("letter_roi_out_of_frame", rois, geometry_source)
    if meter_roi is None:
        return _roi_error("meter_roi_out_of_frame", rois, geometry_source)

    if letter_anchor is not None:
        letter_detection = dict(letter_anchor["detection"])
    else:
        letter_detection = letter_recognition.recognize_letter_roi(letter_roi)
        letter_detection = _offset_letter_detection(letter_detection, letter_recognition_bbox)

    if meter_analyzer is not None:
        raw_meter_result = meter_analyzer(meter_roi)
    elif geometry_source == "letter_anchor":
        raw_meter_result = _analyze_letter_anchored_meter(meter_roi, rois)
    elif meter_hint is not None:
        meter_x0, meter_y0, _, _ = rois["meter_roi_xyxy"]
        center, radius = meter_hint
        local_center = (center[0] - meter_x0, center[1] - meter_y0)
        raw_meter_result = meter_status_recognition.analyze_meter_rgb_image(
            meter_roi,
            center_hint=local_center,
            radius_hint=radius,
        )
        if not _is_clear_meter_result(raw_meter_result):
            retry_meter_result = meter_status_recognition.analyze_meter_rgb_image(meter_roi)
            if _is_clear_meter_result(retry_meter_result):
                raw_meter_result = retry_meter_result
            else:
                smoothed_meter_result = meter_status_recognition.analyze_meter_rgb_image(
                    _jpeg_smoothed_image(meter_roi)
                )
                if _is_clear_meter_result(smoothed_meter_result):
                    raw_meter_result = smoothed_meter_result
    else:
        raw_meter_result = meter_status_recognition.analyze_meter_rgb_image(meter_roi)

    if not letter_detection.get("label"):
        refined_letter = _recognize_letter_from_meter_result(
            pil_image,
            raw_meter_result,
            rois["meter_roi_xyxy"],
        )
        if refined_letter is not None:
            refined_letter_detection, refined_rois = refined_letter
            if refined_letter_detection.get("label"):
                previous_rois = dict(rois)
                rois = dict(refined_rois)
                rois["initial_panel_bbox_xyxy"] = previous_rois.get("panel_bbox_xyxy")
                rois["initial_content_bbox_xyxy"] = previous_rois.get("content_bbox_xyxy")
                rois["initial_letter_roi_xyxy"] = previous_rois.get("letter_roi_xyxy")
                rois["initial_meter_roi_xyxy"] = previous_rois.get("meter_roi_xyxy")
                letter_detection = refined_letter_detection

    meter_detection = meter_status_adapter.adapt_meter_result(raw_meter_result, meter_id="meter_01")

    if letter_detection.get("label"):
        binding = {
            "area": letter_detection["label"],
            "meter_id": meter_detection["meter_id"],
            "binding_status": "ok",
            "reason": "same_fixed_layout_tag",
        }
    else:
        binding = {
            "area": "?",
            "meter_id": meter_detection["meter_id"],
            "binding_status": "ambiguous",
            "reason": letter_detection.get("reason", "letter_unknown"),
        }

    result = result_aggregation.build_inspection_result([binding], [meter_detection])
    result["rois"] = rois
    result["letter_detection"] = letter_detection
    result["meter_detection"] = meter_detection
    result["geometry_source"] = geometry_source
    return result
