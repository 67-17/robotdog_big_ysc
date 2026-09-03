"""Fixed-layout helpers for inspection tags.

The competition tag is treated as a fixed template: the area letter is in the
upper part of the inner content region, and the meter is below it. The actual
letter and meter status are random, so both ROIs still need recognition.
"""


DEFAULT_TEMPLATE = {
    "content_left_ratio": 1.0 / 6.0,
    "content_right_ratio": 5.0 / 6.0,
    "content_top_ratio": 0.0,
    "content_bottom_ratio": 2.0 / 3.0,
    "letter_height_ratio": 0.35,
}


def _normalize_bbox(bbox_xyxy):
    if len(bbox_xyxy) != 4:
        raise ValueError("panel bbox must contain four values")
    x0, y0, x1, y1 = [float(value) for value in bbox_xyxy]
    if x1 <= x0 or y1 <= y0:
        raise ValueError("panel bbox must have positive width and height")
    return x0, y0, x1, y1


def _round_bbox(values):
    return [round(float(value), 3) for value in values]


def split_inspection_tag_rois(panel_bbox_xyxy, template=None):
    """Return content, letter, and meter ROIs from a detected tag panel bbox."""
    cfg = dict(DEFAULT_TEMPLATE)
    if template:
        cfg.update(template)

    x0, y0, x1, y1 = _normalize_bbox(panel_bbox_xyxy)
    width = x1 - x0
    height = y1 - y0

    content_x0 = x0 + width * cfg["content_left_ratio"]
    content_x1 = x0 + width * cfg["content_right_ratio"]
    content_y0 = y0 + height * cfg["content_top_ratio"]
    content_y1 = y0 + height * cfg["content_bottom_ratio"]

    content_height = content_y1 - content_y0
    letter_y1 = content_y0 + content_height * cfg["letter_height_ratio"]

    return {
        "panel_bbox_xyxy": _round_bbox((x0, y0, x1, y1)),
        "content_bbox_xyxy": _round_bbox((content_x0, content_y0, content_x1, content_y1)),
        "letter_roi_xyxy": _round_bbox((content_x0, content_y0, content_x1, letter_y1)),
        "meter_roi_xyxy": _round_bbox((content_x0, letter_y1, content_x1, content_y1)),
    }
