"""Recognize A/B/C/D letters from the fixed tag letter ROI."""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


LETTERS = ("A", "B", "C", "D")
NORMALIZED_SIZE = (64, 64)
RAW_VERY_STRONG_MIN_SCORE = 0.80
RAW_VERY_STRONG_MIN_MARGIN = 0.10
GLYPH_HEIGHT_TOLERANCE_PX = 2


def _resample_lanczos():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _load_font(size=92):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _dark_mask(image):
    gray_image = image.convert("L")
    gray = np.asarray(gray_image, dtype=np.int16)
    local_background = np.asarray(
        gray_image.filter(ImageFilter.BoxBlur(max(5, min(image.size) // 12))),
        dtype=np.int16,
    )
    return gray + 18 < local_background


def _text_bbox(draw, text, font):
    """Return a text bbox for both TrueType and PIL default bitmap fonts."""
    try:
        return draw.textbbox((0, 0), text, font=font)
    except ValueError:
        width, height = draw.textsize(text, font=font)
        return (0, 0, width, height)


def _bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _connected_components(mask):
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []

    for start_y, start_x in zip(*np.where(mask & ~visited)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        points = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if mask[next_y, next_x] and not visited[next_y, next_x]:
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
        components.append(points)
    return components


def _glyph_component_mask(mask):
    height, width = mask.shape
    image_area = float(height * width)
    candidates = []

    for points in _connected_components(mask):
        ys = np.fromiter((point[0] for point in points), dtype=np.int32)
        xs = np.fromiter((point[1] for point in points), dtype=np.int32)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        component_width = x1 - x0
        component_height = y1 - y0
        area = len(points)

        touches_side_edge = x0 <= 1 or x1 >= width - 1
        if touches_side_edge:
            continue
        if area < image_area * 0.008 or area > image_area * 0.45:
            continue
        if (
            component_height + GLYPH_HEIGHT_TOLERANCE_PX < height * 0.22
            or component_height > height
        ):
            continue
        if component_width < width * 0.06 or component_width > width * 0.80:
            continue

        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        center_distance = abs(center_x - width / 2.0) / width + abs(center_y - height / 2.0) / height
        score = area / image_area - center_distance * 0.04
        candidates.append((score, points, (x0, y0, x1, y1)))

    if not candidates:
        return None, None

    _, points, bbox = max(candidates, key=lambda item: item[0])
    component_mask = np.zeros_like(mask, dtype=bool)
    ys, xs = zip(*points)
    component_mask[np.asarray(ys), np.asarray(xs)] = True
    return component_mask, bbox


def _normalize_glyph(image):
    raw_mask = _dark_mask(image)
    mask, bbox = _glyph_component_mask(raw_mask)
    if bbox is None:
        closed = Image.fromarray(raw_mask.astype(np.uint8) * 255, "L")
        closed = closed.filter(ImageFilter.MaxFilter(3))
        closed = closed.filter(ImageFilter.MinFilter(3))
        mask, bbox = _glyph_component_mask(np.asarray(closed) >= 128)
    if bbox is None:
        return None, None

    x0, y0, x1, y1 = bbox
    glyph = Image.fromarray((mask[y0:y1, x0:x1].astype(np.uint8) * 255), "L")
    scale = min(52.0 / glyph.width, 52.0 / glyph.height)
    normalized_size = (
        max(1, int(round(glyph.width * scale))),
        max(1, int(round(glyph.height * scale))),
    )
    glyph = glyph.resize(normalized_size, _resample_lanczos())

    canvas = Image.new("L", NORMALIZED_SIZE, 0)
    offset = (
        (NORMALIZED_SIZE[0] - glyph.width) // 2,
        (NORMALIZED_SIZE[1] - glyph.height) // 2,
    )
    canvas.paste(glyph, offset)
    normalized = np.array(canvas, dtype=np.float32) / 255.0
    return normalized, [float(x0), float(y0), float(x1), float(y1)]


def _render_template(
    letter,
    horizontal_scale=1.0,
    vertical_scale=1.0,
    rotation_deg=0.0,
):
    image = Image.new("RGB", (220, 110), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font()
    bbox = _text_bbox(draw, letter, font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (image.width - text_width) / 2.0 - bbox[0]
    y = (image.height - text_height) / 2.0 - bbox[1]
    draw.text((x, y), letter, fill="black", font=font)
    if horizontal_scale != 1.0 or vertical_scale != 1.0:
        scaled_width = int(round(image.width * horizontal_scale))
        scaled_height = int(round(image.height * vertical_scale))
        compressed = image.resize(
            (scaled_width, scaled_height),
            _resample_lanczos(),
        )
        canvas = Image.new("RGB", image.size, "white")
        canvas.paste(
            compressed,
            (
                (image.width - scaled_width) // 2,
                (image.height - scaled_height) // 2,
            ),
        )
        image = canvas
    if rotation_deg:
        image = image.rotate(
            rotation_deg,
            resample=Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC,
            fillcolor="white",
        )
    normalized, _ = _normalize_glyph(image)
    return normalized


@lru_cache(maxsize=1)
def _templates():
    horizontal_scales = (0.65, 0.75, 0.85, 1.0)
    vertical_scales = (0.75, 1.0)
    rotations = (-10.0, 0.0, 10.0)
    return {
        letter: tuple(
            _render_template(letter, horizontal_scale, vertical_scale, rotation)
            for horizontal_scale in horizontal_scales
            for vertical_scale in vertical_scales
            for rotation in rotations
        )
        for letter in LETTERS
    }


def _similarity(a, b):
    a_mask = a >= 0.5
    b_mask = b >= 0.5
    intersection = float(np.count_nonzero(a_mask & b_mask))
    total = float(np.count_nonzero(a_mask) + np.count_nonzero(b_mask))
    return (2.0 * intersection / total) if total else 0.0


def _count_holes(mask):
    background = ~mask
    height, width = background.shape
    holes = 0
    for points in _connected_components(background):
        ys, xs = zip(*points)
        if min(xs) == 0 or min(ys) == 0 or max(xs) == width - 1 or max(ys) == height - 1:
            continue
        if len(points) >= 6:
            holes += 1
    return holes


def _longest_row_run(row):
    longest = 0
    current = 0
    for value in row:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _shape_is_plausible(normalized, label):
    mask = normalized >= 0.5
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    holes = _count_holes(mask)

    if label == "B":
        return holes == 2
    if label == "C":
        return holes == 0
    if label == "D":
        if holes != 1:
            return False
        left_width = max(2, int(round(width * 0.30)))
        left_columns = mask[y0:y1, x0 : x0 + left_width]
        left_continuity = float(np.max(np.mean(left_columns, axis=0)))
        right_edges = []
        for row in mask[y0:y1, x0:x1]:
            xs = np.where(row)[0]
            if len(xs):
                right_edges.append(int(xs.max()))
        right_curve = max(right_edges) - min(right_edges) if right_edges else 0
        return left_continuity >= 0.85 and right_curve >= width * 0.15
    if label == "A":
        if holes != 1:
            return False
        middle_y0 = y0 + int(round(height * 0.42))
        middle_y1 = y0 + int(round(height * 0.72))
        longest = max(
            _longest_row_run(mask[y, x0:x1])
            for y in range(middle_y0, max(middle_y0 + 1, middle_y1))
        )
        return longest >= width * 0.45
    return False


def recognize_letter_roi(image, min_confidence=0.70, min_margin=0.035):
    """Recognize one A/B/C/D letter from a cropped letter ROI."""
    normalized, bbox = _normalize_glyph(image)
    if normalized is None:
        return {
            "label": None,
            "state": "unknown",
            "confidence": 0.0,
            "center_px": None,
            "bbox_xyxy": None,
            "source": "template_match",
            "reason": "no_letter_pixels",
        }

    scores = {
        letter: max(_similarity(normalized, template) for template in templates)
        for letter, templates in _templates().items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    raw_best_label, raw_best_score = ranked[0]
    raw_margin = raw_best_score - ranked[1][1]
    shape_plausibility = {
        label: _shape_is_plausible(normalized, label) for label in LETTERS
    }
    plausible_labels = [
        label for label, _ in ranked if shape_plausibility[label]
    ]
    raw_very_strong = (
        raw_best_score >= RAW_VERY_STRONG_MIN_SCORE
        and raw_margin >= RAW_VERY_STRONG_MIN_MARGIN
    )
    if raw_very_strong or not plausible_labels:
        best_label = raw_best_label
        best_score = raw_best_score
        margin = raw_margin
    else:
        best_label = max(plausible_labels, key=scores.get)
        best_score = scores[best_label]
        other_plausible = [
            scores[label] for label in plausible_labels if label != best_label
        ]
        margin = best_score - max(other_plausible) if other_plausible else best_score
    confidence = round(best_score, 3)
    strong_match = confidence >= min_confidence and margin >= min_margin
    degraded_match = confidence >= 0.56 and margin >= 0.10
    very_strong_match = best_label == raw_best_label and raw_very_strong
    plausible_shape = shape_plausibility[best_label]
    accepted = very_strong_match or (
        plausible_shape and (strong_match or degraded_match)
    )

    x0, y0, x1, y1 = bbox
    result = {
        "label": best_label if accepted else None,
        "state": "ok" if accepted else "unknown",
        "confidence": confidence,
        "margin": round(margin, 3),
        "center_px": [round((x0 + x1) / 2.0, 3), round((y0 + y1) / 2.0, 3)],
        "bbox_xyxy": bbox,
        "source": "template_match",
        "scores": {letter: round(score, 3) for letter, score in scores.items()},
    }
    if result["label"] is None:
        if not plausible_shape:
            result["reason"] = "structural_mismatch"
        else:
            result["reason"] = "ambiguous_shape" if confidence >= min_confidence else "low_confidence"
    return result
