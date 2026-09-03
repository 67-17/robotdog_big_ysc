#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仪表盘状态识别核心算法。"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


STATUS_TO_COLOR = {
    "偏低": (255, 215, 0),
    "正常": (0, 160, 60),
    "偏高": (220, 30, 30),
    "未知": (60, 120, 255),
}


def import_cv2_optional():
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def white_balance_rgb_image(rgb_image):
    """用高亮中性色估计通道增益，减轻暖光和冷光造成的色相偏移。"""
    rgb_array = np.asarray(rgb_image.convert("RGB"), dtype=np.float32)
    high = np.percentile(rgb_array.reshape(-1, 3), 90, axis=0)
    target = float(np.max(high))
    if target <= 1.0:
        return rgb_image.convert("RGB")
    gains = np.clip(target / np.maximum(high, 1.0), 0.55, 2.80)
    balanced = np.clip(rgb_array * gains[None, None, :], 0, 255).astype(np.uint8)
    return Image.fromarray(balanced, "RGB")


def classify_meter_hsv(hue, sat, val):
    """把色环像素归类为红、黄、绿三类。"""
    if sat < 65 or val < 35:
        return None
    if hue <= 12 or hue >= 245:
        return "red"
    if 18 <= hue <= 55:
        return "yellow"
    if 50 <= hue <= 135:
        return "green"
    return None


def classify_meter_hsv_relaxed(hue, sat, val):
    """低饱和度现场图像的色环存在性分类，不用于最终状态判定。"""
    if val < 35:
        return None
    if (hue <= 12 or hue >= 245) and sat >= 25:
        return "red"
    if 18 <= hue <= 55 and sat >= 20:
        return "yellow"
    if 50 <= hue <= 135 and sat >= 15:
        return "green"
    return None


def summarize_color_runs(angle_labels):
    """统计红黄绿在圆环角度方向上的连续弧段长度和切换次数。"""
    labels = [label for label in angle_labels if label]
    if not labels:
        return {
            "max_run": {"red": 0, "yellow": 0, "green": 0},
            "transitions": 0,
        }

    runs = []
    current = labels[0]
    length = 1
    for label in labels[1:]:
        if label == current:
            length += 1
        else:
            runs.append((current, length))
            current = label
            length = 1
    runs.append((current, length))

    if len(runs) > 1 and runs[0][0] == runs[-1][0]:
        runs = [(runs[0][0], runs[0][1] + runs[-1][1])] + runs[1:-1]

    max_run = {"red": 0, "yellow": 0, "green": 0}
    for color, run_length in runs:
        max_run[color] = max(max_run[color], run_length)

    return {
        "max_run": max_run,
        "transitions": max(0, len(runs) - 1),
    }


def normalize_circle_hint(center_hint, radius_hint, image_shape):
    """检查外部提供的表盘圆心和半径是否可用。"""
    if center_hint is None or radius_hint is None:
        return None, None

    h, w = image_shape[:2]
    cx = float(center_hint[0])
    cy = float(center_hint[1])
    radius = float(radius_hint)
    if not math.isfinite(cx) or not math.isfinite(cy) or not math.isfinite(radius):
        return None, None
    if radius <= 4.0:
        return None, None
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return None, None
    return (cx, cy), radius


def estimate_center_and_radius(rgb_array):
    """在没有先验圆信息时，用非白色区域粗略估计表盘位置。"""
    non_white = rgb_array.mean(axis=2) < 245
    ys, xs = np.where(non_white)
    if len(xs) == 0 or len(ys) == 0:
        h, w = rgb_array.shape[:2]
        return (w / 2.0, h / 2.0), min(w, h) * 0.45

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    radius = 0.5 * min(xmax - xmin, ymax - ymin)
    return (cx, cy), radius


def _fit_circle_to_color_ring(rgb_image):
    """在无OpenCV环境中从三色色环像素拟合圆，供测试和后备使用。"""
    hsv = np.asarray(rgb_image.convert("HSV"), dtype=np.uint8)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    red = ((hue <= 12) | (hue >= 245)) & (sat >= 65) & (val >= 35)
    yellow = (hue >= 18) & (hue <= 55) & (sat >= 65) & (val >= 35)
    green = (hue >= 50) & (hue <= 135) & (sat >= 65) & (val >= 35)
    color_mask = red | yellow | green
    ys, xs = np.where(color_mask)
    if len(xs) < 60:
        return None

    step = max(1, len(xs) // 4000)
    x = xs[::step].astype(np.float64)
    y = ys[::step].astype(np.float64)
    design = np.column_stack((x, y, np.ones_like(x)))
    target = -(x * x + y * y)
    try:
        d, e, f = np.linalg.lstsq(design, target, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    cx = -d / 2.0
    cy = -e / 2.0
    radius_squared = cx * cx + cy * cy - f
    if radius_squared <= 0:
        return None
    color_radius = math.sqrt(radius_squared)
    outer_radius = color_radius / 0.84
    height, width = color_mask.shape
    if not (0 <= cx < width and 0 <= cy < height):
        return None
    if outer_radius < min(width, height) * 0.05 or outer_radius > min(width, height) * 0.55:
        return None
    return (float(cx), float(cy)), float(outer_radius)


def _hough_circle_candidates(rgb_image):
    cv2 = import_cv2_optional()
    if cv2 is None:
        return []

    rgb_array = np.asarray(rgb_image, dtype=np.uint8)
    height, width = rgb_array.shape[:2]
    scale = min(1.0, 800.0 / max(width, height))
    if scale < 1.0:
        resized = cv2.resize(
            rgb_array,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = rgb_array

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    short_side = min(gray.shape[:2])
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(30, short_side // 4),
        param1=110,
        param2=30,
        minRadius=max(18, int(short_side * 0.05)),
        maxRadius=max(24, int(short_side * 0.48)),
    )
    if circles is None:
        return []

    inverse_scale = 1.0 / scale
    return [
        (
            (float(cx) * inverse_scale, float(cy) * inverse_scale),
            float(radius) * inverse_scale,
        )
        for cx, cy, radius in circles[0]
    ]


def score_meter_candidate(ring_stats, radius):
    """优先选择红黄绿弧段连续、颜色切换较少的圆盘候选。"""
    return (
        ring_stats["total"]
        + sum(ring_stats["max_run"].values()) * 4
        + radius * 5
        - ring_stats["transitions"] * 25
    )


def locate_meter_circle(
    rgb_image,
    *,
    expected_center=None,
    min_radius=None,
    max_radius=None,
    max_center_distance=None,
):
    """定位画面中具有可靠红黄绿连续色环的仪表盘。"""
    rgb_image = white_balance_rgb_image(rgb_image)
    candidates = _hough_circle_candidates(rgb_image)
    fitted = _fit_circle_to_color_ring(rgb_image)
    if fitted is not None:
        candidates.append(fitted)

    if expected_center is not None:
        expected_x, expected_y = [float(value) for value in expected_center]
    else:
        expected_x = expected_y = None
    filtered_candidates = []
    for center, radius in candidates:
        radius = float(radius)
        if min_radius is not None and radius < float(min_radius):
            continue
        if max_radius is not None and radius > float(max_radius):
            continue
        if expected_x is not None and max_center_distance is not None:
            if math.hypot(center[0] - expected_x, center[1] - expected_y) > float(
                max_center_distance
            ):
                continue
        filtered_candidates.append((center, radius))
    candidates = filtered_candidates

    valid_candidates = collect_valid_meter_candidates(
        rgb_image,
        candidates,
        classifier=classify_meter_hsv,
    )
    if not valid_candidates:
        valid_candidates = collect_valid_meter_candidates(
            rgb_image,
            candidates,
            classifier=classify_meter_hsv_relaxed,
        )

    if not valid_candidates:
        return None
    return max(valid_candidates, key=lambda item: item["score"])


def collect_valid_meter_candidates(rgb_image, candidates, classifier):
    valid_candidates = []
    for center, radius in candidates:
        ring_stats = measure_ring_color_presence(
            rgb_image,
            center,
            radius,
            classifier=classifier,
        )
        if not has_valid_meter_ring(ring_stats, radius):
            continue
        score = score_meter_candidate(ring_stats, radius)
        valid_candidates.append(
            {
                "center": center,
                "radius": radius,
                "ring_stats": ring_stats,
                "score": float(score),
            }
        )
    return valid_candidates


def build_dark_mask(rgb_array, center, radius):
    """保留表盘内部较暗的像素，作为指针候选区域。"""
    cx, cy = center
    h, w = rgb_array.shape[:2]
    yy, xx = np.indices((h, w))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    gray = rgb_array.mean(axis=2)
    return (gray < 150) & (dist > radius * 0.08) & (dist < radius * 0.85)


def normalize_pointer_luminance(rgb_array):
    """把欠曝图的高亮背景拉回稳定范围，仅供黑色指针检测使用。"""
    gray = rgb_array.mean(axis=2)
    high = float(np.percentile(gray, 95))
    if high <= 1.0:
        return rgb_array
    scale = min(3.0, max(0.85, 235.0 / high))
    return np.clip(rgb_array.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def measure_pointer_segment_support(
    rgb_array,
    start_point,
    end_point,
    sample_count=60,
):
    """Measure dark-pixel continuity along an observed pointer segment."""
    cv2 = import_cv2_optional()
    if cv2 is not None:
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = rgb_array.mean(axis=2)

    x0, y0 = start_point
    x1, y1 = end_point
    height, width = gray.shape[:2]
    hits = []
    for x, y in zip(
        np.linspace(x0, x1, sample_count),
        np.linspace(y0, y1, sample_count),
    ):
        px = int(round(x))
        py = int(round(y))
        if px < 1 or px >= width - 1 or py < 1 or py >= height - 1:
            hits.append(False)
            continue
        hits.append(bool(np.min(gray[py - 1 : py + 2, px - 1 : px + 2]) < 150))

    longest_run = 0
    current_run = 0
    for hit in hits:
        current_run = current_run + 1 if hit else 0
        longest_run = max(longest_run, current_run)
    return {
        "hit_ratio": float(sum(hits)) / len(hits),
        "longest_run_ratio": float(longest_run) / len(hits),
    }


def detect_pointer_line(rgb_array, center, radius):
    """Locate a long radial line while tolerating perspective-shifted hubs."""
    cv2 = import_cv2_optional()
    if cv2 is None:
        return None

    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)

    cx, cy = center
    h, w = gray.shape[:2]
    yy, xx = np.indices((h, w))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    circle_mask = ((dist > radius * 0.06) & (dist < radius * 0.82)).astype(np.uint8) * 255
    edges = cv2.bitwise_and(edges, circle_mask)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(18, int(radius * 0.16)),
        minLineLength=max(16, int(radius * 0.20)),
        maxLineGap=max(6, int(radius * 0.08)),
    )
    if lines is None:
        return None

    best_line = None
    best_score = None
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [float(v) for v in line]
        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)
        near_d = min(d1, d2)
        far_d = max(d1, d2)
        # Under an oblique camera view the printed circle center and physical
        # pointer hub can differ noticeably. The old 0.26 limit rejected the
        # real pointer in field images and then fell back to dial text/ticks.
        if near_d > radius * 0.42:
            continue
        if far_d < radius * 0.28 or far_d > radius * 0.92:
            continue

        if d1 <= d2:
            near_x, near_y, far_x, far_y = x1, y1, x2, y2
        else:
            near_x, near_y, far_x, far_y = x2, y2, x1, y1

        length = math.hypot(far_x - near_x, far_y - near_y)
        if length < radius * 0.28:
            continue

        line_angle = math.atan2(far_y - near_y, far_x - near_x)
        radial_angle = math.atan2(far_y - cy, far_x - cx)
        alignment_deg = math.degrees(_circular_distance(line_angle, radial_angle))
        if alignment_deg > 35.0:
            continue

        support = measure_pointer_segment_support(
            rgb_array,
            (near_x, near_y),
            (far_x, far_y),
        )
        segment_dx = far_x - near_x
        segment_dy = far_y - near_y
        segment_length_squared = segment_dx * segment_dx + segment_dy * segment_dy
        projection_t = (
            ((cx - near_x) * segment_dx + (cy - near_y) * segment_dy)
            / segment_length_squared
        )
        projection_t = max(0.0, min(1.0, projection_t))
        sampling_origin = (
            near_x + projection_t * segment_dx,
            near_y + projection_t * segment_dy,
        )
        score = (
            length * 4.0
            + support["hit_ratio"] * radius * 4.0
            + support["longest_run_ratio"] * radius * 3.0
            - near_d * 1.5
            - alignment_deg * 0.5
        )
        if best_line is None or score > best_score:
            best_line = {
                "origin_point": (near_x, near_y),
                "line_tip_point": (far_x, far_y),
                "sampling_origin": sampling_origin,
                "angle": line_angle,
                "pointer_support": support,
                "center_offset_ratio": near_d / radius,
                "line_length_ratio": length / radius,
                "alignment_deg": alignment_deg,
            }
            best_score = score

    return best_line


def detect_pointer_tip_from_line(rgb_array, center, radius):
    """Return the far endpoint of the best Hough pointer line."""
    pointer_line = detect_pointer_line(rgb_array, center, radius)
    if pointer_line is None:
        return None
    return pointer_line["line_tip_point"]


def detect_pointer_tip(rgb_array, center, radius, angle_steps=240, radius_steps=80):
    """直线法失败时，沿多个角度扫描暗色像素寻找指针方向。"""
    line_tip = detect_pointer_tip_from_line(rgb_array, center, radius)
    if line_tip is not None:
        return (float(line_tip[0]), float(line_tip[1]))

    dark_mask = build_dark_mask(rgb_array, center, radius)
    cx, cy = center
    height, width = dark_mask.shape

    best_score = -1.0
    best_angle = 0.0
    best_tip_radius = radius * 0.45

    for angle in np.linspace(-math.pi, math.pi, angle_steps, endpoint=False):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        total_hits = 0
        longest_run = 0
        current_run = 0
        tip_radius = radius * 0.20

        for r in np.linspace(radius * 0.12, radius * 0.82, radius_steps):
            x = int(round(cx + cos_a * r))
            y = int(round(cy + sin_a * r))
            if x < 1 or x >= width - 1 or y < 1 or y >= height - 1:
                continue

            neighborhood = dark_mask[y - 1 : y + 2, x - 1 : x + 2]
            is_dark = int(neighborhood.sum()) >= 3
            if is_dark:
                total_hits += 1
                current_run += 1
                longest_run = max(longest_run, current_run)
                tip_radius = r
            else:
                current_run = 0

        score = longest_run * 6 + total_hits + tip_radius * 0.02
        if score > best_score:
            best_score = score
            best_angle = angle
            best_tip_radius = tip_radius

    tip_x = cx + math.cos(best_angle) * best_tip_radius
    tip_y = cy + math.sin(best_angle) * best_tip_radius
    return (float(tip_x), float(tip_y))


def measure_pointer_support(rgb_array, center, radius, tip_point, sample_count=60):
    """衡量候选指针是否从圆心附近连续延伸，排除刻度线和眩光伪方向。"""
    dark_mask = build_dark_mask(rgb_array, center, radius)
    cx, cy = center
    tx, ty = tip_point
    angle = math.atan2(ty - cy, tx - cx)
    hits = []
    height, width = dark_mask.shape

    for sample_radius in np.linspace(radius * 0.08, radius * 0.78, sample_count):
        x = int(round(cx + math.cos(angle) * sample_radius))
        y = int(round(cy + math.sin(angle) * sample_radius))
        if x < 1 or x >= width - 1 or y < 1 or y >= height - 1:
            hits.append(False)
            continue
        hits.append(int(dark_mask[y - 1 : y + 2, x - 1 : x + 2].sum()) >= 3)

    longest_run = 0
    current_run = 0
    for hit in hits:
        current_run = current_run + 1 if hit else 0
        longest_run = max(longest_run, current_run)
    return {
        "hit_ratio": float(sum(hits)) / len(hits),
        "longest_run_ratio": float(longest_run) / len(hits),
    }


def _sample_color_status(
    rgb_image,
    center,
    tip_point,
    radius,
    classifier,
    angle_offsets,
    radius_values,
):
    hsv = np.array(rgb_image.convert("HSV"))
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    red_count = 0
    yellow_count = 0
    green_count = 0

    cx, cy = center
    tx, ty = tip_point
    angle = math.atan2(ty - cy, tx - cx)
    for angle_offset in angle_offsets:
        sample_angle = angle + angle_offset
        for r in radius_values:
            x = int(round(cx + math.cos(sample_angle) * r))
            y = int(round(cy + math.sin(sample_angle) * r))
            if y < 0 or y >= hsv.shape[0] or x < 0 or x >= hsv.shape[1]:
                continue

            hue = int(h_channel[y, x])
            sat = int(s_channel[y, x])
            val = int(v_channel[y, x])
            color = classifier(hue, sat, val)
            if color == "red":
                red_count += 1
            elif color == "yellow":
                yellow_count += 1
            elif color == "green":
                green_count += 1

    counts = {
        "偏高": red_count,
        "偏低": yellow_count,
        "正常": green_count,
    }
    status = max(counts, key=counts.get)
    if counts[status] == 0:
        status = "未知"
    return status, counts


def sample_color_status(rgb_image, center, tip_point, radius):
    """沿指针末端方向采样色环颜色，判断偏低、正常或偏高。"""
    # 在指针方向附近的小扇形内投票，比单条射线更耐圆心误差和过曝。
    return _sample_color_status(
        rgb_image,
        center,
        tip_point,
        radius,
        classify_meter_hsv,
        np.linspace(-0.05, 0.05, 7),
        np.linspace(radius * 0.65, radius * 0.90, 30),
    )


def sample_relaxed_pointer_ring_status(rgb_image, center, tip_point, radius):
    """严格采样无票时，用更靠外的低饱和色环采样兜底。"""
    return _sample_color_status(
        rgb_image,
        center,
        tip_point,
        radius,
        classify_meter_hsv_relaxed,
        np.linspace(-0.08, 0.08, 9),
        np.linspace(radius * 0.72, radius * 0.96, 24),
    )


def _circular_distance(angle_a, angle_b):
    return abs(math.atan2(math.sin(angle_a - angle_b), math.cos(angle_a - angle_b)))


def measure_ring_color_angles(rgb_image, center, radius):
    """估计红黄绿弧段的圆周中心角，用于抵抗斜视造成的圆形压缩。"""
    hsv = np.asarray(rgb_image.convert("HSV"), dtype=np.uint8)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    height, width = hue.shape
    yy, xx = np.indices((height, width))
    cx, cy = center
    distance = np.hypot(xx - cx, yy - cy)
    annulus = (distance >= radius * 0.65) & (distance <= radius * 0.96)
    valid_color = (sat >= 65) & (val >= 35) & annulus

    red = ((hue <= 12) | (hue >= 245)) & valid_color
    yellow = (hue >= 18) & (hue <= 55) & valid_color
    green = (hue >= 50) & (hue <= 135) & valid_color & ~yellow
    masks = {"red": red, "yellow": yellow, "green": green}

    result = {}
    for color, mask in masks.items():
        if int(mask.sum()) < 10:
            continue
        angles = np.arctan2(yy[mask] - cy, xx[mask] - cx)
        sin_sum = float(np.sin(angles).sum())
        cos_sum = float(np.cos(angles).sum())
        result[color] = math.atan2(sin_sum, cos_sum)
    return result


def classify_status_from_ring_angles(
    rgb_image,
    center,
    tip_point,
    radius,
    boundary_margin_deg=8.0,
):
    """按指针角度和色环弧段中心角分类，边界证据不足时返回未知。"""
    color_angles = measure_ring_color_angles(rgb_image, center, radius)
    if set(color_angles) != {"red", "yellow", "green"}:
        return "未知", color_angles

    cx, cy = center
    tx, ty = tip_point
    pointer_angle = math.atan2(ty - cy, tx - cx)
    ranked = sorted(
        (
            (_circular_distance(pointer_angle, color_angle), color)
            for color, color_angle in color_angles.items()
        ),
        key=lambda item: item[0],
    )
    margin = ranked[1][0] - ranked[0][0]
    if math.degrees(margin) < boundary_margin_deg:
        return "未知", color_angles

    status_by_color = {
        "red": "偏高",
        "yellow": "偏低",
        "green": "正常",
    }
    return status_by_color[ranked[0][1]], color_angles


def classify_status_from_red_reference(
    pointer_angle,
    red_angle,
    high_half_width_deg=40.0,
    boundary_margin_deg=8.0,
):
    """Classify the fixed dial layout relative to its detected red sector."""
    relative_deg = math.degrees(
        math.atan2(
            math.sin(pointer_angle - red_angle),
            math.cos(pointer_angle - red_angle),
        )
    )
    absolute_deg = abs(relative_deg)
    near_high_boundary = (
        abs(absolute_deg - high_half_width_deg) < boundary_margin_deg
    )
    near_left_boundary = (180.0 - absolute_deg) < boundary_margin_deg
    if near_high_boundary or near_left_boundary:
        return "未知", relative_deg
    if absolute_deg < high_half_width_deg:
        return "偏高", relative_deg
    if relative_deg < 0.0:
        return "正常", relative_deg
    return "偏低", relative_deg


def classify_status_from_pointer_geometry(rgb_image, center, tip_point, radius):
    """Use the stable red sector as the angular reference for this dial type."""
    color_angles = measure_ring_color_angles(rgb_image, center, radius)
    red_angle = color_angles.get("red")
    if red_angle is None:
        return "未知", {
            "color_angles": color_angles,
            "pointer_relative_to_red_deg": None,
        }

    cx, cy = center
    tx, ty = tip_point
    pointer_angle = math.atan2(ty - cy, tx - cx)
    status, relative_deg = classify_status_from_red_reference(
        pointer_angle,
        red_angle,
    )
    return status, {
        "color_angles": color_angles,
        "pointer_relative_to_red_deg": relative_deg,
        "red_reference_angle_deg": math.degrees(red_angle),
    }


def measure_ring_color_presence(rgb_image, center, radius, classifier=classify_meter_hsv):
    """统计表盘外圈的红黄绿覆盖情况，用于判断是否真的存在三色表盘。"""
    hsv = np.array(rgb_image.convert("HSV"))
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    red_count = 0
    yellow_count = 0
    green_count = 0
    valid = 0
    angle_labels = []

    cx, cy = center
    for angle in np.linspace(-math.pi, math.pi, 180, endpoint=False):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        local_counts = {"red": 0, "yellow": 0, "green": 0}
        for r in np.linspace(radius * 0.72, radius * 0.94, 10):
            x = int(round(cx + cos_a * r))
            y = int(round(cy + sin_a * r))
            if y < 0 or y >= hsv.shape[0] or x < 0 or x >= hsv.shape[1]:
                continue

            valid += 1
            hue = int(h_channel[y, x])
            sat = int(s_channel[y, x])
            val = int(v_channel[y, x])
            color = classifier(hue, sat, val)
            if color == "red":
                red_count += 1
                local_counts["red"] += 1
            elif color == "yellow":
                yellow_count += 1
                local_counts["yellow"] += 1
            elif color == "green":
                green_count += 1
                local_counts["green"] += 1

        dominant = None
        dominant_hits = 0
        for color, hits in local_counts.items():
            if hits > dominant_hits:
                dominant = color
                dominant_hits = hits
        angle_labels.append(dominant if dominant_hits >= 2 else None)

    total = red_count + yellow_count + green_count
    present = int(red_count > 0) + int(yellow_count > 0) + int(green_count > 0)
    ratio = (total / float(valid)) if valid else 0.0
    run_stats = summarize_color_runs(angle_labels)
    return {
        "red": red_count,
        "yellow": yellow_count,
        "green": green_count,
        "total": total,
        "present": present,
        "ratio": ratio,
        "max_run": run_stats["max_run"],
        "transitions": run_stats["transitions"],
    }


def has_valid_meter_ring(ring_stats, radius):
    """检查色环是否同时满足红黄绿覆盖和最小连续弧段阈值。"""
    min_single_color = max(10, int(round(radius * 0.10)))
    min_total = max(60, int(round(radius * 0.55)))
    min_run = max(4, int(round(radius * 0.015)))
    return (
        ring_stats["present"] == 3
        and ring_stats["red"] >= min_single_color
        and ring_stats["yellow"] >= min_single_color
        and ring_stats["green"] >= min_single_color
        and ring_stats["total"] >= min_total
        and ring_stats["ratio"] >= 0.08
        and ring_stats["max_run"]["red"] >= min_run
        and ring_stats["max_run"]["yellow"] >= min_run
        and ring_stats["max_run"]["green"] >= min_run
        and ring_stats["transitions"] <= 48
    )


def build_unknown_result(rgb_image, center, radius, ring_stats=None):
    """统一返回“未检测到可靠表盘”的结果结构。"""
    return {
        "status": "未知",
        "center": center,
        "radius": radius,
        "tip_point": center,
        "color_counts": {"偏高": 0, "偏低": 0, "正常": 0},
        "image": rgb_image,
        "meter_found": False,
        "pointer_found": False,
        "pointer_support": None,
        "ring_stats": ring_stats,
    }


def load_font(size=30):
    candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def analyze_meter_rgb_image(
    rgb_image,
    angle_steps=240,
    radius_steps=80,
    center_hint=None,
    radius_hint=None,
):
    """完成单张 RGB 图像的表盘识别并返回结构化结果。"""
    rgb_image = rgb_image.convert("RGB")
    rgb_array = np.array(rgb_image)
    color_image = white_balance_rgb_image(rgb_image)

    center, radius = normalize_circle_hint(center_hint, radius_hint, rgb_array.shape)
    if center is None or radius is None:
        located = locate_meter_circle(color_image)
        if located is not None:
            center, radius = located["center"], located["radius"]
        else:
            center, radius = estimate_center_and_radius(rgb_array)
    ring_stats = measure_ring_color_presence(color_image, center, radius)
    if not has_valid_meter_ring(ring_stats, radius):
        relaxed_ring_stats = measure_ring_color_presence(
            color_image,
            center,
            radius,
            classifier=classify_meter_hsv_relaxed,
        )
        if not has_valid_meter_ring(relaxed_ring_stats, radius):
            return build_unknown_result(rgb_image, center, radius, ring_stats)
        ring_stats = relaxed_ring_stats

    pointer_array = normalize_pointer_luminance(rgb_array)
    pointer_line = detect_pointer_line(pointer_array, center, radius)
    if pointer_line is not None:
        pointer_angle = float(pointer_line["angle"])
        status_center = tuple(pointer_line["sampling_origin"])
        tip_point = (
            status_center[0] + math.cos(pointer_angle) * radius * 0.78,
            status_center[1] + math.sin(pointer_angle) * radius * 0.78,
        )
        pointer_support = dict(pointer_line["pointer_support"])
        pointer_method = "hough_line"
    else:
        status_center = center
        tip_point = detect_pointer_tip(
            pointer_array,
            center,
            radius,
            angle_steps,
            radius_steps,
        )
        pointer_support = measure_pointer_support(
            pointer_array,
            center,
            radius,
            tip_point,
        )
        pointer_method = "radial_scan"
    if pointer_support["longest_run_ratio"] < 0.18:
        return {
            "status": "未知",
            "center": center,
            "radius": radius,
            "tip_point": tip_point,
            "color_counts": {"偏高": 0, "偏低": 0, "正常": 0},
            "image": rgb_image,
            "meter_found": True,
            "pointer_found": False,
            "pointer_support": pointer_support,
            "pointer_method": pointer_method,
            "pointer_line": pointer_line,
            "ring_stats": ring_stats,
        }

    ray_status, counts = sample_color_status(
        color_image,
        status_center,
        tip_point,
        radius,
    )
    geometry_status, geometry_evidence = classify_status_from_pointer_geometry(
        color_image,
        status_center,
        tip_point,
        radius,
    )
    color_angles = geometry_evidence["color_angles"]
    ranked_counts = sorted(counts.values(), reverse=True)
    ray_is_strong = (
        ray_status != "未知"
        and ranked_counts[0] >= 10
        and ranked_counts[0] >= max(1, ranked_counts[1]) * 2
    )
    relaxed_status = "未知"
    relaxed_counts = {"偏高": 0, "偏低": 0, "正常": 0}
    relaxed_is_strong = False
    if not ray_is_strong:
        relaxed_status, relaxed_counts = sample_relaxed_pointer_ring_status(
            color_image,
            status_center,
            tip_point,
            radius,
        )
        ranked_relaxed_counts = sorted(relaxed_counts.values(), reverse=True)
        relaxed_is_strong = (
            relaxed_status != "未知"
            and ranked_relaxed_counts[0] >= 4
            and ranked_relaxed_counts[0] >= max(1, ranked_relaxed_counts[1]) * 2
        )

    supported_ray_status = (
        ray_status
        if ray_is_strong
        else relaxed_status if relaxed_is_strong else "未知"
    )
    ray_agreement = not (
        geometry_status != "未知"
        and supported_ray_status != "未知"
        and geometry_status != supported_ray_status
    )
    if geometry_status != "未知":
        status = geometry_status
        status_source = "red_reference_geometry"
    elif supported_ray_status != "未知":
        status = supported_ray_status
        status_source = "ray_color_fallback"
    else:
        status = "未知"
        status_source = "unsupported"
    status_supported = status != "未知"
    status_agreement = status_supported and ray_agreement

    return {
        "status": status,
        "center": center,
        "radius": radius,
        "tip_point": tip_point,
        "color_counts": counts,
        "image": rgb_image,
        "meter_found": True,
        "pointer_found": True,
        "pointer_support": pointer_support,
        "pointer_method": pointer_method,
        "pointer_line": pointer_line,
        "ring_stats": ring_stats,
        "status_evidence": {
            "ray_status": ray_status,
            "ring_status": geometry_status,
            "geometry_status": geometry_status,
            "relaxed_status": relaxed_status,
            "status_agreement": status_agreement,
            "status_supported": status_supported,
            "status_source": status_source,
            "ray_agreement": ray_agreement,
            "pointer_relative_to_red_deg": geometry_evidence.get(
                "pointer_relative_to_red_deg"
            ),
            "red_reference_angle_deg": geometry_evidence.get(
                "red_reference_angle_deg"
            ),
            "relaxed_color_counts": relaxed_counts,
        },
        "ring_color_angles_deg": {
            color: math.degrees(angle) for color, angle in color_angles.items()
        },
    }


def analyze_meter_rgb_array(
    rgb_array,
    angle_steps=240,
    radius_steps=80,
    center_hint=None,
    radius_hint=None,
):
    rgb_image = Image.fromarray(rgb_array.astype(np.uint8), "RGB")
    return analyze_meter_rgb_image(
        rgb_image,
        angle_steps,
        radius_steps,
        center_hint=center_hint,
        radius_hint=radius_hint,
    )


def scale_result(result, scale):
    """把缩放图上的识别结果换算回原图坐标。"""
    scaled = dict(result)
    cx, cy = result["center"]
    tx, ty = result["tip_point"]
    scaled["center"] = (cx * scale, cy * scale)
    scaled["tip_point"] = (tx * scale, ty * scale)
    scaled["radius"] = result["radius"] * scale
    return scaled


def draw_result(rgb_image, result):
    """在原图上绘制表盘、指针和识别文字。"""
    annotated = rgb_image.copy()
    draw = ImageDraw.Draw(annotated)
    font = load_font(34)

    cx, cy = result["center"]
    tx, ty = result["tip_point"]
    radius = result["radius"]
    status = result["status"]
    color = STATUS_TO_COLOR[status]
    meter_found = result.get("meter_found", True)

    if meter_found and radius > 0:
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(0, 120, 255))
        draw.line((cx, cy, tx, ty), fill=color, width=5)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(0, 120, 255), width=3)
    draw.text((30, 24), f"仪表盘状态：{status}", fill=color, font=font)
    return annotated
