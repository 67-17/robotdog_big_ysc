#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从摄像头读取画面并实时调用仪表盘识别算法。"""

import argparse
import math
import sys
import time
from collections import deque

import numpy as np
from PIL import Image

try:
    from .meter_status_recognition import analyze_meter_rgb_array, draw_result, scale_result
except ImportError:
    from meter_status_recognition import analyze_meter_rgb_array, draw_result, scale_result


def import_cv2():
    try:
        import cv2
    except ImportError:
        print("缺少 OpenCV，请先安装 python3-opencv。", flush=True)
        raise SystemExit(1)
    return cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="0", help="摄像头编号或设备路径，默认 0")
    parser.add_argument("--width", type=int, default=640, help="摄像头宽度")
    parser.add_argument("--height", type=int, default=480, help="摄像头高度")
    parser.add_argument("--format", choices=["auto", "YUYV", "MJPG", "DEFAULT"], default="auto", help="摄像头像素格式")
    parser.add_argument("--process-width", type=int, default=320, help="识别时缩放到的宽度")
    parser.add_argument("--skip-frames", type=int, default=4, help="每隔多少帧识别一次")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="两次识别之间的最小间隔秒数")
    parser.add_argument("--no-window", action="store_true", help="只在终端输出，不显示窗口")
    return parser.parse_args()


def parse_camera_source(camera):
    return int(camera) if str(camera).isdigit() else camera


def try_read_frame(cap, attempts=8):
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return frame
    return None


def configure_capture(cv2, source, width, height, fourcc):
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, None

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame = try_read_frame(cap)
    if frame is None:
        cap.release()
        return None, None
    return cap, frame


def build_capture_configs(pixel_format, width, height):
    """按优先级生成多组摄像头格式与分辨率配置。"""
    if pixel_format == "YUYV":
        return [("YUYV", width, height), ("YUYV", 640, 480), ("YUYV", 320, 240)]
    if pixel_format == "MJPG":
        return [("MJPG", width, height), ("MJPG", 640, 480), ("MJPG", 320, 240)]
    if pixel_format == "DEFAULT":
        return [(None, width, height), (None, 640, 480), (None, 320, 240)]

    return [
        ("YUYV", width, height),
        (None, width, height),
        ("YUYV", 640, 480),
        (None, 640, 480),
        ("MJPG", 640, 480),
        ("YUYV", 320, 240),
        (None, 320, 240),
        ("MJPG", 320, 240),
    ]


def open_camera(cv2, camera, width, height, pixel_format):
    source = parse_camera_source(camera)
    configs = build_capture_configs(pixel_format, width, height)

    for fourcc, cfg_width, cfg_height in configs:
        label = fourcc if fourcc else "默认格式"
        print(f"尝试打开摄像头 {camera}：{label} {cfg_width}x{cfg_height}", flush=True)
        cap, frame = configure_capture(cv2, source, cfg_width, cfg_height, fourcc)
        if cap is not None:
            print(f"摄像头读取成功：{label} {cfg_width}x{cfg_height}", flush=True)
            return cap, frame

    return None, None


def pil_to_bgr(pil_image, cv2):
    rgb_array = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def smooth_status(history, current_status):
    """对最近几次有效结果投票，减小单帧识别抖动。"""
    history.append(current_status)
    valid = [item for item in history if item != "未知"]
    if not valid:
        return current_status

    counts = {}
    for item in valid:
        counts[item] = counts.get(item, 0) + 1
    return max(counts, key=counts.get)


def offset_result(result, dx, dy):
    shifted = dict(result)
    cx, cy = result["center"]
    tx, ty = result["tip_point"]
    shifted["center"] = (cx + dx, cy + dy)
    shifted["tip_point"] = (tx + dx, ty + dy)
    return shifted


def build_unknown_result_for_frame(frame_bgr):
    """整帧里没有找到可靠表盘时，直接返回未知状态且不绘制误检标记。"""
    h, w = frame_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    return {
        "status": "未知",
        "center": center,
        "radius": 0.0,
        "tip_point": center,
        "color_counts": {"偏高": 0, "偏低": 0, "正常": 0},
        "meter_found": False,
        "ring_stats": None,
    }


def configure_preview_window(cv2, frame_bgr):
    """按图像尺寸缩放预览窗口。"""
    window_name = "meter_status_recognition"
    height, width = frame_bgr.shape[:2]
    max_width = 1280
    max_height = 960
    scale = min(max_width / float(width), max_height / float(height))
    scale = max(1.4, min(scale, 2.0))
    window_width = max(1, int(round(width * scale)))
    window_height = max(1, int(round(height * scale)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)
    return window_name


def fit_circle_from_points(points):
    """用最小二乘法拟合彩色弧线所在圆。"""
    if points.shape[0] < 80:
        return None

    if points.shape[0] > 2400:
        step = max(1, points.shape[0] // 2400)
        points = points[::step]

    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    a = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)

    try:
        coeffs, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    cx = -coeffs[0] / 2.0
    cy = -coeffs[1] / 2.0
    radius_sq = cx * cx + cy * cy - coeffs[2]
    if not math.isfinite(radius_sq) or radius_sq <= 0:
        return None

    radius = math.sqrt(radius_sq)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    residual = np.abs(dist - radius)
    median_residual = float(np.median(residual))

    # 用角度覆盖率过滤局部噪声形成的伪圆。
    angles = np.arctan2(y - cy, x - cx)
    bins = np.zeros(36, dtype=bool)
    angle_ids = ((angles + math.pi) / (2.0 * math.pi) * bins.size).astype(int) % bins.size
    bins[angle_ids] = True
    coverage = float(bins.mean())

    return {
        "center": (float(cx), float(cy)),
        "radius": float(radius),
        "median_residual": median_residual,
        "coverage": coverage,
        "count": int(points.shape[0]),
    }


def classify_meter_hsv(hue, sat, val):
    """把色环像素归类为红、黄、绿三类。"""
    if sat < 60 or val < 60:
        return None
    if hue <= 12 or hue >= 165:
        return "red"
    if 15 <= hue <= 40:
        return "yellow"
    if 45 <= hue <= 95:
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


def score_circle_by_arc_colors(cv2, frame_bgr, center, radius):
    """统计圆环附近的红、黄、绿像素，给表盘候选打分。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    cx, cy = center
    h, w = hsv.shape[:2]

    red = 0
    yellow = 0
    green = 0
    valid = 0
    angle_labels = []

    for angle in np.linspace(-math.pi, math.pi, 180, endpoint=False):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        local_counts = {"red": 0, "yellow": 0, "green": 0}
        for r in np.linspace(radius * 0.72, radius * 0.94, 12):
            x = int(round(cx + cos_a * r))
            y = int(round(cy + sin_a * r))
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            valid += 1
            hue, sat, val = [int(v) for v in hsv[y, x]]
            color = classify_meter_hsv(hue, sat, val)
            if color == "red":
                red += 1
                local_counts["red"] += 1
            elif color == "yellow":
                yellow += 1
                local_counts["yellow"] += 1
            elif color == "green":
                green += 1
                local_counts["green"] += 1

        dominant = None
        dominant_hits = 0
        for color, hits in local_counts.items():
            if hits > dominant_hits:
                dominant = color
                dominant_hits = hits
        angle_labels.append(dominant if dominant_hits >= 2 else None)

    total = red + yellow + green
    present = int(red > 0) + int(yellow > 0) + int(green > 0)
    ratio = (total / float(valid)) if valid else 0.0
    run_stats = summarize_color_runs(angle_labels)
    return {
        "score": total + present * 80 + ratio * 400.0,
        "red": red,
        "yellow": yellow,
        "green": green,
        "total": total,
        "present": present,
        "ratio": ratio,
        "max_run": run_stats["max_run"],
        "transitions": run_stats["transitions"],
    }


def has_full_meter_arc(arc, radius):
    """赛题表盘固定有红黄绿三色，缺任一颜色就不接受这个圆候选。"""
    min_single_color = max(8, int(round(radius * 0.08)))
    min_total = max(60, int(round(radius * 0.55)))
    min_run = max(5, int(round(radius * 0.015)))
    return (
        arc["present"] == 3
        and arc["red"] >= min_single_color
        and arc["yellow"] >= min_single_color
        and arc["green"] >= min_single_color
        and arc["total"] >= min_total
        and arc["ratio"] >= 0.08
        and arc["max_run"]["red"] >= min_run
        and arc["max_run"]["yellow"] >= min_run
        and arc["max_run"]["green"] >= min_run
        and arc["transitions"] <= 12
    )


def detect_colored_arc_circle(cv2, frame_bgr):
    """根据彩色刻度弧线定位表盘外圈。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, (0, 70, 70), (12, 255, 255))
    red_mask |= cv2.inRange(hsv, (165, 70, 70), (179, 255, 255))
    yellow_mask = cv2.inRange(hsv, (15, 70, 70), (40, 255, 255))
    green_mask = cv2.inRange(hsv, (45, 70, 70), (95, 255, 255))
    mask = red_mask | yellow_mask | green_mask

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    ys, xs = np.where(mask > 0)
    if len(xs) < 80:
        return None

    circle = fit_circle_from_points(np.column_stack([xs, ys]))
    if circle is None:
        return None

    h, w = frame_bgr.shape[:2]
    cx, cy = circle["center"]
    radius = circle["radius"]
    if radius < min(h, w) * 0.10 or radius > min(h, w) * 0.48:
        return None
    if circle["median_residual"] > max(6.0, radius * 0.10):
        return None
    if circle["coverage"] < 0.18:
        return None
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return None

    arc = score_circle_by_arc_colors(cv2, frame_bgr, circle["center"], radius)
    if not has_full_meter_arc(arc, radius):
        return None

    frame_center = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    dist_to_center = float(np.linalg.norm(np.array([cx, cy]) - frame_center))
    score = circle["coverage"] * 260.0 + radius * 1.0 - dist_to_center * 0.30 + arc["score"]
    return {
        "center": (cx, cy),
        "radius": radius,
        "score": score,
    }


def detect_hough_circle(cv2, frame_bgr):
    """彩色弧线不足时，改用 Hough 圆检测作补充。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    h, w = gray.shape[:2]
    min_radius = max(20, int(min(h, w) * 0.10))
    max_radius = max(min_radius + 10, int(min(h, w) * 0.48))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, int(min(h, w) * 0.20)),
        param1=120,
        param2=36,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None

    frame_center = np.array([w / 2.0, h / 2.0], dtype=np.float64)
    best = None
    best_score = None
    for cx, cy, radius in circles[0]:
        arc = score_circle_by_arc_colors(cv2, frame_bgr, (float(cx), float(cy)), float(radius))
        if not has_full_meter_arc(arc, float(radius)):
            continue
        dist_to_center = float(np.linalg.norm(np.array([cx, cy]) - frame_center))
        score = arc["score"] + float(radius) * 0.8 - dist_to_center * 0.35
        if best is None or score > best_score:
            best = (float(cx), float(cy), float(radius))
            best_score = score

    if best is None:
        return None
    cx, cy, radius = best
    return {
        "center": (cx, cy),
        "radius": radius,
        "score": best_score,
    }


def locate_meter_roi(cv2, frame_bgr):
    """在整帧中裁出表盘区域，降低后续识别的干扰。"""
    h, w = frame_bgr.shape[:2]
    candidates = []

    colored_circle = detect_colored_arc_circle(cv2, frame_bgr)
    if colored_circle is not None:
        candidates.append(colored_circle)

    hough_circle = detect_hough_circle(cv2, frame_bgr)
    if hough_circle is not None:
        candidates.append(hough_circle)

    if not candidates:
        return frame_bgr, (0, 0), None

    best_circle = max(candidates, key=lambda item: item["score"])
    cx, cy = best_circle["center"]
    radius = best_circle["radius"]

    margin = max(12, int(round(radius * 0.24)))
    x0 = max(0, int(round(cx - radius - margin)))
    y0 = max(0, int(round(cy - radius - margin)))
    x1 = min(w, int(round(cx + radius + margin)))
    y1 = min(h, int(round(cy + radius + margin)))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return frame_bgr, (0, 0), None

    roi_bgr = frame_bgr[y0:y1, x0:x1].copy()
    circle_in_roi = {
        "center": (cx - x0, cy - y0),
        "radius": radius,
    }
    return roi_bgr, (x0, y0), circle_in_roi


def main():
    args = parse_args()
    cv2 = import_cv2()

    cap, first_frame = open_camera(cv2, args.camera, args.width, args.height, args.format)
    if cap is None:
        print(f"无法读取摄像头画面：{args.camera}", flush=True)
        print("请检查 VMware 摄像头连接，或更换 --camera 参数。", flush=True)
        return 1

    last_status = None
    last_result = None
    loop_index = 0
    frame_bgr = first_frame
    last_sample_time = 0.0
    status_history = deque(maxlen=5)
    window_name = None
    if not args.no_window:
        window_name = configure_preview_window(cv2, first_frame)
    print("摄像头识别已启动，按 q 退出。", flush=True)

    while True:
        roi_bgr, roi_origin, roi_circle = locate_meter_roi(cv2, frame_bgr)
        rgb_array = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        now = time.monotonic()
        # 通过跳帧和最小采样间隔控制识别频率。
        should_sample = (
            last_result is None
            or (
                loop_index % max(1, args.skip_frames) == 0
                and now - last_sample_time >= max(0.1, args.sample_interval)
            )
        )
        if should_sample:
            if roi_circle is None:
                status_history.clear()
                last_result = build_unknown_result_for_frame(frame_bgr)
                last_sample_time = now
            else:
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                h, w = roi_rgb.shape[:2]
                scale = 1.0
                work_array = roi_rgb
                center_hint = roi_circle["center"]
                radius_hint = roi_circle["radius"]
                if args.process_width > 0 and w > args.process_width:
                    scale = w / float(args.process_width)
                    work_height = max(1, int(round(h / scale)))
                    work_bgr = cv2.resize(roi_bgr, (args.process_width, work_height))
                    work_array = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)
                    center_hint = (center_hint[0] / scale, center_hint[1] / scale)
                    radius_hint = radius_hint / scale

                result = analyze_meter_rgb_array(
                    work_array,
                    angle_steps=160,
                    radius_steps=60,
                    center_hint=center_hint,
                    radius_hint=radius_hint,
                )
                last_result = offset_result(
                    scale_result(result, scale),
                    roi_origin[0],
                    roi_origin[1],
                )
                if last_result.get("meter_found", True):
                    last_result["status"] = smooth_status(status_history, last_result["status"])
                else:
                    status_history.clear()
                    last_result["status"] = "未知"
                last_sample_time = now

        annotated_pil = draw_result(Image.fromarray(rgb_array), last_result)
        annotated_bgr = pil_to_bgr(annotated_pil, cv2)

        status = last_result["status"]
        if status != last_status:
            print(f"仪表盘状态：{status}", flush=True)
            last_status = status

        if not args.no_window:
            cv2.imshow(window_name, annotated_bgr)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = 255

        if key == ord("q"):
            break

        ok, next_frame = cap.read()
        if ok and next_frame is not None and next_frame.size > 0:
            frame_bgr = next_frame
            loop_index += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
