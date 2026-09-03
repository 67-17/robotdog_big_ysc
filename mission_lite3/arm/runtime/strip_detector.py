from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from collections import deque
from datetime import datetime
from pathlib import Path
import time
from typing import NamedTuple
import sys

import cv2
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "strip_detector_config.json"
DEFAULT_CALIBRATION_PATH = MODULE_DIR / "camera_calibration.json"
WINDOW_NAME = "strip detector"
DISPLAY_MODES = ("annotated", "red_mask", "green_mask", "combined_mask")
METRIC_KEYS = ("fps", "latency_mean_ms", "latency_p95_ms", "cpu_percent")


def _load_strip_detection():
    try:
        import strip_detection as module
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location(
            "strip_detection", MODULE_DIR / "strip_detection.py"
        )
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules["strip_detection"] = module
        spec.loader.exec_module(module)
    return module


strip_detection = _load_strip_detection()


def _load_camera_calibration_module():
    try:
        import camera_calibration as module
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location(
            "camera_calibration", MODULE_DIR / "camera_calibration.py"
        )
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules["camera_calibration"] = module
        spec.loader.exec_module(module)
    return module


camera_calibration = _load_camera_calibration_module()


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_finite_number(value, name, *, minimum=None):
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return numeric


def _require_positive_number(value, name):
    numeric = _require_finite_number(value, name, minimum=0.0)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _require_integer(value, name, *, minimum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def positive_int(value):
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if isinstance(value, bool) or numeric <= 0 or str(numeric) != str(value):
        raise argparse.ArgumentTypeError("must be a positive integer")
    return numeric


def camera_device_arg(value):
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError(
            "device must be a non-empty path or non-negative integer"
        )
    if isinstance(value, int):
        if value < 0:
            raise argparse.ArgumentTypeError(
                "device must be a non-empty path or non-negative integer"
            )
        return int(value)
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError(
            "device must be a non-empty path or non-negative integer"
        )
    normalized = value.strip()
    if not normalized:
        raise argparse.ArgumentTypeError(
            "device must be a non-empty path or non-negative integer"
        )
    try:
        numeric = int(normalized)
    except ValueError:
        return normalized
    if numeric < 0:
        raise argparse.ArgumentTypeError(
            "device must be a non-empty path or non-negative integer"
        )
    return numeric


def _json_compact(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _timestamp_prefix(now=None):
    now = datetime.now() if now is None else now
    return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _colorize_mask(mask, color):
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    if color == "red":
        canvas[..., 2] = mask
    elif color == "green":
        canvas[..., 1] = mask
    else:
        canvas[:] = mask[..., None]
    return canvas


def _annotation_color(name):
    return {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
    }.get(name, (255, 255, 255))


def _draw_text_box(frame, lines, origin, color):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    line_gap = 4
    sizes = [
        cv2.getTextSize(line, font, scale, thickness)[0]
        for line in lines
    ]
    width = max((size[0] for size in sizes), default=0) + 8
    height = sum(size[1] for size in sizes) + line_gap * max(0, len(lines) - 1) + 8
    top = max(0, y - height)
    left = max(0, x)
    bottom = min(frame.shape[0] - 1, top + height)
    right = min(frame.shape[1] - 1, left + width)
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), -1)
    cursor_y = top + 16
    for line in lines:
        cv2.putText(
            frame,
            line,
            (left + 4, cursor_y),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        cursor_y += int(size[1] + line_gap) if (size := cv2.getTextSize(line, font, scale, thickness)[0]) else 0


def _format_detection_lines(strip):
    return [
        f"id={strip.track_id} color={strip.color}",
        (
            f"angle={strip.angle_deg:.1f} cont={strip.angle_unwrapped_deg:.1f} "
            f"angle_ok={int(strip.angle_reliable)} conf={strip.confidence:.2f}"
        ),
        (
            f"stable={int(strip.stable)} grasp={int(strip.grasp_candidate)} "
            f"size={strip.size_px[0]:.0f}x{strip.size_px[1]:.0f}"
        ),
    ]


def _annotate_frame(frame_bgr, detections, metrics_snapshot):
    annotated = frame_bgr.copy()
    for strip in detections:
        points = np.asarray(strip.box, dtype=np.int32).reshape((-1, 1, 2))
        color = _annotation_color(strip.color)
        cv2.polylines(annotated, [points], True, color, 2, cv2.LINE_AA)
        center = (
            int(round(strip.center_px[0])),
            int(round(strip.center_px[1])),
        )
        cv2.circle(annotated, center, 3, color, -1, cv2.LINE_AA)
        _draw_text_box(
            annotated,
            _format_detection_lines(strip),
            (center[0] + 8, center[1] - 8),
            color,
        )

    metrics_lines = [
        (
            f"FPS {metrics_snapshot['fps']:.1f} "
            f"P95 {metrics_snapshot['latency_p95_ms']:.1f}ms "
            f"CPU {metrics_snapshot['cpu_percent']:.1f}%"
        ),
        f"latency {metrics_snapshot['latency_mean_ms']:.1f}ms",
    ]
    _draw_text_box(annotated, metrics_lines, (10, 10), (255, 255, 255))
    return annotated


def select_display_frame(mode, annotated_frame, masks):
    if mode == "annotated":
        return annotated_frame
    if mode == "red_mask":
        return _colorize_mask(masks["red"], "red")
    if mode == "green_mask":
        return _colorize_mask(masks["green"], "green")
    if mode == "combined_mask":
        combined = cv2.bitwise_or(masks["red"], masks["green"])
        return _colorize_mask(combined, "gray")
    raise ValueError(f"unknown display mode: {mode}")


class _MetricSample(NamedTuple):
    end_wall_seconds: float
    wall_seconds: float
    process_seconds: float
    latency_ms: float


class RuntimeMetrics:
    def __init__(self, window_seconds=10):
        self._window_seconds = _require_positive_number(
            window_seconds, "window_seconds"
        )
        self._samples = deque()
        self._elapsed_wall_seconds = 0.0

    def _prune(self):
        cutoff = self._elapsed_wall_seconds - self._window_seconds
        while self._samples and self._samples[0].end_wall_seconds <= cutoff:
            self._samples.popleft()

    def add_sample(self, wall_seconds, process_seconds, latency_ms):
        wall_seconds = _require_positive_number(wall_seconds, "wall_seconds")
        process_seconds = _require_finite_number(
            process_seconds, "process_seconds", minimum=0.0
        )
        latency_ms = _require_finite_number(latency_ms, "latency_ms", minimum=0.0)
        self._elapsed_wall_seconds += wall_seconds
        self._samples.append(
            _MetricSample(
                end_wall_seconds=self._elapsed_wall_seconds,
                wall_seconds=wall_seconds,
                process_seconds=process_seconds,
                latency_ms=latency_ms,
            )
        )
        self._prune()

    def snapshot(self):
        self._prune()
        if not self._samples:
            return {
                "fps": 0.0,
                "latency_mean_ms": 0.0,
                "latency_p95_ms": 0.0,
                "cpu_percent": 0.0,
            }

        sample_count = len(self._samples)
        wall_total = sum(sample.wall_seconds for sample in self._samples)
        process_total = sum(sample.process_seconds for sample in self._samples)
        latencies = sorted(sample.latency_ms for sample in self._samples)
        fps = sample_count / wall_total if wall_total > 0 else 0.0
        latency_mean = sum(latencies) / sample_count
        latency_index = max(0, math.ceil(0.95 * sample_count) - 1)
        latency_p95 = latencies[min(latency_index, sample_count - 1)]
        cpu_percent = (process_total / wall_total * 100.0) if wall_total > 0 else 0.0
        return {
            "fps": float(fps),
            "latency_mean_ms": float(latency_mean),
            "latency_p95_ms": float(latency_p95),
            "cpu_percent": float(cpu_percent),
        }


def build_metrics_payload(metrics_snapshot):
    if not isinstance(metrics_snapshot, dict):
        raise ValueError("metrics_snapshot must be a mapping")
    missing = [key for key in METRIC_KEYS if key not in metrics_snapshot]
    if missing:
        raise ValueError(
            "metrics_snapshot is missing fields: " + ", ".join(missing)
        )
    return {
        "metrics": {key: float(metrics_snapshot[key]) for key in METRIC_KEYS}
    }


def measure_runtime_sample(
    *,
    previous_frame_end_ns,
    current_frame_end_ns,
    process_seconds,
    latency_ms,
):
    previous_frame_end_ns = _require_integer(
        previous_frame_end_ns, "previous_frame_end_ns", minimum=0
    )
    current_frame_end_ns = _require_integer(
        current_frame_end_ns, "current_frame_end_ns", minimum=0
    )
    process_seconds = _require_finite_number(
        process_seconds, "process_seconds", minimum=0.0
    )
    latency_ms = _require_finite_number(latency_ms, "latency_ms", minimum=0.0)
    if current_frame_end_ns <= previous_frame_end_ns:
        raise ValueError(
            "current_frame_end_ns must be greater than previous_frame_end_ns"
        )
    wall_seconds = (current_frame_end_ns - previous_frame_end_ns) / 1_000_000_000.0
    return {
        "wall_seconds": wall_seconds,
        "process_seconds": process_seconds,
        "latency_ms": latency_ms,
    }


def debug_bundle_paths(root, timestamp_prefix):
    if not isinstance(timestamp_prefix, str) or not timestamp_prefix:
        raise ValueError("timestamp_prefix must be a non-empty string")
    root = Path(root)
    return {
        "raw": root / f"{timestamp_prefix}_raw.jpg",
        "annotated": root / f"{timestamp_prefix}_annotated.jpg",
        "red_mask": root / f"{timestamp_prefix}_red_mask.png",
        "green_mask": root / f"{timestamp_prefix}_green_mask.png",
        "combined_mask": root / f"{timestamp_prefix}_combined_mask.png",
        "payload_json": root / f"{timestamp_prefix}.json",
        "metrics_json": root / f"{timestamp_prefix}_metrics.json",
    }


def write_debug_bundle(
    paths,
    *,
    frame_bgr,
    masks,
    payload,
    metrics,
    annotated_bgr=None,
    cv2_module=cv2,
):
    annotated_bgr = frame_bgr if annotated_bgr is None else annotated_bgr
    combined_mask = cv2_module.bitwise_or(masks["red"], masks["green"])
    for path in paths.values():
        _ensure_parent(path)
    artifacts = {
        "raw": frame_bgr,
        "annotated": annotated_bgr,
        "red_mask": masks["red"],
        "green_mask": masks["green"],
        "combined_mask": combined_mask,
    }
    created_paths = []
    try:
        for key, image in artifacts.items():
            path = paths[key]
            created_paths.append(path)
            if not cv2_module.imwrite(str(path), image):
                raise RuntimeError(f"failed to write {paths[key]}")
        path = paths["payload_json"]
        created_paths.append(path)
        paths["payload_json"].write_text(
            _json_compact(payload), encoding="utf-8"
        )
        path = paths["metrics_json"]
        created_paths.append(path)
        paths["metrics_json"].write_text(
            _json_compact(build_metrics_payload(metrics)), encoding="utf-8"
        )
    except Exception:
        for path in created_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def open_camera(cv2_module, camera):
    capture = None
    try:
        capture = cv2_module.VideoCapture(camera["device"], cv2_module.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError("failed to open camera")
        fourcc = camera.get("fourcc", "MJPG")
        if not isinstance(fourcc, str) or len(fourcc) != 4:
            raise ValueError("camera.fourcc must be a four-character string")
        capture.set(
            cv2_module.CAP_PROP_FOURCC,
            cv2_module.VideoWriter_fourcc(*fourcc),
        )
        capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, camera["width"])
        capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, camera["height"])
        capture.set(cv2_module.CAP_PROP_FPS, camera["fps"])
        capture.set(cv2_module.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("failed to read a verification frame from camera")
        _verify_camera_mode(capture, frame, camera, cv2_module)
        return capture, frame
    except RuntimeError:
        if capture is not None:
            capture.release()
        raise
    except Exception as exc:
        if capture is not None:
            capture.release()
        raise RuntimeError("failed to initialize camera") from exc


def _decode_fourcc(value):
    if not _is_number(value) or not math.isfinite(float(value)) or float(value) <= 0:
        return None
    packed = int(round(float(value))) & 0xFFFFFFFF
    bytes_le = [(packed >> (8 * index)) & 0xFF for index in range(4)]
    if any(byte < 32 or byte > 126 for byte in bytes_le):
        return None
    return "".join(chr(byte) for byte in bytes_le)


def _verify_camera_mode(capture, frame, camera, cv2_module):
    requested_width = int(camera["width"])
    requested_height = int(camera["height"])
    actual_height, actual_width = frame.shape[:2]
    if actual_width != requested_width or actual_height != requested_height:
        raise RuntimeError(
            "camera mode verification failed: "
            f"requested {requested_width}x{requested_height}, "
            f"actual {actual_width}x{actual_height}"
        )

    actual_fps = capture.get(cv2_module.CAP_PROP_FPS)
    if _is_number(actual_fps) and math.isfinite(float(actual_fps)) and float(actual_fps) > 0:
        requested_fps = float(camera["fps"])
        tolerance = max(1.0, requested_fps * 0.10)
        if abs(float(actual_fps) - requested_fps) > tolerance:
            raise RuntimeError(
                "camera mode verification failed: "
                f"requested fps {requested_fps:g}, actual {float(actual_fps):g}"
            )

    actual_fourcc = _decode_fourcc(capture.get(cv2_module.CAP_PROP_FOURCC))
    if (
        actual_fourcc is not None
        and actual_fourcc.upper() != str(camera["fourcc"]).upper()
    ):
        raise RuntimeError(
            "camera mode verification failed: "
            f"requested fourcc {camera['fourcc']}, actual {actual_fourcc}"
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Realtime strip detector runtime")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="path to the detector config file",
    )
    parser.add_argument(
        "--device",
        type=camera_device_arg,
        default=None,
        help="camera device or index",
    )
    parser.add_argument("--width", type=int, default=None, help="capture width")
    parser.add_argument("--height", type=int, default=None, help="capture height")
    parser.add_argument("--fps", type=int, default=None, help="capture fps")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="disable display windows and keyboard input",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=None,
        help="stop after processing this many frames",
    )
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION_PATH),
        help="path to camera calibration JSON",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="disable camera undistortion",
    )
    return parser


def _normalize_camera_config(config, args):
    camera = dict(config["camera"])
    if args.device is not None:
        camera["device"] = camera_device_arg(args.device)
    if args.width is not None:
        camera["width"] = _require_integer(args.width, "width", minimum=1)
    if args.height is not None:
        camera["height"] = _require_integer(args.height, "height", minimum=1)
    if args.fps is not None:
        camera["fps"] = _require_integer(args.fps, "fps", minimum=1)
    normalized_config = copy.deepcopy(config)
    normalized_config["camera"] = camera
    strip_detection.validate_config(normalized_config)
    return normalized_config["camera"]


def _select_view(mode, annotated_frame, masks):
    return select_display_frame(mode, annotated_frame, masks)


def load_optional_undistorter(path, camera_config, *, disabled=False):
    if disabled:
        return None
    calibration_path = Path(path)
    if not calibration_path.exists():
        return None
    calibration = camera_calibration.load_calibration(calibration_path)
    image_size = (int(camera_config["width"]), int(camera_config["height"]))
    if tuple(calibration["image_size"]) != image_size:
        raise ValueError(
            "camera resolution does not match calibration: "
            f"calibrated {calibration['image_size'][0]}x"
            f"{calibration['image_size'][1]}, "
            f"current {image_size[0]}x{image_size[1]}"
        )
    return camera_calibration.FrameUndistorter(calibration, alpha=0.0)


def _maybe_print_json(payload, metrics_snapshot):
    print(_json_compact(payload))
    print(_json_compact(build_metrics_payload(metrics_snapshot)))


def _build_annotation_frame(frame_bgr, detections, metrics_snapshot):
    return _annotate_frame(frame_bgr, detections, metrics_snapshot)


def _save_current_bundle(debug_root, frame_bgr, annotated_frame, masks, payload, metrics_snapshot):
    prefix = _timestamp_prefix()
    paths = debug_bundle_paths(debug_root, prefix)
    write_debug_bundle(
        paths,
        frame_bgr=frame_bgr,
        annotated_bgr=annotated_frame,
        masks=masks,
        payload=payload,
        metrics=metrics_snapshot,
    )


def handle_display_key(key, display_mode_index, on_save):
    if key in (ord("q"), 27):
        return display_mode_index, True
    if key in (ord("m"), ord("M")):
        return (display_mode_index + 1) % len(DISPLAY_MODES), False
    if key in (ord("s"), ord("S")):
        on_save()
    return display_mode_index, False


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = strip_detection.load_config(Path(args.config))
    camera_config = _normalize_camera_config(config, args)
    cv2.setNumThreads(int(camera_config["opencv_threads"]))
    undistorter = load_optional_undistorter(
        args.calibration,
        camera_config,
        disabled=args.no_undistort,
    )
    tracker = strip_detection.StripTracker(config)
    runtime_metrics = RuntimeMetrics(window_seconds=10)
    capture, first_frame = open_camera(cv2, camera_config)

    frame_seq = 0
    display_mode_index = 0
    next_payload_print = 0.0
    next_metrics_print = 0.0
    previous_frame_end_ns = None
    pending_first_frame = True

    try:
        debug_root = MODULE_DIR / config["output"]["debug_directory"]
        debug_root.mkdir(parents=True, exist_ok=True)
        while True:
            if pending_first_frame:
                frame_bgr = first_frame
                pending_first_frame = False
            else:
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError("failed to read frame from camera")

            process_start_ns = time.perf_counter_ns()
            process_time_start = time.process_time()
            if undistorter is not None:
                frame_bgr = undistorter.apply(frame_bgr)
            candidates, masks = strip_detection.detect_candidates(frame_bgr, config)
            detections = tracker.update(candidates)
            process_end_ns = time.perf_counter_ns()
            process_seconds = time.process_time() - process_time_start
            latency_ms = (process_end_ns - process_start_ns) / 1_000_000.0
            wall_start_ns = (
                process_start_ns if previous_frame_end_ns is None else previous_frame_end_ns
            )
            runtime_sample = measure_runtime_sample(
                previous_frame_end_ns=wall_start_ns,
                current_frame_end_ns=process_end_ns,
                process_seconds=process_seconds,
                latency_ms=latency_ms,
            )
            runtime_metrics.add_sample(
                runtime_sample["wall_seconds"],
                runtime_sample["process_seconds"],
                runtime_sample["latency_ms"],
            )
            metrics_snapshot = runtime_metrics.snapshot()
            payload = strip_detection.build_frame_payload(
                timestamp_ns=time.time_ns(),
                frame_id=config["output"]["frame_id"],
                frame_seq=frame_seq,
                image_size=(int(frame_bgr.shape[1]), int(frame_bgr.shape[0])),
                detections=detections,
            )

            if time.monotonic() >= next_payload_print:
                print(_json_compact(payload))
                next_payload_print = (
                    time.monotonic() + config["output"]["print_interval_seconds"]
                )
            if time.monotonic() >= next_metrics_print:
                print(_json_compact(build_metrics_payload(metrics_snapshot)))
                next_metrics_print = (
                    time.monotonic() + config["output"]["metrics_interval_seconds"]
                )

            annotated_frame = None
            save_requested = False
            if not args.headless:
                annotated_frame = _build_annotation_frame(
                    frame_bgr, detections, metrics_snapshot
                )
                view_name = DISPLAY_MODES[display_mode_index]
                display_frame = _select_view(view_name, annotated_frame, masks)
                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF

                def _save():
                    nonlocal save_requested
                    save_requested = True

                display_mode_index, should_exit = handle_display_key(
                    key, display_mode_index, _save
                )
                if should_exit:
                    break

            if save_requested:
                _save_current_bundle(
                    debug_root,
                    frame_bgr,
                    annotated_frame if annotated_frame is not None else frame_bgr,
                    masks,
                    payload,
                    metrics_snapshot,
                )

            previous_frame_end_ns = process_end_ns
            frame_seq += 1
            if args.max_frames is not None and frame_seq >= args.max_frames:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
