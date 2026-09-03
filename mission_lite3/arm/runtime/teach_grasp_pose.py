from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "strip_detector_config.json"
DEFAULT_REFERENCE_PATH = MODULE_DIR / "grasp_reference_square_face.json"
DEFAULT_CALIBRATION_PATH = MODULE_DIR / "ost.yaml"
WINDOW_NAME = "teach grasp pose"


def unit_interval_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1")
    return numeric


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


def _load_strip_detector():
    try:
        import strip_detector as module
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location(
            "strip_detector", MODULE_DIR / "strip_detector.py"
        )
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules["strip_detector"] = module
        spec.loader.exec_module(module)
    return module


def _json_compact(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _now_iso8601():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _require_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _number(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _number_pair(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two numbers")
    return [_number(value[0], f"{name}[0]"), _number(value[1], f"{name}[1]")]


def _normalize_camera(camera):
    camera = dict(_require_mapping(camera, "camera"))
    required = ("device", "width", "height", "fps", "calibration")
    missing = [field for field in required if field not in camera]
    if missing:
        raise ValueError("camera is missing fields: " + ", ".join(missing))
    if not isinstance(camera["device"], (str, int)) or isinstance(camera["device"], bool):
        raise ValueError("camera.device must be a path or non-negative integer")
    if isinstance(camera["device"], str) and not camera["device"]:
        raise ValueError("camera.device must not be empty")
    normalized = {
        "device": camera["device"],
        "width": int(_number(camera["width"], "camera.width")),
        "height": int(_number(camera["height"], "camera.height")),
        "fps": int(_number(camera["fps"], "camera.fps")),
        "calibration": str(camera["calibration"]),
    }
    if normalized["width"] <= 0 or normalized["height"] <= 0 or normalized["fps"] <= 0:
        raise ValueError("camera width, height and fps must be positive")
    return normalized


def _normalize_target(target):
    target = dict(_require_mapping(target, "target"))
    required = (
        "track_id",
        "color",
        "center_px",
        "angle_deg",
        "angle_reliable",
        "size_px",
        "area_px",
        "confidence",
        "stable_frames",
        "stable",
        "grasp_candidate",
    )
    missing = [field for field in required if field not in target]
    if missing:
        raise ValueError("target is missing fields: " + ", ".join(missing))
    if target["color"] != "red":
        raise ValueError("target.color must be red")
    if not target["stable"]:
        raise ValueError("target.stable must be true")
    if not target["grasp_candidate"]:
        raise ValueError("target.grasp_candidate must be true")
    return {
        "track_id": int(target["track_id"]),
        "color": target["color"],
        "center_px": _number_pair(target["center_px"], "target.center_px"),
        "angle_deg": _number(target["angle_deg"], "target.angle_deg"),
        "angle_reliable": bool(target["angle_reliable"]),
        "size_px": _number_pair(target["size_px"], "target.size_px"),
        "area_px": _number(target["area_px"], "target.area_px"),
        "confidence": _number(target["confidence"], "target.confidence"),
        "stable_frames": int(target["stable_frames"]),
        "stable": bool(target["stable"]),
        "grasp_candidate": bool(target["grasp_candidate"]),
    }


def _normalize_arm_status(arm_status):
    if arm_status is None:
        return {}
    status = _require_mapping(arm_status, "arm_status")
    return {str(key): _number(value, f"arm_status.{key}") for key, value in status.items()}


def default_terminal_sequence():
    return {
        "open_gripper_h": -45,
        "close_gripper_h": 45,
        "lift_joints_deg": {"e": -5.0},
    }


def build_grasp_reference(
    *,
    camera,
    target,
    arm_status=None,
    terminal_sequence=None,
    created_at=None,
):
    return {
        "schema_version": 1,
        "created_at": created_at or _now_iso8601(),
        "camera": _normalize_camera(camera),
        "target": _normalize_target(target),
        "arm_status": _normalize_arm_status(arm_status),
        "terminal_sequence": terminal_sequence or default_terminal_sequence(),
    }


def annotated_image_path(reference_path):
    reference_path = Path(reference_path)
    return reference_path.with_name(reference_path.stem + "_annotated.jpg")


def _write_image(path, image, cv2_module=None):
    cv2_module = cv2 if cv2_module is None else cv2_module
    if cv2_module is not None:
        return cv2_module.imwrite(str(path), image)
    Path(path).write_bytes(b"opencv-not-available\n")
    return True


def save_reference_bundle(reference_path, reference, *, annotated_bgr=None, cv2_module=None):
    reference_path = Path(reference_path)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(_json_compact(reference), encoding="utf-8")
    result = {"json": reference_path}
    if annotated_bgr is not None:
        image_path = annotated_image_path(reference_path)
        if not _write_image(image_path, annotated_bgr, cv2_module=cv2_module):
            raise RuntimeError(f"failed to write {image_path}")
        result["annotated"] = image_path
    return result


def target_dict_from_detection(detection):
    strip_detection = _load_strip_detection()
    return strip_detection.tracked_strip_to_dict(detection)


def camera_reference_from_config(camera_config, calibration_path):
    return {
        "device": camera_config["device"],
        "width": int(camera_config["width"]),
        "height": int(camera_config["height"]),
        "fps": int(camera_config["fps"]),
        "calibration": str(calibration_path),
    }


def load_arm_status(path):
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as status_file:
        return json.load(status_file)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Save a grasp teaching reference")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--output", default=str(DEFAULT_REFERENCE_PATH))
    parser.add_argument("--arm-status-json", default=None)
    parser.add_argument("--no-undistort", action="store_true")
    parser.add_argument("--undistort-optimal", action="store_true")
    parser.add_argument(
        "--undistort-alpha",
        type=unit_interval_float,
        default=0.0,
    )
    parser.add_argument("--undistort-use-projection", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if cv2 is None:
        raise RuntimeError("OpenCV is required to run the camera teaching tool")
    runtime = _load_strip_detector()
    if args.undistort_optimal and args.undistort_use_projection:
        raise ValueError(
            "--undistort-optimal and --undistort-use-projection cannot be used together"
        )
    strip_detection = _load_strip_detection()
    config = strip_detection.load_config(args.config)
    camera_args = argparse.Namespace(
        device=runtime.camera_device_arg(args.device) if args.device is not None else None,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    camera_config = runtime._normalize_camera_config(config, camera_args)
    cv2.setNumThreads(int(camera_config["opencv_threads"]))
    undistorter = runtime.load_optional_undistorter(
        args.calibration,
        camera_config,
        disabled=args.no_undistort,
        alpha=args.undistort_alpha,
        use_projection_matrix=args.undistort_use_projection,
        use_optimal_matrix=args.undistort_optimal,
    )
    tracker = strip_detection.StripTracker(config)
    capture, first_frame = runtime.open_camera(cv2, camera_config)
    pending_first_frame = True
    arm_status = load_arm_status(args.arm_status_json)

    try:
        while True:
            if pending_first_frame:
                frame_bgr = first_frame
                pending_first_frame = False
            else:
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    raise RuntimeError("failed to read frame from camera")
            if undistorter is not None:
                frame_bgr = undistorter.apply(frame_bgr)
            candidates, _masks = strip_detection.detect_candidates(frame_bgr, config)
            detections = tracker.update(candidates)
            target = strip_detection.select_grasp_target(
                detections,
                image_size=(frame_bgr.shape[1], frame_bgr.shape[0]),
            )
            annotated = runtime._build_annotation_frame(
                frame_bgr,
                detections,
                {
                    "fps": 0.0,
                    "latency_mean_ms": 0.0,
                    "latency_p95_ms": 0.0,
                    "cpu_percent": 0.0,
                },
            )
            cv2.imshow(WINDOW_NAME, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 1
            if key in (ord("s"), ord("S")):
                if target is None or not target.grasp_candidate:
                    print("当前没有 stable=1、grasp=1 的红色目标，未保存")
                    continue
                reference = build_grasp_reference(
                    camera=camera_reference_from_config(camera_config, args.calibration),
                    target=target_dict_from_detection(target),
                    arm_status=arm_status,
                )
                paths = save_reference_bundle(
                    args.output,
                    reference,
                    annotated_bgr=annotated,
                )
                print(_json_compact({key: str(value) for key, value in paths.items()}))
                return 0
            time.sleep(0.001)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
