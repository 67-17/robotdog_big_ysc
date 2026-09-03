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
DEFAULT_CALIBRATION_PATH = MODULE_DIR / "ost.yaml"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "可抓取参考图"
WINDOW_NAME = "save grasp reference image"


def unit_interval_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1")
    return numeric


def _load_local_module(module_name):
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location(
            module_name,
            MODULE_DIR / f"{module_name}.py",
        )
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def _json_compact(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _now():
    return datetime.now(timezone.utc).astimezone()


def _isoformat(value):
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value)


def reference_paths(output_dir, created_at=None):
    created_at = _now() if created_at is None else created_at
    prefix = f"grasp_reference_{created_at.strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_dir)
    return {
        "image": output_dir / f"{prefix}.jpg",
        "json": output_dir / f"{prefix}.json",
    }


def _frame_image_size(frame_bgr):
    if not hasattr(frame_bgr, "shape") or len(frame_bgr.shape) < 2:
        raise ValueError("frame_bgr must be an image array")
    height, width = frame_bgr.shape[:2]
    return int(width), int(height)


def _camera_reference(camera, calibration_path):
    return {
        "device": camera["device"],
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "fps": int(camera["fps"]),
        "calibration": str(calibration_path),
    }


def build_reference_metadata(
    *,
    camera,
    calibration_path,
    image_path,
    image_size,
    created_at,
):
    width, height = image_size
    return {
        "schema_version": 1,
        "kind": "manual_grasp_reference_image",
        "created_at": _isoformat(created_at),
        "camera": _camera_reference(camera, calibration_path),
        "image": {
            "path": str(image_path),
            "size_px": [int(width), int(height)],
        },
        "note": "Manual image reference only; no color target detection was required.",
    }


def save_reference_image(
    output_dir,
    frame_bgr,
    *,
    camera,
    calibration_path,
    created_at=None,
    cv2_module=None,
):
    cv2_module = cv2 if cv2_module is None else cv2_module
    if cv2_module is None:
        raise RuntimeError("OpenCV is required to save the reference image")
    created_at = _now() if created_at is None else created_at
    paths = reference_paths(output_dir, created_at)
    paths["image"].parent.mkdir(parents=True, exist_ok=True)
    if not cv2_module.imwrite(str(paths["image"]), frame_bgr):
        raise RuntimeError(f"failed to write {paths['image']}")
    metadata = build_reference_metadata(
        camera=camera,
        calibration_path=calibration_path,
        image_path=paths["image"],
        image_size=_frame_image_size(frame_bgr),
        created_at=created_at,
    )
    paths["json"].write_text(_json_compact(metadata), encoding="utf-8")
    return paths


def _display_frame(frame_bgr, cv2_module):
    frame = frame_bgr.copy()
    cv2_module.putText(
        frame,
        "s: save  q/esc: quit",
        (24, 38),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2_module.LINE_AA,
    )
    cv2_module.putText(
        frame,
        "s: save  q/esc: quit",
        (24, 38),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        1,
        cv2_module.LINE_AA,
    )
    return frame


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Save a manual grasp reference image without target detection"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
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
        raise RuntimeError("OpenCV is required to run the camera reference tool")
    if args.undistort_optimal and args.undistort_use_projection:
        raise ValueError(
            "--undistort-optimal and --undistort-use-projection cannot be used together"
        )

    strip_detection = _load_local_module("strip_detection")
    runtime = _load_local_module("strip_detector")
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
    capture, first_frame = runtime.open_camera(cv2, camera_config)
    pending_first_frame = True

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
            cv2.imshow(WINDOW_NAME, _display_frame(frame_bgr, cv2))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 1
            if key in (ord("s"), ord("S")):
                paths = save_reference_image(
                    args.output_dir,
                    frame_bgr,
                    camera=camera_config,
                    calibration_path=args.calibration,
                    cv2_module=cv2,
                )
                print(_json_compact({key: str(value) for key, value in paths.items()}))
                return 0
            time.sleep(0.001)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
