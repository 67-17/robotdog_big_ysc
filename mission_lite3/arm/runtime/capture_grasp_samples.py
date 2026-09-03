from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "strip_detector_config.json"
DEFAULT_CALIBRATION_PATH = MODULE_DIR / "ost.yaml"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "grasp_samples"
WINDOW_NAME = "grasp sample capture"


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_integer(value, name, *, minimum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _validate_fourcc(value, name="fourcc"):
    if not isinstance(value, str) or len(value) != 4:
        raise ValueError(f"{name} must be a four-character string")
    return value


def positive_int(value):
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if isinstance(value, bool) or numeric <= 0 or str(numeric) != str(value):
        raise argparse.ArgumentTypeError("must be a positive integer")
    return numeric


def unit_interval_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1")
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


def validate_camera_config(camera):
    if not isinstance(camera, dict):
        raise ValueError("camera must be an object")
    required = {"device", "width", "height", "fps", "fourcc", "opencv_threads"}
    missing = required - set(camera)
    if missing:
        raise ValueError("camera is missing fields: " + ", ".join(sorted(missing)))

    device = camera["device"]
    valid_device = (
        (isinstance(device, str) and bool(device))
        or (isinstance(device, int) and not isinstance(device, bool) and device >= 0)
    )
    if not valid_device:
        raise ValueError("camera.device must be a non-empty string or non-negative integer")

    for field in ("width", "height", "fps", "opencv_threads"):
        _require_integer(camera[field], f"camera.{field}", minimum=1)
    _validate_fourcc(camera["fourcc"], "camera.fourcc")
    return copy.deepcopy(camera)


def load_camera_config(path):
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict) or "camera" not in config:
        raise ValueError("config must contain a camera section")
    return validate_camera_config(config["camera"])


def normalize_camera_config(camera, args):
    normalized = copy.deepcopy(camera)
    overrides = (
        ("device", getattr(args, "device", None)),
        ("width", getattr(args, "width", None)),
        ("height", getattr(args, "height", None)),
        ("fps", getattr(args, "fps", None)),
        ("fourcc", getattr(args, "fourcc", None)),
        ("opencv_threads", getattr(args, "opencv_threads", None)),
    )
    for key, value in overrides:
        if value is not None:
            normalized[key] = value
    return validate_camera_config(normalized)


def timestamp_prefix(now=None):
    now = datetime.now() if now is None else now
    return now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"


def sample_paths(output_dir, timestamp, frame_seq):
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError("timestamp must be a non-empty string")
    frame_seq = _require_integer(frame_seq, "frame_seq", minimum=0)
    base = Path(output_dir) / f"{timestamp}_{frame_seq:06d}"
    return {
        "image": base.with_suffix(".jpg"),
        "metadata": base.with_suffix(".json"),
    }


def _json_compact(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _load_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for camera capture") from exc
    return cv2


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


def _validate_frame(frame_bgr):
    if (
        frame_bgr is None
        or not hasattr(frame_bgr, "shape")
        or len(frame_bgr.shape) != 3
        or frame_bgr.shape[2] != 3
    ):
        raise ValueError("frame_bgr must have shape (H, W, 3)")


def load_optional_undistorter(
    path,
    camera_config,
    *,
    disabled=False,
    alpha=0.0,
    use_projection_matrix=False,
    use_optimal_matrix=False,
    calibration_module=None,
):
    if disabled or path is None:
        return None
    calibration_path = Path(path)
    if not calibration_path.exists():
        return None
    module = (
        _load_camera_calibration_module()
        if calibration_module is None
        else calibration_module
    )
    calibration = module.load_calibration(calibration_path)
    image_size = (int(camera_config["width"]), int(camera_config["height"]))
    if tuple(calibration["image_size"]) != image_size:
        raise ValueError(
            "camera resolution does not match calibration: "
            f"calibrated {calibration['image_size'][0]}x"
            f"{calibration['image_size'][1]}, "
            f"current {image_size[0]}x{image_size[1]}"
        )
    return module.FrameUndistorter(
        calibration,
        alpha=alpha,
        use_projection_matrix=use_projection_matrix,
        use_optimal_matrix=use_optimal_matrix,
    )


def prepare_sample_frame(frame_bgr, undistorter):
    _validate_frame(frame_bgr)
    if undistorter is None:
        return frame_bgr, False
    corrected = undistorter.apply(frame_bgr)
    _validate_frame(corrected)
    return corrected, True


def write_sample(paths, frame_bgr, metadata, *, cv2_module=None):
    cv2_module = _load_cv2() if cv2_module is None else cv2_module
    _validate_frame(frame_bgr)
    image_path = Path(paths["image"])
    metadata_path = Path(paths["metadata"])
    created_paths = []
    try:
        for path in (image_path, metadata_path):
            _ensure_parent(path)

        created_paths.append(image_path)
        if not cv2_module.imwrite(str(image_path), frame_bgr):
            raise RuntimeError(f"failed to write {image_path}")

        created_paths.append(metadata_path)
        metadata_path.write_text(_json_compact(metadata), encoding="utf-8")
    except Exception:
        for path in created_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def build_sample_metadata(
    *,
    timestamp_ns,
    frame_seq,
    frame_bgr,
    camera,
    undistorted=False,
    calibration_path=None,
):
    _validate_frame(frame_bgr)
    if not _is_number(timestamp_ns) or not math.isfinite(float(timestamp_ns)):
        raise ValueError("timestamp_ns must be a finite number")
    frame_seq = _require_integer(frame_seq, "frame_seq", minimum=0)
    height, width = frame_bgr.shape[:2]
    return {
        "schema_version": 1,
        "timestamp_ns": int(timestamp_ns),
        "frame_seq": frame_seq,
        "image_size": [int(width), int(height)],
        "camera": validate_camera_config(camera),
        "image_processing": {
            "undistorted": bool(undistorted),
            "calibration_path": None if calibration_path is None else str(calibration_path),
        },
    }


def save_frame(
    output_dir,
    frame_seq,
    frame_bgr,
    camera,
    *,
    undistorted=False,
    calibration_path=None,
    cv2_module=None,
):
    paths = sample_paths(output_dir, timestamp_prefix(), frame_seq)
    metadata = build_sample_metadata(
        timestamp_ns=time.time_ns(),
        frame_seq=frame_seq,
        frame_bgr=frame_bgr,
        camera=camera,
        undistorted=undistorted,
        calibration_path=calibration_path,
    )
    write_sample(paths, frame_bgr, metadata, cv2_module=cv2_module)
    return paths


def open_camera(cv2_module, camera):
    capture = cv2_module.VideoCapture(camera["device"])
    try:
        if not capture.isOpened():
            raise RuntimeError("failed to open camera")
        capture.set(
            cv2_module.CAP_PROP_FOURCC,
            cv2_module.VideoWriter_fourcc(*camera["fourcc"]),
        )
        capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, camera["width"])
        capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, camera["height"])
        capture.set(cv2_module.CAP_PROP_FPS, camera["fps"])
        if hasattr(cv2_module, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2_module.CAP_PROP_BUFFERSIZE, 1)
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise RuntimeError("failed to read a verification frame from camera")
        return capture, first_frame
    except Exception:
        capture.release()
        raise


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Save camera images for grasp sample collection"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="path to config JSON; only the camera section is used",
    )
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION_PATH),
        help="path to camera calibration YAML/JSON for undistortion",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="save raw camera frames without undistortion",
    )
    parser.add_argument(
        "--undistort-optimal",
        action="store_true",
        help="use OpenCV optimal new camera matrix for cropped undistortion",
    )
    parser.add_argument(
        "--undistort-alpha",
        type=unit_interval_float,
        default=0.0,
        help="OpenCV undistortion alpha; 0 crops edges most, 1 keeps more field",
    )
    parser.add_argument(
        "--undistort-use-projection",
        action="store_true",
        help="use projection_matrix from calibration as the undistorted output matrix",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory for saved sample images and metadata",
    )
    parser.add_argument("--device", type=camera_device_arg, default=None)
    parser.add_argument("--width", type=positive_int, default=None)
    parser.add_argument("--height", type=positive_int, default=None)
    parser.add_argument("--fps", type=positive_int, default=None)
    parser.add_argument("--fourcc", default=None)
    parser.add_argument("--opencv-threads", type=positive_int, default=None)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="disable display and save frames automatically",
    )
    parser.add_argument(
        "--save-every",
        type=positive_int,
        default=1,
        help="headless save interval in frames",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=None,
        help="stop after reading this many frames",
    )
    return parser


def _handle_display_frame(cv2_module, frame_bgr):
    cv2_module.imshow(WINDOW_NAME, frame_bgr)
    return cv2_module.waitKey(1) & 0xFF


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.undistort_optimal and args.undistort_use_projection:
        parser.error(
            "--undistort-optimal and --undistort-use-projection cannot be used together"
        )
    camera = normalize_camera_config(load_camera_config(args.config), args)
    cv2_module = _load_cv2()
    cv2_module.setNumThreads(int(camera["opencv_threads"]))
    undistorter = load_optional_undistorter(
        args.calibration,
        camera,
        disabled=args.no_undistort,
        alpha=args.undistort_alpha,
        use_projection_matrix=args.undistort_use_projection,
        use_optimal_matrix=args.undistort_optimal,
    )
    capture, first_frame = open_camera(cv2_module, camera)

    frame_seq = 0
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
            sample_frame, was_undistorted = prepare_sample_frame(
                frame_bgr,
                undistorter,
            )

            if args.headless:
                if frame_seq % int(args.save_every) == 0:
                    paths = save_frame(
                        args.output_dir,
                        frame_seq,
                        sample_frame,
                        camera,
                        undistorted=was_undistorted,
                        calibration_path=(
                            args.calibration if was_undistorted else None
                        ),
                        cv2_module=cv2_module,
                    )
                    print(f"saved {paths['image']}")
            else:
                key = _handle_display_frame(cv2_module, sample_frame)
                if key in (ord("s"), ord("S")):
                    paths = save_frame(
                        args.output_dir,
                        frame_seq,
                        sample_frame,
                        camera,
                        undistorted=was_undistorted,
                        calibration_path=(
                            args.calibration if was_undistorted else None
                        ),
                        cv2_module=cv2_module,
                    )
                    print(f"saved {paths['image']}")
                if key in (ord("q"), 27):
                    break

            frame_seq += 1
            if args.max_frames is not None and frame_seq >= args.max_frames:
                break
    finally:
        capture.release()
        cv2_module.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
