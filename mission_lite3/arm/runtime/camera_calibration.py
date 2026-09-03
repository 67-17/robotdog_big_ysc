import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_MODEL = "opencv_rational"
SUPPORTED_MODELS = {CALIBRATION_MODEL, "plumb_bob"}


def _finite_matrix(value, name, shape=None):
    matrix = np.asarray(value, dtype=np.float64)
    if shape is not None and matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain finite numbers")
    return matrix


def validate_calibration(data):
    if not isinstance(data, dict):
        raise ValueError("calibration must be an object")
    required = {
        "schema_version",
        "model",
        "image_size",
        "camera_matrix",
        "distortion_coefficients",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(
            "calibration is missing fields: " + ", ".join(sorted(missing))
        )
    if data["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported calibration schema version")
    if data["model"] not in SUPPORTED_MODELS:
        raise ValueError(
            "calibration.model must be one of: "
            + ", ".join(sorted(SUPPORTED_MODELS))
        )

    image_size = data["image_size"]
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in image_size
        )
    ):
        raise ValueError("calibration.image_size must contain two positive integers")

    camera_matrix = _finite_matrix(
        data["camera_matrix"], "calibration.camera_matrix", (3, 3)
    )
    distortion = _finite_matrix(
        data["distortion_coefficients"],
        "calibration.distortion_coefficients",
    ).reshape(-1)
    if distortion.size not in {4, 5, 8, 12, 14}:
        raise ValueError(
            "calibration.distortion_coefficients has an unsupported length"
        )
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise ValueError("calibration focal lengths must be positive")

    if "rms_reprojection_error" in data:
        rms = data["rms_reprojection_error"]
        if (
            not isinstance(rms, (int, float))
            or isinstance(rms, bool)
            or not math.isfinite(float(rms))
            or float(rms) < 0
        ):
            raise ValueError(
                "calibration.rms_reprojection_error must be a non-negative number"
            )

    if "board" in data:
        board = data["board"]
        if not isinstance(board, dict):
            raise ValueError("calibration.board must be an object")
        inner_corners = board.get("inner_corners")
        if (
            not isinstance(inner_corners, list)
            or len(inner_corners) != 2
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in inner_corners
            )
        ):
            raise ValueError(
                "calibration.board.inner_corners must contain two positive integers"
            )
        square_size = board.get("square_size_mm")
        if (
            not isinstance(square_size, (int, float))
            or isinstance(square_size, bool)
            or not math.isfinite(float(square_size))
            or float(square_size) <= 0
        ):
            raise ValueError(
                "calibration.board.square_size_mm must be a positive number"
            )

    if "rectification_matrix" in data:
        _finite_matrix(
            data["rectification_matrix"],
            "calibration.rectification_matrix",
            (3, 3),
        )
    if "projection_matrix" in data:
        _finite_matrix(
            data["projection_matrix"],
            "calibration.projection_matrix",
            (3, 4),
        )


def _yaml_scalar(text, key):
    match = re.search(
        rf"(?m)^\s*{re.escape(key)}\s*:\s*([^\s#]+)\s*$",
        text,
    )
    if match is None:
        raise ValueError(f"ROS calibration is missing {key}")
    return match.group(1)


def _yaml_matrix(text, key, rows, cols):
    match = re.search(
        rf"(?ms)^\s*{re.escape(key)}\s*:\s*.*?"
        rf"^\s*data\s*:\s*\[(.*?)\]\s*(?=^\S|\Z)",
        text,
    )
    if match is None:
        raise ValueError(f"ROS calibration is missing {key}.data")
    values = [
        float(value)
        for value in re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            match.group(1),
        )
    ]
    if len(values) != rows * cols:
        raise ValueError(
            f"ROS calibration {key}.data must contain {rows * cols} numbers"
        )
    return np.asarray(values, dtype=np.float64).reshape(rows, cols).tolist()


def load_ros_calibration(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    width = int(_yaml_scalar(text, "image_width"))
    height = int(_yaml_scalar(text, "image_height"))
    model = _yaml_scalar(text, "distortion_model")
    distortion_match = re.search(
        r"(?ms)^\s*distortion_coefficients\s*:\s*.*?"
        r"^\s*data\s*:\s*\[(.*?)\]\s*(?=^\S|\Z)",
        text,
    )
    if distortion_match is None:
        raise ValueError(
            "ROS calibration is missing distortion_coefficients.data"
        )
    distortion = [
        float(value)
        for value in re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            distortion_match.group(1),
        )
    ]
    data = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "model": model,
        "image_size": [width, height],
        "camera_matrix": _yaml_matrix(text, "camera_matrix", 3, 3),
        "distortion_coefficients": distortion,
        "rectification_matrix": _yaml_matrix(
            text,
            "rectification_matrix",
            3,
            3,
        ),
        "projection_matrix": _yaml_matrix(text, "projection_matrix", 3, 4),
        "source": "ros_camera_calibration",
    }
    validate_calibration(data)
    return data


def load_calibration(path):
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_ros_calibration(path)
    with path.open("r", encoding="utf-8") as calibration_file:
        data = json.load(calibration_file)
    validate_calibration(data)
    return data


def save_calibration(path, data):
    validate_calibration(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_undistort_maps(
    data,
    image_size=None,
    alpha=0.0,
    use_projection_matrix=False,
    use_optimal_matrix=False,
):
    validate_calibration(data)
    calibrated_size = tuple(data["image_size"])
    requested_size = calibrated_size if image_size is None else tuple(image_size)
    if requested_size != calibrated_size:
        raise ValueError(
            "camera resolution does not match calibration: "
            f"calibrated {calibrated_size[0]}x{calibrated_size[1]}, "
            f"current {requested_size[0]}x{requested_size[1]}"
        )
    if not isinstance(alpha, (int, float)) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(
        data["distortion_coefficients"], dtype=np.float64
    ).reshape(-1, 1)
    if use_projection_matrix and "projection_matrix" in data:
        new_camera_matrix = np.asarray(
            data["projection_matrix"],
            dtype=np.float64,
        )[:, :3]
        rectification = np.asarray(
            data.get("rectification_matrix", np.eye(3)),
            dtype=np.float64,
        )
        roi = (0, 0, calibrated_size[0], calibrated_size[1])
    elif use_optimal_matrix:
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion,
            calibrated_size,
            float(alpha),
            calibrated_size,
        )
        rectification = None
    else:
        new_camera_matrix = camera_matrix.copy()
        rectification = np.asarray(
            data.get("rectification_matrix", np.eye(3)),
            dtype=np.float64,
        )
        roi = (0, 0, calibrated_size[0], calibrated_size[1])
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        rectification,
        new_camera_matrix,
        calibrated_size,
        cv2.CV_32FC1,
    )
    return map_x, map_y, new_camera_matrix, tuple(int(value) for value in roi)


class FrameUndistorter:
    def __init__(
        self,
        calibration,
        *,
        alpha=0.0,
        use_projection_matrix=False,
        use_optimal_matrix=False,
    ):
        self.image_size = tuple(calibration["image_size"])
        (
            self.map_x,
            self.map_y,
            self.new_camera_matrix,
            self.valid_roi,
        ) = build_undistort_maps(
            calibration,
            alpha=alpha,
            use_projection_matrix=use_projection_matrix,
            use_optimal_matrix=use_optimal_matrix,
        )

    def apply(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        if (width, height) != self.image_size:
            raise ValueError(
                "frame resolution does not match calibration: "
                f"expected {self.image_size[0]}x{self.image_size[1]}, "
                f"received {width}x{height}"
            )
        return cv2.remap(
            frame_bgr,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
