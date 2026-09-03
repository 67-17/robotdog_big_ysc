from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from ..camera import CameraSource
from ..config_loader import load_config


DEFAULT_PATTERN = (8, 11)
DEFAULT_SQUARE_SIZE_M = 0.015
DETECTION_FLAGS = (
    cv2.CALIB_CB_EXHAUSTIVE
    | cv2.CALIB_CB_ACCURACY
    | cv2.CALIB_CB_NORMALIZE_IMAGE
)


@dataclass(frozen=True)
class BoardView:
    center_x: float
    center_y: float
    width_ratio: float
    height_ratio: float
    angle_deg: float
    horizontal_skew: float
    vertical_skew: float
    sharpness: float


def parse_pattern(value: str) -> tuple[int, int]:
    text = value.lower().replace("*", "x").replace(",", "x")
    parts = [item.strip() for item in text.split("x") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("pattern must look like 8x11")
    pattern = (int(parts[0]), int(parts[1]))
    if min(pattern) < 3:
        raise argparse.ArgumentTypeError("each inner-corner dimension must be >= 3")
    return pattern


def detect_board(
    frame: np.ndarray,
    pattern: tuple[int, int],
) -> tuple[bool, np.ndarray | None, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray,
        pattern,
        DETECTION_FLAGS,
    )
    return bool(found), corners, gray


def board_view(
    corners: np.ndarray,
    pattern: tuple[int, int],
    image_size: tuple[int, int],
    sharpness: float,
) -> BoardView:
    width, height = image_size
    columns, rows = pattern
    points = np.asarray(corners, dtype=np.float64).reshape(rows, columns, 2)
    top_left = points[0, 0]
    top_right = points[0, -1]
    bottom_left = points[-1, 0]
    bottom_right = points[-1, -1]
    top = float(np.linalg.norm(top_right - top_left))
    bottom = float(np.linalg.norm(bottom_right - bottom_left))
    left = float(np.linalg.norm(bottom_left - top_left))
    right = float(np.linalg.norm(bottom_right - top_right))
    center = points.mean(axis=(0, 1))
    horizontal = top_right - top_left
    angle_deg = math.degrees(math.atan2(horizontal[1], horizontal[0]))
    return BoardView(
        center_x=float(center[0] / width),
        center_y=float(center[1] / height),
        width_ratio=max(top, bottom) / width,
        height_ratio=max(left, right) / height,
        angle_deg=float(angle_deg),
        horizontal_skew=float(math.log(max(top, 1e-9) / max(bottom, 1e-9))),
        vertical_skew=float(math.log(max(left, 1e-9) / max(right, 1e-9))),
        sharpness=float(sharpness),
    )


def _angle_difference_degrees(first: float, second: float) -> float:
    return abs((first - second + 90.0) % 180.0 - 90.0)


def is_diverse_view(candidate: BoardView, accepted: Sequence[BoardView]) -> bool:
    for previous in accepted:
        center_distance = math.hypot(
            candidate.center_x - previous.center_x,
            candidate.center_y - previous.center_y,
        )
        scale_change = max(
            abs(math.log(max(candidate.width_ratio, 1e-9) / max(previous.width_ratio, 1e-9))),
            abs(math.log(max(candidate.height_ratio, 1e-9) / max(previous.height_ratio, 1e-9))),
        )
        angle_change = _angle_difference_degrees(
            candidate.angle_deg,
            previous.angle_deg,
        )
        perspective_change = max(
            abs(candidate.horizontal_skew - previous.horizontal_skew),
            abs(candidate.vertical_skew - previous.vertical_skew),
        )
        if (
            center_distance < 0.075
            and scale_change < 0.12
            and angle_change < 7.0
            and perspective_change < 0.08
        ):
            return False
    return True


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def capture(args: argparse.Namespace) -> int:
    config = load_config(args.config_dir)
    source = args.device or config["camera"]["front"]
    root = Path(args.output_dir)
    session = root / (args.session or datetime.now().strftime("%Y%m%d-%H%M%S"))
    raw_dir = session / "raw"
    annotated_dir = session / "annotated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    camera = CameraSource(
        source,
        dry_run=False,
        flush_grab_frames=2,
        stale_frame_reconnect_count=15,
        digital_zoom=1.0,
    )
    manifest_path = session / "manifest.json"
    previous_manifest: dict[str, object] = {}
    accepted: list[BoardView] = []
    records: list[dict[str, object]] = []
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_pattern = tuple(
            int(value)
            for value in previous_manifest.get("pattern_inner_corners", [])
        )
        previous_square_size = float(previous_manifest.get("square_size_m", 0.0))
        if previous_pattern != args.pattern:
            raise RuntimeError(
                f"resume pattern mismatch: {previous_pattern} != {args.pattern}"
            )
        if abs(previous_square_size - args.square_size_mm / 1000.0) > 1e-9:
            raise RuntimeError("resume square size does not match the session")
        records = list(previous_manifest.get("samples", []))
        accepted = [
            BoardView(**dict(record["view"]))
            for record in records
        ]
        print(
            f"[wide-calibration] resuming {session} with {len(accepted)} samples",
            flush=True,
        )
    rejected_blur = int(previous_manifest.get("rejected_blur", 0))
    rejected_duplicate = int(previous_manifest.get("rejected_duplicate", 0))
    detection_misses = int(previous_manifest.get("detection_misses", 0))
    started_at = time.monotonic()
    last_accept_at = -math.inf
    frame_index = 0
    try:
        while (
            len(accepted) < args.max_samples
            and time.monotonic() - started_at < args.duration
        ):
            frame = camera.read()
            if frame is None:
                detection_misses += 1
                continue
            frame_index += 1
            found, corners, gray = detect_board(frame, args.pattern)
            if not found or corners is None:
                detection_misses += 1
                continue
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            view = board_view(
                corners,
                args.pattern,
                (frame.shape[1], frame.shape[0]),
                sharpness,
            )
            if sharpness < args.min_sharpness:
                rejected_blur += 1
                continue
            if time.monotonic() - last_accept_at < args.min_interval:
                continue
            if not is_diverse_view(view, accepted):
                rejected_duplicate += 1
                continue

            sample_index = len(accepted) + 1
            filename = f"board_{sample_index:03d}.jpg"
            raw_path = raw_dir / filename
            annotated_path = annotated_dir / filename
            annotated = frame.copy()
            cv2.drawChessboardCorners(
                annotated,
                args.pattern,
                corners,
                True,
            )
            if not cv2.imwrite(str(raw_path), frame):
                raise RuntimeError(f"failed to save {raw_path}")
            if not cv2.imwrite(str(annotated_path), annotated):
                raise RuntimeError(f"failed to save {annotated_path}")
            accepted.append(view)
            last_accept_at = time.monotonic()
            record = {
                "sample": sample_index,
                "raw": str(raw_path),
                "annotated": str(annotated_path),
                "view": asdict(view),
            }
            records.append(record)
            print(
                "[wide-calibration] accepted "
                f"{sample_index}/{args.max_samples} "
                f"center=({view.center_x:.2f},{view.center_y:.2f}) "
                f"size=({view.width_ratio:.2f},{view.height_ratio:.2f}) "
                f"angle={view.angle_deg:+.1f} sharpness={view.sharpness:.0f}",
                flush=True,
            )
    finally:
        camera.release()

    previous_duration = float(previous_manifest.get("duration_seconds", 0.0))
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "image_size": [1280, 720],
        "pattern_inner_corners": list(args.pattern),
        "square_size_m": args.square_size_mm / 1000.0,
        "accepted_samples": len(accepted),
        "duration_seconds": previous_duration + time.monotonic() - started_at,
        "rejected_blur": rejected_blur,
        "rejected_duplicate": rejected_duplicate,
        "detection_misses": detection_misses,
        "samples": records,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"ok": len(accepted) >= args.min_samples, "session": str(session), **manifest}, ensure_ascii=False))
    return 0 if len(accepted) >= args.min_samples else 1


def _object_points(pattern: tuple[int, int], square_size_m: float) -> np.ndarray:
    columns, rows = pattern
    points = np.zeros((columns * rows, 3), dtype=np.float64)
    points[:, :2] = (
        np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size_m
    )
    return points


def _per_view_errors_pinhole(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
) -> list[float]:
    errors = []
    for objects, observed, rvec, tvec in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
    ):
        projected, _ = cv2.projectPoints(
            objects,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        difference = projected.reshape(-1, 2) - observed.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))))
    return errors


def _per_view_errors_fisheye(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rvecs: Sequence[np.ndarray],
    tvecs: Sequence[np.ndarray],
) -> list[float]:
    errors = []
    for objects, observed, rvec, tvec in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
    ):
        projected, _ = cv2.fisheye.projectPoints(
            objects.reshape(1, -1, 3),
            rvec,
            tvec,
            camera_matrix,
            distortion,
        )
        difference = projected.reshape(-1, 2) - observed.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))))
    return errors


def _line_residual(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    distances = centered @ normal
    return float(np.sqrt(np.mean(distances**2)))


def straightness_error(
    image_points: Sequence[np.ndarray],
    pattern: tuple[int, int],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    fisheye: bool,
) -> float:
    columns, rows = pattern
    residuals: list[float] = []
    for corners in image_points:
        if fisheye:
            undistorted = cv2.fisheye.undistortPoints(
                corners.astype(np.float64),
                camera_matrix,
                distortion,
                P=camera_matrix,
            )
        else:
            undistorted = cv2.undistortPoints(
                corners.astype(np.float64),
                camera_matrix,
                distortion,
                P=camera_matrix,
            )
        grid = undistorted.reshape(rows, columns, 2)
        residuals.extend(_line_residual(grid[row, :, :]) for row in range(rows))
        residuals.extend(_line_residual(grid[:, column, :]) for column in range(columns))
    return float(np.sqrt(np.mean(np.square(residuals))))


def _coverage(views: Sequence[BoardView]) -> dict[str, float]:
    if not views:
        return {}
    centers_x = [view.center_x for view in views]
    centers_y = [view.center_y for view in views]
    scales = [view.width_ratio * view.height_ratio for view in views]
    angles = [view.angle_deg for view in views]
    return {
        "center_x_span": max(centers_x) - min(centers_x),
        "center_y_span": max(centers_y) - min(centers_y),
        "area_scale_ratio": max(scales) / max(min(scales), 1e-9),
        "angle_span_deg": max(angles) - min(angles),
    }


def calibrate(args: argparse.Namespace) -> int:
    session = Path(args.session)
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    pattern = tuple(int(value) for value in manifest["pattern_inner_corners"])
    square_size_m = float(manifest["square_size_m"])
    raw_paths = sorted((session / "raw").glob("board_*.jpg"))
    object_template = _object_points(pattern, square_size_m)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    views: list[BoardView] = []
    image_size: tuple[int, int] | None = None
    for path in raw_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        found, corners, gray = detect_board(frame, pattern)
        if not found or corners is None:
            continue
        current_size = (frame.shape[1], frame.shape[0])
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            raise RuntimeError("calibration images have inconsistent sizes")
        object_points.append(object_template.copy())
        image_points.append(corners.astype(np.float64))
        views.append(
            board_view(
                corners,
                pattern,
                current_size,
                float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            )
        )
    if image_size is None or len(image_points) < args.min_samples:
        raise RuntimeError(
            f"need at least {args.min_samples} valid samples; got {len(image_points)}"
        )

    pinhole_rms, pinhole_k, pinhole_d, pinhole_rvecs, pinhole_tvecs = (
        cv2.calibrateCamera(
            [points.astype(np.float32) for points in object_points],
            [points.astype(np.float32) for points in image_points],
            image_size,
            None,
            None,
        )
    )
    pinhole_errors = _per_view_errors_pinhole(
        object_points,
        image_points,
        pinhole_k,
        pinhole_d,
        pinhole_rvecs,
        pinhole_tvecs,
    )
    pinhole_straightness = straightness_error(
        image_points,
        pattern,
        pinhole_k,
        pinhole_d,
        fisheye=False,
    )

    fisheye_result: dict[str, object] | None = None
    try:
        fisheye_k = np.zeros((3, 3), dtype=np.float64)
        fisheye_d = np.zeros((4, 1), dtype=np.float64)
        fisheye_rms, fisheye_k, fisheye_d, fisheye_rvecs, fisheye_tvecs = (
            cv2.fisheye.calibrate(
                [points.reshape(1, -1, 3) for points in object_points],
                image_points,
                image_size,
                fisheye_k,
                fisheye_d,
                flags=(
                    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                    | cv2.fisheye.CALIB_CHECK_COND
                    | cv2.fisheye.CALIB_FIX_SKEW
                ),
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    100,
                    1e-7,
                ),
            )
        )
        fisheye_errors = _per_view_errors_fisheye(
            object_points,
            image_points,
            fisheye_k,
            fisheye_d,
            fisheye_rvecs,
            fisheye_tvecs,
        )
        fisheye_straightness = straightness_error(
            image_points,
            pattern,
            fisheye_k,
            fisheye_d,
            fisheye=True,
        )
        fisheye_result = {
            "model": "fisheye_kb4",
            "rms_reprojection_px": float(fisheye_rms),
            "mean_view_error_px": float(np.mean(fisheye_errors)),
            "max_view_error_px": float(np.max(fisheye_errors)),
            "straightness_rms_px": fisheye_straightness,
            "camera_matrix": fisheye_k.tolist(),
            "distortion_coefficients": fisheye_d.reshape(-1).tolist(),
        }
    except cv2.error as exc:
        fisheye_result = {"model": "fisheye_kb4", "error": str(exc)}

    pinhole_result = {
        "model": "pinhole",
        "rms_reprojection_px": float(pinhole_rms),
        "mean_view_error_px": float(np.mean(pinhole_errors)),
        "max_view_error_px": float(np.max(pinhole_errors)),
        "straightness_rms_px": pinhole_straightness,
        "camera_matrix": pinhole_k.tolist(),
        "distortion_coefficients": pinhole_d.reshape(-1).tolist(),
    }
    candidates = [pinhole_result]
    if fisheye_result is not None and "error" not in fisheye_result:
        candidates.append(fisheye_result)
    selected = min(
        candidates,
        key=lambda result: float(result["straightness_rms_px"])
        + 0.25 * float(result["rms_reprojection_px"]),
    )
    coverage = _coverage(views)
    quality_pass = (
        len(image_points) >= args.min_samples
        and coverage.get("center_x_span", 0.0) >= 0.45
        and coverage.get("center_y_span", 0.0) >= 0.35
        and coverage.get("area_scale_ratio", 0.0) >= 1.8
        and coverage.get("angle_span_deg", 0.0) >= 20.0
        and float(selected["rms_reprojection_px"]) <= 1.0
        and float(selected["straightness_rms_px"]) <= 0.5
    )
    result = {
        "schema_version": 1,
        "validated_for_control": False,
        "quality_pass": quality_pass,
        "image_size": list(image_size),
        "pattern_inner_corners": list(pattern),
        "square_size_m": square_size_m,
        "sample_count": len(image_points),
        "coverage": coverage,
        "selected_model": selected["model"],
        "selected": selected,
        "candidates": candidates,
        "note": "candidate only; inspect undistorted straight lines before enabling control",
    }
    output = Path(args.output) if args.output else session / "calibration_candidate.json"
    _write_json(output, result)
    print(json.dumps({"ok": quality_pass, "output": str(output), **result}, ensure_ascii=False))
    return 0 if quality_pass else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and calibrate the Lite3 front wide-angle camera"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--config-dir", type=Path, default=None)
    capture_parser.add_argument("--device")
    capture_parser.add_argument("--pattern", type=parse_pattern, default=DEFAULT_PATTERN)
    capture_parser.add_argument("--square-size-mm", type=float, default=15.0)
    capture_parser.add_argument("--duration", type=float, default=45.0)
    capture_parser.add_argument("--max-samples", type=int, default=30)
    capture_parser.add_argument("--min-samples", type=int, default=18)
    capture_parser.add_argument("--min-interval", type=float, default=0.35)
    capture_parser.add_argument("--min-sharpness", type=float, default=60.0)
    capture_parser.add_argument(
        "--output-dir",
        default="wide_camera_calibration_runs",
    )
    capture_parser.add_argument("--session")

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("session")
    calibrate_parser.add_argument("--min-samples", type=int, default=18)
    calibrate_parser.add_argument("--output")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "capture":
        if args.square_size_mm <= 0.0:
            raise SystemExit("square size must be positive")
        if args.duration <= 0.0 or args.max_samples < 1 or args.min_samples < 1:
            raise SystemExit("capture limits must be positive")
        return capture(args)
    return calibrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
