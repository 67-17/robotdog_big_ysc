from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..camera import CameraSource
from ..config_loader import load_config
from ..wide_camera import (
    WideCameraUndistorter,
    annotate_box_parallel,
    detect_box_parallel,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Lite3 wide-camera box parallelism measurement"
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--tolerance-deg", type=float, default=1.5)
    parser.add_argument("--max-range-deg", type=float, default=0.8)
    parser.add_argument("--output-dir", default="wide_box_parallel_runs")
    args = parser.parse_args()
    config = load_config(args.config_dir)
    project_root = Path(__file__).resolve().parents[2]
    calibration_path = Path(config["camera"]["wide_calibration"])
    if not calibration_path.is_absolute():
        calibration_path = project_root / calibration_path
    undistorter = WideCameraUndistorter.from_file(calibration_path)
    camera = CameraSource(
        config["camera"]["front"],
        dry_run=False,
        flush_grab_frames=int(config["camera"].get("flush_grab_frames", 4)),
        stale_frame_reconnect_count=int(
            config["camera"].get("stale_frame_reconnect_count", 15)
        ),
        digital_zoom=1.0,
    )
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    try:
        for index in range(1, max(1, args.frames) + 1):
            frame = camera.read()
            if frame is None:
                samples.append({"frame": index, "ok": False, "reason": "camera_read_failed"})
                continue
            undistorted = undistorter.apply(frame)
            result = detect_box_parallel(undistorted)
            sample = {"frame": index, **asdict(result)}
            samples.append(sample)
            annotated = annotate_box_parallel(undistorted, result)
            cv2.imwrite(str(run_dir / f"annotated_{index:03d}.jpg"), annotated)
    finally:
        camera.release()

    errors = [
        float(sample["parallel_error_deg"])
        for sample in samples
        if sample.get("ok") and sample.get("parallel_error_deg") is not None
    ]
    median_error = float(np.median(errors)) if errors else None
    full_error_range = float(max(errors) - min(errors)) if errors else None
    error_range = (
        float(np.percentile(errors, 90) - np.percentile(errors, 10))
        if len(errors) >= 8
        else full_error_range
    )
    stable = (
        len(errors) >= max(5, args.frames // 2)
        and error_range is not None
        and error_range <= args.max_range_deg
    )
    aligned = (
        stable
        and median_error is not None
        and abs(median_error) <= args.tolerance_deg
    )
    result = {
        "ok": stable,
        "aligned": aligned,
        "median_parallel_error_deg": median_error,
        "error_range_deg": error_range,
        "full_error_range_deg": full_error_range,
        "successful_frames": len(errors),
        "requested_frames": args.frames,
        "run_dir": str(run_dir),
        "motion_command_count": 0,
        "samples": samples,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
