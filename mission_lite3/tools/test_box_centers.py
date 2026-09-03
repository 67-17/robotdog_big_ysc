from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from mission_lite3.box_center_alignment import (
    BoxCenterMeasurer,
    annotate_box_centers,
)
from mission_lite3.camera import CameraSource
from mission_lite3.config_loader import PROJECT_ROOT, load_config
from mission_lite3.wide_camera import WideCameraUndistorter


def _resolve_project_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _prompt(message: str, assume_yes: bool) -> None:
    print(message)
    if not assume_yes:
        input("按 Enter 开始采集 7 帧（不会发送任何运动命令）...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只读测试放置区和抓取区箱体中心识别")
    parser.add_argument(
        "--scene",
        choices=("both", "placement", "pickup"),
        default="both",
        help="要采集的现场场景",
    )
    parser.add_argument("--target-letter", choices=tuple("ABCD"), default=None)
    parser.add_argument("--yes", action="store_true", help="跳过每个场景的 Enter 提示")
    parser.add_argument("--output-dir", default=None, help="覆盖默认 box_recognition_runs 输出根目录")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()
    center_config = dict(config["box_center_alignment"])
    center_config["frames_per_measurement"] = 7
    center_config["min_valid_frames"] = 4
    center_config["max_center_range_fraction"] = 0.03
    output_root = _resolve_project_path(
        args.output_dir or center_config["recognition_run_log_dir"]
    )
    run_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True, exist_ok=False)

    camera_config = config["camera"]
    camera = CameraSource(
        camera_config["front"],
        dry_run=False,
        flush_grab_frames=int(camera_config.get("flush_grab_frames", 2)),
        stale_frame_reconnect_count=int(camera_config.get("stale_frame_reconnect_count", 15)),
        digital_zoom=1.0,
        open_timeout_ms=int(camera_config.get("open_timeout_ms", 3000)),
        read_timeout_ms=int(camera_config.get("read_timeout_ms", 2000)),
        reconnect_backoff_s=float(camera_config.get("reconnect_backoff_s", 0.25)),
    )
    undistorter = WideCameraUndistorter.from_file(
        _resolve_project_path(camera_config["wide_calibration"])
    )
    measurer = BoxCenterMeasurer(
        camera=camera,
        undistorter=undistorter,
        config=center_config,
    )
    scenes = (
        ("placement", "请将机器狗人工移动到放置区停车位，确保前向广角画面能同时看到四个箱子。"),
        ("pickup", "请将机器狗人工移动到抓取区停车位，确保前向广角画面能看到纸箱/物块箱。"),
    )
    if args.scene != "both":
        scenes = tuple(item for item in scenes if item[0] == args.scene)
    scene_results: dict[str, object] = {}
    try:
        for mode, prompt in scenes:
            _prompt(prompt, args.yes)
            camera.release()

            def save_frame(index, raw, undistorted, result) -> None:
                if raw is not None:
                    cv2.imwrite(str(run_dir / f"{mode}_{index:03d}_raw.jpg"), raw)
                if undistorted is not None:
                    cv2.imwrite(
                        str(run_dir / f"{mode}_{index:03d}_annotated.jpg"),
                        annotate_box_centers(undistorted, result),
                    )

            measurement = measurer.measure(
                mode,
                args.target_letter if mode == "placement" else None,
                frame_callback=save_frame,
            )
            scene_results[mode] = asdict(measurement)
            print(
                f"[{mode}] ok={measurement.ok} reason={measurement.reason} "
                f"stable_frames={measurement.stable_frames}/7 centers={measurement.centers}"
            )
    finally:
        camera.release()
    result = {
        "schema_version": 1,
        "tool": "test_box_centers",
        "motion_command_count": 0,
        "run_dir": str(run_dir),
        "scenes": scene_results,
        "ok": bool(scene_results) and all(
            bool(value.get("ok")) for value in scene_results.values() if isinstance(value, dict)
        ),
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果已写入 {run_dir / 'result.json'}；motion_command_count=0")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
