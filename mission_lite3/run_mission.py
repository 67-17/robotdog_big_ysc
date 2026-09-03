from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

from .config_loader import ConfigError, load_config
from .camera import CameraSource
from .mission import LargeQuadrupedMission
from .vision import VisionPipeline
from .vision.common import Detection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lite3 large quadruped competition mission runner")
    parser.add_argument("--config-dir", type=Path, default=None, help="Directory containing field.yaml and robot.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run the mission state machine without hardware output")
    parser.add_argument("--robot", action="store_true", help="Run on robot hardware")
    parser.add_argument(
        "--vision-test",
        nargs="?",
        const="camera",
        default=None,
        metavar="SOURCE",
        help="Run vision on an image/video/camera source; omit SOURCE to use the configured front camera",
    )
    parser.add_argument("--vision-frames", type=int, default=0, help="Number of frames for live vision test, 0 means until q")
    parser.add_argument("--headless", action="store_true", help="Do not open OpenCV preview windows during vision test")
    parser.add_argument("--udp-fallback", action="store_true", help="Use direct UDP instead of ROS2 /cmd_vel")
    parser.add_argument("--axis-fallback", action="store_true", help="Use UDP axis commands instead of complex velocity commands")
    parser.add_argument("--skip-arm", action="store_true", help="Disable mechanical arm operations")
    parser.add_argument("--ignore-obstacles", action="store_true", help="Bypass obstacle checks during route testing")
    parser.add_argument(
        "--allow-open-loop",
        action="store_true",
        help="Use time-based action termination while retaining sensor safety checks; intended only for controlled calibration",
    )
    parser.add_argument(
        "--inspection-window",
        action="store_true",
        help="Show a live front-camera preview with inspection recognition overlay during inspection stops",
    )
    parser.add_argument(
        "--ignore-ultrasound-obstacle",
        action="store_true",
        help="Ignore front ultrasound obstacle triggers but keep vision obstacle checks",
    )
    return parser


def run_vision_test(config: dict, source_arg: str, max_frames: int = 0, headless: bool = False) -> int:
    import cv2 as cv

    if not headless and os.name != "nt" and not os.environ.get("DISPLAY"):
        print("[vision-test] DISPLAY is not set; running headless")
        headless = True
    vision = VisionPipeline(config)
    image_path = Path(source_arg)
    if source_arg != "camera" and image_path.exists() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        image = cv.imread(str(image_path))
        if image is None:
            print(f"Failed to read image: {image_path}")
            return 2
        for _ in range(int(config["vision"]["stable_window"])):
            result = _inspect_and_draw(vision, image)
        print(result["summary"])
        if not headless:
            cv.imshow("mission_lite3 vision-test", result["frame"])
            cv.waitKey(0)
        return 0

    source = _capture_source(config, source_arg)
    camera_cfg = config.get("camera", {})
    camera = CameraSource(
        source,
        int(camera_cfg.get("frame_width", 0)) or None,
        int(camera_cfg.get("frame_height", 0)) or None,
        flush_grab_frames=int(camera_cfg.get("flush_grab_frames", 2)),
        stale_frame_reconnect_count=int(camera_cfg.get("stale_frame_reconnect_count", 15)),
        digital_zoom=float(camera_cfg.get("digital_zoom", 1.0)),
        open_timeout_ms=int(camera_cfg.get("open_timeout_ms", 3000)),
        read_timeout_ms=int(camera_cfg.get("read_timeout_ms", 2000)),
        reconnect_backoff_s=float(camera_cfg.get("reconnect_backoff_s", 0.25)),
    )
    if not camera.open():
        print(f"Failed to open vision source: {source}")
        return 2
    frame_id = 0
    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("[vision-test] frame read failed")
                return 3
            frame_id += 1
            result = _inspect_and_draw(vision, frame)
            if frame_id == 1 or frame_id % 15 == 0:
                print(result["summary"])
            if not headless:
                cv.imshow("mission_lite3 vision-test", result["frame"])
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
            if max_frames > 0 and frame_id >= max_frames:
                break
    finally:
        camera.release()
        if not headless:
            cv.destroyAllWindows()
    return 0


def _capture_source(config: dict, source_arg: str) -> Any:
    if source_arg == "camera":
        return config["camera"]["front"]
    try:
        return int(source_arg)
    except ValueError:
        return source_arg


def _inspect_and_draw(vision: VisionPipeline, frame) -> dict[str, Any]:
    record = vision.inspect_frame(frame, source_camera="vision-test")
    cones = vision.detect_cones(frame)
    red_bars = vision.detect_red_bars(frame)
    green_bars = vision.detect_green_bars(frame)
    preview = frame.copy()
    _draw_detections(preview, cones, (0, 165, 255))
    _draw_detections(preview, red_bars, (0, 0, 255))
    _draw_detections(preview, green_bars, (0, 255, 0))
    summary = f"inspection={record} cones={len(cones)} red_bars={len(red_bars)} green_bars={len(green_bars)}"
    return {"frame": preview, "summary": summary}


def _draw_detections(frame, detections: Iterable[Detection], color: tuple[int, int, int]) -> None:
    import cv2 as cv

    for det in detections:
        if det.bbox is None:
            continue
        box = det.bbox
        cv.rectangle(frame, (box.x, box.y), (box.x + box.w, box.y + box.h), color, 2)
        cv.putText(frame, det.label, (box.x, max(20, box.y - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config_dir)
    except ConfigError as exc:
        print(f"[config] {exc}")
        return 2
    if args.vision_test is not None:
        return run_vision_test(config, args.vision_test, args.vision_frames, args.headless)
    dry_run = args.dry_run or not args.robot
    mission = LargeQuadrupedMission(
        config,
        dry_run=dry_run,
        udp_fallback=args.udp_fallback,
        axis_fallback=args.axis_fallback,
        skip_arm=args.skip_arm,
        ignore_obstacles=args.ignore_obstacles,
        ignore_ultrasound_obstacle=args.ignore_ultrasound_obstacle,
        show_inspection_window=args.inspection_window,
        allow_open_loop=args.allow_open_loop,
    )
    result = mission.run()
    print(
        "[mission] result "
        f"status={result.status} state={result.state} reason={result.reason!r} "
        f"placed={result.placed_letters} carried_bar={result.carried_bar}"
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
