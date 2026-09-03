from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..config_loader import load_config


HSV_KEYS = ("cone_hsv", "red_hsv_1", "red_hsv_2", "green_hsv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive HSV debugger for competition targets")
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--source", default="camera", help="camera, device index, video path, image path, or RTSP URL")
    parser.add_argument("--hsv-key", choices=HSV_KEYS, default="red_hsv_1")
    parser.add_argument("--frames", type=int, default=0, help="Frame limit, 0 means until q")
    parser.add_argument("--headless", action="store_true", help="Print configured HSV and exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config_dir)
    hsv_cfg = config["vision"][args.hsv_key]
    print(f"[vision-debug] {args.hsv_key}: lower={hsv_cfg['lower']} upper={hsv_cfg['upper']} min_area={hsv_cfg['min_area']}")
    if args.headless:
        return 0

    import cv2 as cv
    import numpy as np

    source = _capture_source(config, args.source)
    image_path = Path(str(source))
    image = None
    if isinstance(source, str) and image_path.exists() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        image = cv.imread(str(image_path))
        if image is None:
            print(f"Failed to read image: {image_path}")
            return 2
        cap = None
    else:
        cap = cv.VideoCapture(source)
        if not cap.isOpened():
            print(f"Failed to open source: {source}")
            return 2

    cv.namedWindow("mask")
    cv.namedWindow("preview")
    lower = list(hsv_cfg["lower"])
    upper = list(hsv_cfg["upper"])
    names = ["lh", "ls", "lv", "uh", "us", "uv"]
    values = lower + upper
    limits = [179, 255, 255, 179, 255, 255]
    for name, value, limit in zip(names, values, limits):
        cv.createTrackbar(name, "mask", int(value), limit, lambda _value: None)

    frame_id = 0
    try:
        while True:
            if image is not None:
                frame = image.copy()
            else:
                ok, frame = cap.read()
                if not ok:
                    print("[vision-debug] frame read failed")
                    return 3
            frame_id += 1
            current = [cv.getTrackbarPos(name, "mask") for name in names]
            lo = np.array(current[:3], dtype=np.uint8)
            hi = np.array(current[3:], dtype=np.uint8)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, lo, hi)
            preview = cv.bitwise_and(frame, frame, mask=mask)
            cv.imshow("mask", mask)
            cv.imshow("preview", preview)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                print(f"[vision-debug] lower={current[:3]} upper={current[3:]}")
            if image is not None:
                cv.waitKey(30)
            if args.frames > 0 and frame_id >= args.frames:
                break
    finally:
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()
    return 0


def _capture_source(config: dict, source_arg: str) -> Any:
    if source_arg == "camera":
        return config["camera"]["front"]
    try:
        return int(source_arg)
    except ValueError:
        return source_arg


if __name__ == "__main__":
    raise SystemExit(main())
