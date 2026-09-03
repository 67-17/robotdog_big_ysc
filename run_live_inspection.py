from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from mission_lite3.audio import AudioReporter
from mission_lite3.camera import CameraSource
from mission_lite3.config_loader import load_config
from mission_lite3.round_result import (
    DEFAULT_LATEST_STOP_RESULT_PATH,
    DEFAULT_ROUND_RESULT_PATH,
    build_round_result,
    merge_record_into_round,
    load_round_result,
    write_empty_round_result,
    write_json_atomic,
    write_latest_stop_result,
)
from mission_lite3.vision import InspectionRecord, VisionPipeline
from mission_lite3.wide_camera import WideCameraUndistorter


WINDOW_NAME = "robot_runtime live inspection"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_video4linux_cameras(video_root: Path = Path("/sys/class/video4linux")) -> list[dict[str, object]]:
    cameras: list[dict[str, object]] = []
    if not video_root.exists():
        return cameras
    for item in sorted(video_root.glob("video*")):
        device = f"/dev/{item.name}"
        try:
            name = (item / "name").read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            name = ""
        cameras.append({"device": device, "name": name, "device_exists": Path(device).exists()})
    return cameras


def format_camera_listing(camera: dict[str, object]) -> str:
    status = "ready" if camera.get("device_exists") else "missing-dev-node"
    return f"{camera['device']}: {camera.get('name') or 'unknown'} [{status}]"


def find_camera_device_by_name(camera_name: str) -> Optional[str]:
    target = camera_name.strip().lower()
    for camera in list_video4linux_cameras():
        if target in str(camera.get("name") or "").lower():
            return str(camera["device"])
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live single-stop inspection runner")
    parser.add_argument("--config-dir", type=Path, default=None, help="Directory containing field.yaml and robot.yaml")
    parser.add_argument("--camera", default=None, help="Camera/video/image source; defaults to camera.front in config")
    parser.add_argument("--camera-name", default=None, help="Prefer a V4L2 camera whose sysfs name contains this text")
    parser.add_argument("--list-cameras", action="store_true", help="List V4L2 cameras and exit")
    parser.add_argument("--source-camera", default="front", help="Source camera label written to JSON results")
    parser.add_argument("--round-result", type=Path, default=None, help="Path to round_result.json")
    parser.add_argument("--latest-result", type=Path, default=None, help="Path to latest_stop_result.json")
    parser.add_argument("--evidence-dir", type=Path, default=Path("evidence"), help="Directory for stable-result evidence images")
    parser.add_argument("--reset-round", action="store_true", help="Atomically reset round_result.json to an empty not-ready round")
    parser.add_argument("--once", action="store_true", help="Read one frame/pass and exit")
    parser.add_argument("--exit-after-stable", action="store_true", help="Exit after the first stable inspection result")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum live frames to process, 0 means until q/Ctrl-C")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Maximum live processing time, 0 means no time limit",
    )
    parser.add_argument("--max-read-failures", type=int, default=30, help="Consecutive camera read failures before nonzero exit")
    parser.add_argument("--debug-frame-dir", type=Path, default=None, help="Save raw frames and metadata when no stable result is produced")
    parser.add_argument("--debug-frame-interval", type=float, default=2.0, help="Minimum seconds between saved debug frames")
    parser.add_argument("--no-window", action="store_true", help="Do not open OpenCV preview windows")
    parser.add_argument("--json-terminal", action="store_true", help="Print stable results as JSON lines")
    parser.add_argument("--no-speak", action="store_true", help="Disable speech output")
    parser.add_argument(
        "--no-speak-on-repeat",
        dest="no_speak_on_repeat",
        action="store_true",
        default=True,
        help="Suppress repeated speech for the same area/status result",
    )
    parser.add_argument(
        "--speak-on-repeat",
        dest="no_speak_on_repeat",
        action="store_false",
        help="Allow repeated speech for the same area/status result",
    )
    return parser


class LiveInspectionRunner:
    def __init__(self, args: argparse.Namespace, config: dict[str, Any]):
        self.args = args
        self.config = config
        inspection_cfg = config.get("inspection", {})
        self.round_result_path = args.round_result or Path(
            str(inspection_cfg.get("round_result_path", DEFAULT_ROUND_RESULT_PATH))
        )
        self.latest_result_path = args.latest_result or Path(
            str(inspection_cfg.get("latest_stop_result_path", DEFAULT_LATEST_STOP_RESULT_PATH))
        )
        existing_round = load_round_result(self.round_result_path, source_camera=args.source_camera)
        self.run_id = (
            uuid.uuid4().hex
            if args.reset_round
            else str(existing_round.get("run_id") or uuid.uuid4().hex)
        )
        self.window_enabled = self._window_enabled()
        self.vision = VisionPipeline(config)
        self.inspection_undistorter = None
        if bool(inspection_cfg.get("use_wide_undistortion", False)):
            camera_cfg = config.get("camera", {})
            calibration_path = Path(str(camera_cfg["wide_calibration"])).expanduser()
            if not calibration_path.is_absolute():
                calibration_path = Path(__file__).resolve().parent / calibration_path
            self.inspection_undistorter = WideCameraUndistorter.from_file(
                calibration_path
            )
        self.audio = AudioReporter(config, dry_run=args.no_speak)
        self.announced: set[tuple[str, str, str]] = set()
        self._last_debug_frame_at: Optional[float] = None

    def _window_enabled(self) -> bool:
        if self.args.no_window:
            return False
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            print("[inspection] DISPLAY is not set; running --no-window")
            return False
        return True

    def run(self) -> int:
        if self.args.reset_round:
            write_empty_round_result(
                self.round_result_path,
                self.args.source_camera,
                block_reason="reset_round",
                run_id=self.run_id,
            )

        source = self._source()
        source_path = Path(str(source))
        if isinstance(source, str) and source_path.exists() and source_path.suffix.lower() in IMAGE_SUFFIXES:
            return self._run_image(source_path)
        return self._run_capture(source)

    def _source(self) -> Any:
        if self.args.camera_name:
            matched = find_camera_device_by_name(self.args.camera_name)
            if matched is None:
                known = ", ".join(format_camera_listing(camera) for camera in list_video4linux_cameras())
                raise SystemExit(f"No camera name contains {self.args.camera_name!r}. Known cameras: {known}")
            print(json.dumps({"camera_name": self.args.camera_name, "camera_device": matched}, ensure_ascii=False))
            return matched
        source = self.args.camera
        if source is None:
            source = self.config["camera"]["front"]
        if isinstance(source, str):
            try:
                return int(source)
            except ValueError:
                return source
        return source

    def _run_image(self, path: Path) -> int:
        import cv2 as cv

        frame = cv.imread(str(path))
        if frame is None:
            print(f"[inspection] failed to read image: {path}")
            write_empty_round_result(
                self.round_result_path,
                self.args.source_camera,
                block_reason="camera_failed",
                run_id=self.run_id,
            )
            return 2
        frame = self._prepare_inspection_frame(frame)

        # A static image is replayed enough times for TemporalConsensus-style votes.
        passes = max(1, int(self.config["vision"]["stable_window"]))
        stable_record = None
        for _ in range(passes):
            record = self._handle_frame(frame)
            if record is not None:
                stable_record = record
            if record is not None and self.args.exit_after_stable:
                break
        if self.window_enabled:
            preview = frame.copy()
            cv.imshow(WINDOW_NAME, preview)
            cv.waitKey(0)
            cv.destroyAllWindows()
        if stable_record is None:
            self._mark_round_block("inspection_not_stable")
            return 4
        return 0

    def _run_capture(self, source: Any) -> int:
        import cv2 as cv

        camera_cfg = self.config.get("camera", {})
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
            print(f"[inspection] failed to open camera source: {source}")
            write_empty_round_result(
                self.round_result_path,
                self.args.source_camera,
                block_reason="camera_failed",
                run_id=self.run_id,
            )
            return 2

        if self.window_enabled:
            cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
        frame_count = 0
        read_failures = 0
        stable_seen = False
        bounded_exit = False
        max_seconds = max(0.0, float(self.args.max_seconds))
        deadline = time.monotonic() + max_seconds if max_seconds > 0.0 else None
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    bounded_exit = True
                    break
                frame = camera.read()
                if frame is None:
                    read_failures += 1
                    print(f"[inspection] camera_read_failed {read_failures}/{self.args.max_read_failures}")
                    if read_failures >= self.args.max_read_failures:
                        write_empty_round_result(
                            self.round_result_path,
                            self.args.source_camera,
                            block_reason="camera_failed",
                            run_id=self.run_id,
                        )
                        return 3
                    time.sleep(0.05)
                    continue
                read_failures = 0
                frame_count += 1
                frame = self._prepare_inspection_frame(frame)
                record = self._handle_frame(frame)
                if record is not None:
                    stable_seen = True
                if record is None:
                    self._maybe_write_debug_frame(frame, "no_stable_result", frame_count)
                if self.window_enabled:
                    preview = self._draw_preview(frame, record, frame_count)
                    cv.imshow(WINDOW_NAME, preview)
                    if cv.waitKey(1) & 0xFF == ord("q"):
                        break
                if record is not None and self.args.exit_after_stable:
                    break
                if self.args.once:
                    bounded_exit = True
                    break
                if self.args.max_frames > 0 and frame_count >= self.args.max_frames:
                    bounded_exit = True
                    break
        finally:
            camera.release()
            if self.window_enabled:
                cv.destroyAllWindows()
        if bounded_exit and not stable_seen:
            self._mark_round_block("inspection_not_stable")
            return 4
        return 0

    def _prepare_inspection_frame(self, frame):
        if self.inspection_undistorter is None:
            return frame
        return self.inspection_undistorter.apply(frame)

    def _mark_round_block(self, block_reason: str) -> None:
        current = load_round_result(self.round_result_path, source_camera=self.args.source_camera)
        records = current.get("records") if current.get("run_id") == self.run_id else {}
        data = build_round_result(
            records or {},
            source_camera=self.args.source_camera,
            block_reason=block_reason,
            run_id=self.run_id,
        )
        write_json_atomic(self.round_result_path, data)
        print(f"[inspection] round blocked reason={block_reason} path={self.round_result_path}")

    def _handle_frame(self, frame) -> Optional[InspectionRecord]:
        record = self.vision.inspect_frame(frame, source_camera=self.args.source_camera)
        if record is None:
            return None
        evidence_image = self._write_evidence(record, frame)
        if evidence_image is not None:
            record = replace(record, evidence_image=str(evidence_image))

        latest_data = write_latest_stop_result(self.latest_result_path, record)
        round_data = merge_record_into_round(
            self.round_result_path,
            record,
            source_camera=self.args.source_camera,
            run_id=self.run_id,
        )
        speech_error = self._speak(record)
        if speech_error:
            latest_data["speech_error"] = speech_error
            round_data["speech_error"] = speech_error
            write_json_atomic(self.latest_result_path, latest_data)
            write_json_atomic(self.round_result_path, round_data)

        self._print_result(latest_data, round_data)
        return record

    def _write_evidence(self, record: InspectionRecord, frame) -> Optional[Path]:
        import cv2 as cv

        self.args.evidence_dir.mkdir(parents=True, exist_ok=True)
        safe_timestamp = (record.timestamp or str(time.time())).replace(":", "").replace("+", "Z")
        # Store the analyzed pixels losslessly: meter status can sit directly
        # on a color-sector boundary, where JPEG artifacts are material.
        path = self.args.evidence_dir / f"{safe_timestamp}_{record.letter}_{record.state}.png"
        if cv.imwrite(str(path), frame):
            return path
        print(f"[inspection] failed to write evidence image: {path}")
        return None

    def _maybe_write_debug_frame(self, frame, reason: str, frame_count: int) -> None:
        if self.args.debug_frame_dir is None:
            return
        now = time.monotonic()
        if (
            self._last_debug_frame_at is not None
            and now - self._last_debug_frame_at < max(0.0, float(self.args.debug_frame_interval))
        ):
            return
        import cv2 as cv
        self.args.debug_frame_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stem = f"debug_{reason}_{timestamp}_{time.time_ns() % 1_000_000_000:09d}"
        image_path = self.args.debug_frame_dir / f"{stem}.jpg"
        metadata_path = self.args.debug_frame_dir / f"{stem}.json"
        if not cv.imwrite(str(image_path), frame):
            print(f"[inspection] failed to write debug frame: {image_path}")
            return
        metadata_path.write_text(
            json.dumps(
                {
                    "reason": reason,
                    "frame_count": frame_count,
                    "source_camera": self.args.source_camera,
                    "image": str(image_path),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._last_debug_frame_at = now
        print(f"[inspection] debug_frame={image_path} reason={reason}")

    def _speak(self, record: InspectionRecord) -> Optional[str]:
        if self.args.no_speak:
            return None
        speech_key = (record.letter, record.level, record.state)
        if self.args.no_speak_on_repeat and speech_key in self.announced:
            return None
        self.announced.add(speech_key)
        return self.audio.say_record(record)

    def _print_result(self, latest_data: dict[str, Any], round_data: dict[str, Any]) -> None:
        if self.args.json_terminal:
            event = {
                "event": "stable_inspection",
                "latest_stop_result": latest_data,
                "round_ready": round_data.get("ready"),
                "abnormal_areas": round_data.get("abnormal_areas", []),
                "unknown_areas": round_data.get("unknown_areas", []),
                "block_reason": round_data.get("block_reason"),
            }
            print(json.dumps(event, ensure_ascii=False, sort_keys=True))
            return
        print(
            "[inspection] "
            f"{latest_data['letter']} level={latest_data['level']} state={latest_data['state']} "
            f"ready={round_data.get('ready')} abnormal={round_data.get('abnormal_areas')} "
            f"unknown={round_data.get('unknown_areas')} block_reason={round_data.get('block_reason')}"
        )

    def _draw_preview(self, frame, record: Optional[InspectionRecord], frame_count: int):
        import cv2 as cv

        preview = frame.copy()
        if record is None:
            lines = [f"frame: {frame_count}", "result: detecting..."]
            color = (0, 220, 255)
        else:
            level = {"正常": "normal", "偏低": "low", "偏高": "high"}.get(record.level, record.level)
            state = {"正常": "normal", "异常": "abnormal"}.get(record.state, record.state)
            lines = [
                f"frame: {frame_count}",
                f"result: {record.letter} {level} {state}",
                f"confidence: {record.confidence:.2f}",
            ]
            color = (0, 255, 0) if record.state == "正常" else (0, 0, 255)
        x, y = 18, 34
        line_height = 34
        width = max(360, max(len(line) for line in lines) * 18)
        height = line_height * len(lines) + 18
        cv.rectangle(preview, (8, 8), (8 + width, 8 + height), (0, 0, 0), -1)
        cv.rectangle(preview, (8, 8), (8 + width, 8 + height), color, 2)
        for index, line in enumerate(lines):
            cv.putText(
                preview,
                line,
                (x, y + index * line_height),
                cv.FONT_HERSHEY_SIMPLEX,
                0.85,
                color if index == 1 else (255, 255, 255),
                2,
                cv.LINE_AA,
            )
        return preview


def main() -> int:
    args = build_parser().parse_args()
    if args.list_cameras:
        for camera in list_video4linux_cameras():
            print(format_camera_listing(camera))
        return 0
    config = load_config(args.config_dir)
    return LiveInspectionRunner(args, config).run()


if __name__ == "__main__":
    raise SystemExit(main())
