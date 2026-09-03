from __future__ import annotations

import json
import math
import statistics
import threading
import time
import uuid
from collections import deque, namedtuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .controller import AvoidanceController
from .model import SensorFrame, TrackUpdate
from .tracking import TargetTracker
from .vision import ConeDetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CameraSnapshot = namedtuple("CameraSnapshot", "image timestamp sequence")


@dataclass(frozen=True)
class StartupAvoidanceResult:
    ok: bool
    avoidance_count: int
    reason: str
    log_path: Optional[str] = None


class LatestFrameCamera:
    """Continuously capture frames so inference never consumes stale backlog."""

    def __init__(self, config: dict, *, cv2_module=None, clock=None) -> None:
        self.config = config
        self._cv2 = cv2_module
        self._clock = clock or time.monotonic
        self._capture = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[CameraSnapshot] = None
        self._error: Optional[str] = None
        self._sequence = 0

    def open(self) -> None:
        if self._capture is not None:
            return
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        source = self.config["source"]
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        capture = self._cv2.VideoCapture(source)
        self._capture = capture
        try:
            if not capture.isOpened():
                raise RuntimeError(f"failed to open avoidance camera: {source}")
            capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, self.config["width"])
            capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self.config["height"])
            capture.set(self._cv2.CAP_PROP_FPS, self.config["fps"])
            ok, image = capture.read()
            if not ok or image is None:
                raise RuntimeError("avoidance camera opened but returned no frame")
            self._store_frame(image)
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="startup-avoidance-camera",
                daemon=True,
            )
            self._thread.start()
        except Exception:
            self.release()
            raise

    def read(self) -> CameraSnapshot:
        with self._lock:
            error = self._error
            snapshot = self._latest
        if error is not None:
            raise RuntimeError(error)
        if snapshot is None:
            raise RuntimeError("avoidance camera is not open")
        return snapshot

    def release(self) -> None:
        self._stop_event.set()
        capture = self._capture
        self._capture = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        with self._lock:
            self._latest = None
            self._error = None

    def _capture_loop(self) -> None:
        capture = self._capture
        if capture is None:
            return
        try:
            while not self._stop_event.is_set():
                ok, image = capture.read()
                if self._stop_event.is_set():
                    return
                if not ok or image is None:
                    self._set_error("avoidance camera frame read failed")
                    return
                self._store_frame(image)
        except Exception as exc:
            if not self._stop_event.is_set():
                self._set_error(f"avoidance camera capture failed: {exc}")
        finally:
            try:
                capture.release()
            except Exception:
                pass

    def _store_frame(self, image: Any) -> None:
        shape = getattr(image, "shape", ())
        if (
            len(shape) < 2
            or int(shape[0]) != int(self.config["height"])
            or int(shape[1]) != int(self.config["width"])
        ):
            raise RuntimeError(
                "avoidance camera frame size does not match configured width/height"
            )
        with self._lock:
            self._sequence += 1
            self._latest = CameraSnapshot(
                image, self._clock(), self._sequence
            )
            self._error = None

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message


class StartupAvoidanceRunner:
    def __init__(
        self,
        mission_config: dict,
        motion,
        state_reader,
        *,
        dry_run: bool = False,
        camera=None,
        detector=None,
        tracker=None,
        controller=None,
        clock=None,
        sleep=None,
        log_root: Optional[Path] = None,
        log_opener=None,
        max_hold_retries: Optional[int] = None,
    ) -> None:
        self.config = mission_config["startup_avoidance"]
        self.motion = motion
        self.state_reader = state_reader
        self.dry_run = bool(dry_run)
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self.camera = camera or LatestFrameCamera(
            self.config["image"], clock=self.clock
        )
        self.detector = detector or ConeDetector(self.config)
        self.tracker = tracker or TargetTracker(self.config)
        self.controller = controller or AvoidanceController(self.config)
        self.log_root = log_root
        self.log_opener = log_opener or (
            lambda path: path.open("a", encoding="utf-8")
        )
        self.max_hold_retries = max_hold_retries
        self.max_hold_s = float(self.config.get("fault_hold_max_s", 30.0))
        self._hold_started_at: Optional[float] = None
        self._ultrasound_values = deque(
            maxlen=int(self.config["distance"]["ultrasound_window"])
        )
        self._last_ultrasound_stamp: Optional[float] = None
        self._latest_ultrasound_valid = False

    def run(self) -> StartupAvoidanceResult:
        if self.dry_run:
            return StartupAvoidanceResult(
                True, 0, "dry-run simulated obstacle-zone crossing", None
            )

        log_stream = None
        log_path: Optional[Path] = None
        terminal_error: Optional[BaseException] = None
        hold_retries = 0
        try:
            log_path = self._new_log_path()
            log_stream = self.log_opener(log_path)
            while True:
                try:
                    self.camera.open()
                    return self._run_loop(log_stream, log_path)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    hold_retries += 1
                    hold_elapsed_s = self._hold_elapsed_s()
                    force_hold = getattr(self.controller, "force_hold", None)
                    if callable(force_hold):
                        force_hold(f"runtime recovery: {exc}")
                    self._write_json(
                        log_stream,
                        {
                            "event": "fault_hold",
                            "time": self.clock(),
                            "reason": str(exc),
                            "retry": hold_retries,
                            "hold_elapsed_s": hold_elapsed_s,
                            "hold_limit_s": self.max_hold_s,
                        },
                    )
                    try:
                        self.motion.stop()
                    except Exception as stop_error:
                        self._write_json(
                            log_stream,
                            {
                                "event": "fault_hold_stop_error",
                                "time": self.clock(),
                                "error": str(stop_error),
                            },
                        )
                    try:
                        self.camera.release()
                    except Exception as release_error:
                        self._write_json(
                            log_stream,
                            {
                                "event": "fault_hold_camera_release_error",
                                "time": self.clock(),
                                "error": str(release_error),
                            },
                        )
                    if (
                        self.max_hold_retries is not None
                        and hold_retries > self.max_hold_retries
                    ):
                        terminal_error = exc
                        self._write_json(
                            log_stream,
                            {
                                "event": "error",
                                "time": self.clock(),
                                "error": str(exc),
                            },
                        )
                        raise
                    if hold_elapsed_s >= self.max_hold_s:
                        timeout_error = RuntimeError(
                            "startup avoidance fault hold timed out after "
                            f"{hold_elapsed_s:.1f}s limit={self.max_hold_s:.1f}s: {exc}"
                        )
                        terminal_error = timeout_error
                        self._write_json(
                            log_stream,
                            {
                                "event": "error",
                                "time": self.clock(),
                                "error": str(timeout_error),
                            },
                        )
                        raise timeout_error from exc
                    self.sleep(float(self.config.get("fault_hold_retry_s", 0.5)))
        except BaseException as exc:
            terminal_error = exc
            raise
        finally:
            cleanup_errors = []
            try:
                self.motion.stop()
            except Exception as exc:
                cleanup_errors.append(f"motion stop failed: {exc}")
            try:
                self.camera.release()
            except Exception as exc:
                cleanup_errors.append(f"camera release failed: {exc}")
            if log_stream is not None:
                try:
                    log_stream.close()
                except Exception as exc:
                    cleanup_errors.append(f"log close failed: {exc}")
            if cleanup_errors and terminal_error is None:
                raise RuntimeError("; ".join(cleanup_errors))

    def _run_loop(self, log_stream, log_path: Path) -> StartupAvoidanceResult:
        period = 1.0 / float(self.config["image"]["fps"])
        last_camera_sequence = None
        last_update = TrackUpdate([], [], False)
        last_decision = None

        while True:
            tick_started = self.clock()
            safety_error = self.state_reader.safety_error(
                require_ultrasound=True,
                require_fresh=True,
            )
            state = self.state_reader.state
            camera_snapshot = self.camera.read()
            is_new_frame = camera_snapshot.sequence != last_camera_sequence
            if is_new_frame:
                detections = self.detector.detect(camera_snapshot.image)
                update = self.tracker.update(detections)
            else:
                update = TrackUpdate(
                    last_update.tracks, [], last_update.ambiguous
                )

            now = self.clock()
            ultrasound_m = self._median_ultrasound(state)
            frame = SensorFrame(
                now,
                update.tracks,
                update.cleared_ids,
                update.ambiguous,
                ultrasound_m,
                float(state.x),
                float(state.y),
                float(state.yaw),
                self._age(now, camera_snapshot.timestamp),
                self._age(now, state.ultrasound_updated_at),
                self._age(now, state.odom_updated_at),
            )
            if safety_error:
                force_hold = getattr(self.controller, "force_hold", None)
                if not callable(force_hold):
                    raise RuntimeError(f"robot state unsafe: {safety_error}")
                decision = force_hold(f"robot state unsafe: {safety_error}")
            elif is_new_frame:
                decision = self.controller.step(frame)
                self._synchronize_active_tracker(update.tracks)
                last_camera_sequence = camera_snapshot.sequence
                last_update = update
                last_decision = decision
            else:
                safety_decision = self.controller.check_safety(frame)
                decision = safety_decision or last_decision
                if decision is None:
                    raise RuntimeError("avoidance controller produced no decision")

            self.motion.move(decision.vx, decision.vy, decision.wz)
            self._write_iteration(log_stream, frame, update, decision)
            if decision.state == "HOLD":
                hold_elapsed_s = self._hold_elapsed_s(now)
                if hold_elapsed_s >= self.max_hold_s:
                    raise RuntimeError(
                        "startup avoidance controller hold timed out after "
                        f"{hold_elapsed_s:.1f}s limit={self.max_hold_s:.1f}s: "
                        f"{decision.reason}"
                    )
            else:
                self._hold_started_at = None
            if decision.fault:
                raise RuntimeError(
                    f"startup avoidance fault: {decision.reason}"
                )
            if decision.finished:
                return StartupAvoidanceResult(
                    True,
                    int(self.controller.avoidance_count),
                    decision.reason,
                    str(log_path),
                )

            remaining = period - (self.clock() - tick_started)
            if remaining > 0.0:
                self.sleep(remaining)

    def _hold_elapsed_s(self, now: Optional[float] = None) -> float:
        current = float(self.clock() if now is None else now)
        if self._hold_started_at is None:
            self._hold_started_at = current
        return max(0.0, current - float(self._hold_started_at))

    def _median_ultrasound(self, state) -> float:
        stamp = state.ultrasound_updated_at
        if stamp is None:
            self._latest_ultrasound_valid = False
            return float("nan")
        if stamp != self._last_ultrasound_stamp:
            self._last_ultrasound_stamp = stamp
            value = state.front_ultrasound_m
            self._latest_ultrasound_valid = (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            )
            if self._latest_ultrasound_valid:
                self._ultrasound_values.append(float(value))
        if not self._latest_ultrasound_valid or not self._ultrasound_values:
            return float("nan")
        return float(statistics.median(self._ultrasound_values))

    def _synchronize_active_tracker(self, tracks) -> None:
        active_track_id = self.controller.active_track_id
        if active_track_id is None:
            if self.tracker.active_id is not None:
                self.tracker.clear_active()
            return
        track_ids = {track.track_id for track in tracks}
        if (
            active_track_id in track_ids
            and self.tracker.active_id != active_track_id
        ):
            self.tracker.set_active(active_track_id)

    def _new_log_path(self) -> Path:
        log_dir = (
            Path(self.log_root)
            if self.log_root is not None
            else Path(str(self.config["log_dir"])).expanduser()
        )
        if not log_dir.is_absolute():
            log_dir = PROJECT_ROOT / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return log_dir / f"{stamp}_{uuid.uuid4().hex[:8]}.jsonl"

    def _write_iteration(self, stream, frame, update, decision) -> None:
        self._write_json(
            stream,
            {
                "event": "decision",
                "time": frame.now,
                "state": decision.state,
                "reason": decision.reason,
                "vx": decision.vx,
                "vy": decision.vy,
                "wz": decision.wz,
                "finished": decision.finished,
                "fault": decision.fault,
                "avoidance_count": self.controller.avoidance_count,
                "forward_progress_m": self._json_number(
                    self.controller.forward_progress_m
                ),
                "pass_progress_m": self._json_number(
                    getattr(self.controller, "pass_progress_m", None)
                ),
                "active_track_id": self.controller.active_track_id,
                "track_ids": [track.track_id for track in update.tracks],
                "cleared_track_ids": list(update.cleared_ids),
                "ambiguous": update.ambiguous,
                "ultrasound_m": self._json_number(frame.ultrasound_m),
                "odom_x": self._json_number(frame.odom_x),
                "odom_y": self._json_number(frame.odom_y),
                "yaw": self._json_number(frame.yaw),
                "image_age_s": self._json_number(frame.image_age_s),
                "ultrasound_age_s": self._json_number(
                    frame.ultrasound_age_s
                ),
                "odom_age_s": self._json_number(frame.odom_age_s),
                "return_line_error_m": self._json_number(
                    self.controller.return_line_error_m
                ),
            },
        )

    @staticmethod
    def _write_json(stream, record: dict) -> None:
        stream.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        )
        stream.flush()

    @staticmethod
    def _age(now: float, timestamp: Optional[float]) -> float:
        if timestamp is None:
            return float("inf")
        return now - float(timestamp)

    @staticmethod
    def _json_number(value):
        if (
            value is not None
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        ):
            return value
        return None
