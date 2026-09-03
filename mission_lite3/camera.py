from __future__ import annotations

import time
from typing import Any, Optional

from .persistent_camera import PersistentLatestFrameReader


class CameraSource:
    def __init__(
        self,
        source: Any,
        width: int | None = None,
        height: int | None = None,
        dry_run: bool = False,
        flush_grab_frames: int = 0,
        stale_frame_reconnect_count: int = 0,
        digital_zoom: float = 1.0,
        open_timeout_ms: int = 3000,
        read_timeout_ms: int = 2000,
        reconnect_backoff_s: float = 0.25,
        persistent_latest: bool = False,
        shared_camera: Optional['CameraSource'] = None,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.dry_run = dry_run
        self.flush_grab_frames = max(0, int(flush_grab_frames))
        self.stale_frame_reconnect_count = max(0, int(stale_frame_reconnect_count))
        self.digital_zoom = max(1.0, float(digital_zoom))
        self.open_timeout_ms = max(0, int(open_timeout_ms))
        self.read_timeout_ms = max(0, int(read_timeout_ms))
        self.reconnect_backoff_s = max(0.0, float(reconnect_backoff_s))
        self.cap = None
        self._last_signature: Optional[bytes] = None
        self._same_signature_count = 0
        self._last_reconnect_at: Optional[float] = None
        self.last_frame_at: Optional[float] = None
        self._last_frame_sequence = 0
        self._owns_persistent_reader = shared_camera is None
        if shared_camera is not None:
            self._persistent_reader = shared_camera._persistent_reader
        elif persistent_latest and not dry_run:
            self._persistent_reader = PersistentLatestFrameReader(
                source,
                int(width or 1280),
                int(height or 720),
                self.open_timeout_ms,
                self.read_timeout_ms,
                self.reconnect_backoff_s,
            )
        else:
            self._persistent_reader = None

    def open(self) -> bool:
        if self.dry_run:
            return False
        if self._persistent_reader is not None:
            return self._persistent_reader.start('open')
        import cv2 as cv

        self.release()
        self.cap = cv.VideoCapture()
        for property_name, value in (
            ('CAP_PROP_OPEN_TIMEOUT_MSEC', self.open_timeout_ms),
            ('CAP_PROP_READ_TIMEOUT_MSEC', self.read_timeout_ms),
        ):
            property_id = getattr(cv, property_name, None)
            if property_id is not None and value > 0:
                try:
                    self.cap.set(property_id, value)
                except Exception:
                    pass
        self.cap.open(self.source)
        try:
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if self.width:
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        return bool(self.cap.isOpened())

    def ensure_running(self, reason: str = 'reuse') -> bool:
        if self.dry_run:
            return False
        if self._persistent_reader is not None:
            return self._persistent_reader.ensure_running(reason)
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    return True
            except Exception:
                pass
        return self.open()

    def read(self, timeout_s: Optional[float] = None) -> Optional[Any]:
        if self._persistent_reader is not None:
            result = self._persistent_reader.read_latest(
                self._last_frame_sequence,
                timeout_s,
            )
            if result is None:
                return None
            frame, sequence, captured_at = result
            self._last_frame_sequence = sequence
            self.last_frame_at = captured_at
            if self._is_stale_frame(frame):
                self._reopen('stale_frame')
                return None
            return self._apply_digital_zoom(frame)
        if self.cap is None:
            if not self.open():
                return None
        ok, frame = self._read_latest_frame()
        if not ok or frame is None:
            self._reopen('read_failed')
            return None
        if self._is_stale_frame(frame):
            self._reopen('stale_frame')
            return None
        self.last_frame_at = time.monotonic()
        return self._apply_digital_zoom(frame)

    def read_latest(self, timeout_s: Optional[float] = None) -> Optional[Any]:
        return self.read(timeout_s=timeout_s)

    def release(self) -> None:
        if self._persistent_reader is not None and not self._owns_persistent_reader:
            return
        if self._persistent_reader is not None:
            self._persistent_reader.stop()
            self._last_frame_sequence = 0
        elif self.cap is not None:
            self.cap.release()
            self.cap = None
        self._last_signature = None
        self._same_signature_count = 0
        self.last_frame_at = None

    def reconnect(self, reason: str = 'manual') -> None:
        self._reopen(reason)

    def status_snapshot(self) -> dict[str, Any]:
        if self._persistent_reader is not None:
            return self._persistent_reader.status_snapshot()
        return {
            'status': 'open' if self.cap is not None else 'closed',
            'last_frame_at': self.last_frame_at,
        }

    def _read_latest_frame(self) -> tuple[bool, Optional[Any]]:
        if self.cap is None:
            return False, None
        for _ in range(self.flush_grab_frames):
            if not self.cap.grab():
                break
        if self.flush_grab_frames:
            ok, frame = self.cap.retrieve()
            if ok:
                return True, frame
        return self.cap.read()

    def _is_stale_frame(self, frame: Any) -> bool:
        if self.stale_frame_reconnect_count <= 0:
            return False
        signature = self._frame_signature(frame)
        if signature == self._last_signature:
            self._same_signature_count += 1
        else:
            self._last_signature = signature
            self._same_signature_count = 0
        return self._same_signature_count >= self.stale_frame_reconnect_count

    def _frame_signature(self, frame: Any) -> bytes:
        import cv2 as cv

        sample = cv.resize(frame, (32, 18), interpolation=cv.INTER_AREA)
        return sample.tobytes()

    def _apply_digital_zoom(self, frame: Any) -> Any:
        if self.digital_zoom <= 1.0:
            return frame
        import cv2 as cv

        height, width = frame.shape[:2]
        crop_width = max(1, int(width / self.digital_zoom))
        crop_height = max(1, int(height / self.digital_zoom))
        x0 = max(0, (width - crop_width) // 2)
        y0 = max(0, (height - crop_height) // 2)
        cropped = frame[y0 : y0 + crop_height, x0 : x0 + crop_width]
        return cv.resize(cropped, (width, height), interpolation=cv.INTER_LINEAR)

    def _reopen(self, reason: str) -> None:
        now = time.monotonic()
        if (
            self._last_reconnect_at is not None
            and now - self._last_reconnect_at < self.reconnect_backoff_s
        ):
            return
        self._last_reconnect_at = now
        self._last_signature = None
        self._same_signature_count = 0
        self._last_frame_sequence = 0
        if self._persistent_reader is not None:
            self._persistent_reader.force_restart(reason)
            return
        print(
            '[camera] reconnect source={} reason={}'.format(self.source, reason),
            flush=True,
        )
        self.open()
