from __future__ import annotations

import ctypes
import multiprocessing as mp
import time
from datetime import datetime
from queue import Empty, Full
from threading import Event, RLock, Thread, current_thread


def _status(queue, event, **fields):
    try:
        queue.put_nowait({'event': event, 'event_time_unix': time.time(), **fields})
    except Full:
        pass


def _capture_worker(
    source, width, height, open_timeout_ms, read_timeout_ms,
    reconnect_backoff_s, generation, stop_event, frame_ready,
    frame_buffer, active_slot, publish_version, sequence, captured_at,
    status_queue,
):
    import cv2 as cv
    import numpy as np

    _status(status_queue, 'worker_started', generation=generation)
    while not stop_event.is_set():
        cap = cv.VideoCapture()
        for name, value in (
            ('CAP_PROP_OPEN_TIMEOUT_MSEC', open_timeout_ms),
            ('CAP_PROP_READ_TIMEOUT_MSEC', read_timeout_ms),
        ):
            property_id = getattr(cv, name, None)
            if property_id is not None and value > 0:
                try:
                    cap.set(property_id, value)
                except Exception:
                    pass
        _status(status_queue, 'opening', generation=generation)
        try:
            cap.open(source)
            try:
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
            if not cap.isOpened():
                _status(status_queue, 'open_failed', generation=generation)
                if stop_event.wait(reconnect_backoff_s):
                    break
                continue
            _status(status_queue, 'connected', generation=generation)
            resize_reported = False
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    _status(status_queue, 'read_failed', generation=generation)
                    break
                if frame.ndim != 3 or frame.shape[2] != 3:
                    _status(
                        status_queue, 'invalid_frame', generation=generation,
                        shape=tuple(getattr(frame, 'shape', ())),
                    )
                    break
                if frame.shape[:2] != (height, width):
                    if not resize_reported:
                        _status(
                            status_queue, 'frame_resized', generation=generation,
                            source_width=int(frame.shape[1]),
                            source_height=int(frame.shape[0]),
                            output_width=width, output_height=height,
                        )
                        resize_reported = True
                    frame = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                frame_size = width * height * 3
                next_slot = 1 - int(active_slot.value)
                target = np.frombuffer(
                    frame_buffer,
                    dtype=np.uint8,
                    count=frame_size,
                    offset=next_slot * frame_size,
                )
                target[:] = frame.reshape(-1)
                publish_version.value += 1
                captured_at.value = time.monotonic()
                sequence.value += 1
                active_slot.value = next_slot
                publish_version.value += 1
                frame_ready.set()
        except BaseException as exc:
            _status(
                status_queue, 'worker_error', generation=generation,
                error='{}: {}'.format(type(exc).__name__, exc),
            )
        finally:
            try:
                cap.release()
            except Exception:
                pass
        if not stop_event.is_set():
            _status(
                status_queue, 'reconnect_wait', generation=generation,
                backoff_s=reconnect_backoff_s,
            )
            if stop_event.wait(reconnect_backoff_s):
                break
    _status(status_queue, 'worker_stopped', generation=generation)


class PersistentLatestFrameReader:
    def __init__(
        self, source, width, height, open_timeout_ms=3000,
        read_timeout_ms=2000, reconnect_backoff_s=0.25,
    ):
        self.source = source
        self.width = int(width)
        self.height = int(height)
        self.open_timeout_ms = max(1, int(open_timeout_ms))
        self.read_timeout_ms = max(1, int(read_timeout_ms))
        self.reconnect_backoff_s = max(0.0, float(reconnect_backoff_s))
        self._context = mp.get_context('spawn')
        self._frame_size = self.width * self.height * 3
        self._frame_buffer = self._context.RawArray(
            ctypes.c_ubyte, self._frame_size * 2
        )
        self._active_slot = self._context.Value('b', 0, lock=False)
        self._publish_version = self._context.Value('Q', 0, lock=False)
        self._sequence = self._context.Value('Q', 0, lock=False)
        self._captured_at = self._context.Value('d', 0.0, lock=False)
        self._frame_ready = self._context.Event()
        self._worker_stop = self._context.Event()
        self._status_queue = self._context.Queue(maxsize=64)
        self._process = None
        self._monitor = None
        self._monitor_stop = Event()
        self._lock = RLock()
        self._status_lock = RLock()
        self._generation = 0
        self._generation_started_at = 0.0
        self._last_status = 'closed'
        self._consecutive_failures = 0
        self._total_failures = 0
        self._reconnect_count = 0
        self._read_timeout_count = 0

    def start(self, reason='open'):
        with self._lock:
            if self._process is not None and self._process.is_alive():
                return True
            self._monitor_stop.clear()
            self._start_worker(reason)
            if self._monitor is None or not self._monitor.is_alive():
                self._monitor = Thread(
                    target=self._monitor_loop,
                    name='persistent-camera-supervisor',
                    daemon=True,
                )
                self._monitor.start()
            return self._process is not None and self._process.is_alive()

    def ensure_running(self, reason):
        running = self.start(reason)
        if running:
            self._log('reader_reused', reason=reason)
        return running

    def force_restart(self, reason):
        with self._lock:
            self._restart_worker(reason)

    def read_latest(self, after_sequence, timeout_s=None):
        import numpy as np

        if not self.start('read'):
            return None
        timeout = self.read_timeout_ms / 1000.0
        if timeout_s is not None:
            timeout = max(0.0, float(timeout_s))
        deadline = time.monotonic() + timeout
        while True:
            self._drain_status()
            frame_result = self._copy_published_frame(int(after_sequence), np)
            if frame_result is not None:
                frame, frame_sequence, frame_time = frame_result
                if self._last_status == 'consumer_timeout':
                    recovered = self._consecutive_failures
                    self._consecutive_failures = 0
                    self._last_status = 'frame_recovered'
                    self._log(
                        'frame_recovered',
                        recovered_after_failures=recovered,
                    )
                return frame, frame_sequence, frame_time
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self._read_timeout_count += 1
                self._total_failures += 1
                self._consecutive_failures += 1
                self._last_status = 'consumer_timeout'
                self._log(
                    'consumer_timeout', timeout_s='{:.3f}'.format(timeout),
                    after_sequence=int(after_sequence),
                )
                return None
            self._frame_ready.wait(min(remaining, 0.05))
            self._frame_ready.clear()

    def _copy_published_frame(self, after_sequence, np):
        for _attempt in range(3):
            version_before = int(self._publish_version.value)
            if version_before % 2:
                continue
            slot_before = int(self._active_slot.value)
            sequence_before = int(self._sequence.value)
            captured_before = float(self._captured_at.value)
            if sequence_before <= after_sequence or captured_before <= 0.0:
                return None
            frame = np.frombuffer(
                self._frame_buffer,
                dtype=np.uint8,
                count=self._frame_size,
                offset=slot_before * self._frame_size,
            ).copy()
            slot_after = int(self._active_slot.value)
            sequence_after = int(self._sequence.value)
            captured_after = float(self._captured_at.value)
            version_after = int(self._publish_version.value)
            if (
                version_before == version_after
                and version_after % 2 == 0
                and slot_before == slot_after
                and sequence_before == sequence_after
                and captured_before == captured_after
            ):
                return (
                    frame.reshape((self.height, self.width, 3)),
                    sequence_after,
                    captured_after,
                )
        return None

    def stop(self):
        self._monitor_stop.set()
        monitor = self._monitor
        with self._lock:
            self._stop_worker('mission_cleanup')
            self._drain_status()
        if monitor is not None and monitor is not current_thread():
            monitor.join(timeout=1.0)
        self._monitor = None
        self._last_status = 'closed'
        self._log('reader_closed')

    def status_snapshot(self):
        process = self._process
        frame_time = float(self._captured_at.value)
        return {
            'status': self._last_status,
            'generation': self._generation,
            'process_alive': bool(process is not None and process.is_alive()),
            'frame_sequence': int(self._sequence.value),
            'last_frame_at': frame_time or None,
            'last_frame_age_s': (
                None if frame_time <= 0.0 else max(0.0, time.monotonic() - frame_time)
            ),
            'consecutive_failures': self._consecutive_failures,
            'total_failures': self._total_failures,
            'reconnect_count': self._reconnect_count,
            'read_timeout_count': self._read_timeout_count,
        }

    def _monitor_loop(self):
        while not self._monitor_stop.wait(0.10):
            self._drain_status()
            with self._lock:
                process = self._process
                if process is None:
                    continue
                if not process.is_alive():
                    self._total_failures += 1
                    self._consecutive_failures += 1
                    self._restart_worker('worker_exit:{}'.format(process.exitcode))
                    continue
                now = time.monotonic()
                frame_time = float(self._captured_at.value)
                if frame_time <= 0.0:
                    stale_for = now - self._generation_started_at
                    limit = self.open_timeout_ms / 1000.0
                    reason = 'open_watchdog_timeout'
                else:
                    stale_for = now - frame_time
                    limit = max(2.0, self.read_timeout_ms / 1000.0 * 1.5)
                    reason = 'frame_watchdog_timeout'
                if stale_for > limit:
                    self._total_failures += 1
                    self._consecutive_failures += 1
                    self._restart_worker('{}:{:.3f}s'.format(reason, stale_for))

    def _start_worker(self, reason):
        self._generation += 1
        if self._generation > 1:
            self._reconnect_count += 1
        self._generation_started_at = time.monotonic()
        self._captured_at.value = 0.0
        self._active_slot.value = 0
        self._publish_version.value = 0
        self._worker_stop.clear()
        self._frame_ready.clear()
        self._process = self._context.Process(
            target=_capture_worker,
            name='persistent-camera-{}'.format(self._generation),
            args=(
                self.source, self.width, self.height, self.open_timeout_ms,
                self.read_timeout_ms, self.reconnect_backoff_s, self._generation,
                self._worker_stop, self._frame_ready, self._frame_buffer,
                self._active_slot, self._publish_version, self._sequence,
                self._captured_at, self._status_queue,
            ),
            daemon=True,
        )
        self._process.start()
        self._last_status = 'starting'
        self._log('reader_start', reason=reason, pid=self._process.pid)

    def _restart_worker(self, reason):
        self._log('reconnect_start', reason=reason)
        self._stop_worker(reason)
        self._drain_status()
        if not self._monitor_stop.is_set():
            self._start_worker(reason)

    def _stop_worker(self, reason):
        process = self._process
        if process is None:
            return
        self._worker_stop.set()
        process.join(timeout=0.25)
        forced = process.is_alive()
        if forced:
            process.terminate()
            process.join(timeout=0.75)
        if process.is_alive() and hasattr(process, 'kill'):
            process.kill()
            process.join(timeout=0.25)
        self._log(
            'reader_stop', reason=reason, forced=forced,
            exitcode=process.exitcode,
        )
        self._process = None

    def _drain_status(self):
        with self._status_lock:
            while True:
                try:
                    status = self._status_queue.get_nowait()
                except Empty:
                    return
                event = str(status.pop('event', 'unknown'))
                event_time = status.pop('event_time_unix', None)
                event_generation = status.pop('generation', None)
                if (
                    isinstance(event_generation, int)
                    and event_generation != self._generation
                ):
                    self._log(
                        'stale_worker_status',
                        event_time_unix=event_time,
                        stale_event=event,
                        event_generation=event_generation,
                    )
                    continue
                failures = {
                    'open_failed', 'read_failed', 'invalid_frame', 'worker_error'
                }
                if event in failures:
                    self._total_failures += 1
                    self._consecutive_failures += 1
                elif event == 'connected':
                    status['recovered_after_failures'] = self._consecutive_failures
                    self._consecutive_failures = 0
                self._last_status = event
                self._log(
                    event,
                    event_time_unix=event_time,
                    event_generation=event_generation,
                    **status,
                )

    def _log(self, event, event_time_unix=None, **fields):
        if isinstance(event_time_unix, (int, float)):
            timestamp = datetime.fromtimestamp(float(event_time_unix)).astimezone()
        else:
            timestamp = datetime.now().astimezone()
        print(
            '[camera]',
            'ts=' + timestamp.isoformat(timespec='milliseconds'),
            'event=' + str(event),
            'source=' + str(self.source),
            'generation=' + str(self._generation),
            'reconnects=' + str(self._reconnect_count),
            'consecutive_failures=' + str(self._consecutive_failures),
            'total_failures=' + str(self._total_failures),
            'details=' + str(fields),
            flush=True,
        )
