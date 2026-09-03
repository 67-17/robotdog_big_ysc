from __future__ import annotations

import json
import math
import os
import shutil
import socket
import subprocess
import time
import uuid
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_AUDIO_DIR = Path(__file__).resolve().parent / "inspection_audio"
RESULT_SUFFIX_BY_LEVEL = {
    "正常": "normal",
    "偏低": "low",
    "偏高": "high",
}


def clip_name_for_result(letter: str, level: str, state: str = "") -> Optional[str]:
    area = str(letter or "").upper()
    if area not in {"A", "B", "C", "D"}:
        return None

    result_level = str(level or "")
    result_state = str(state or "")
    if result_level == "正常" or result_state == "正常":
        return f"{area}_normal.wav"
    suffix = RESULT_SUFFIX_BY_LEVEL.get(result_level)
    if result_state == "异常" and suffix in {"low", "high"}:
        return f"{area}_{suffix}.wav"
    if suffix in {"low", "high"}:
        return f"{area}_{suffix}.wav"
    return None


def resolve_audio_clip(
    letter: str,
    level: str,
    state: str = "",
    audio_dir: Path | str = DEFAULT_AUDIO_DIR,
) -> Optional[Path]:
    clip_name = clip_name_for_result(letter, level, state)
    if clip_name is None:
        return None
    path = Path(audio_dir) / clip_name
    return path if path.is_file() else None


def build_announcement(letter: str, level: str, state: str = "") -> Optional[str]:
    area = str(letter or "").upper()
    if area not in {"A", "B", "C", "D"}:
        return None

    result_level = str(level or "")
    result_state = str(state or "")
    if not result_state:
        result_state = "正常" if result_level == "正常" else "异常"
    if not result_level:
        result_level = "正常" if result_state == "正常" else ""
    if not result_level or result_state not in {"正常", "异常"}:
        return None
    return f"{area}区域仪表盘显示{result_level}，状态{result_state}"


def wav_duration_seconds(path: Path | str) -> Optional[float]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            rate = wav_file.getframerate()
            if rate <= 0:
                return None
            return wav_file.getnframes() / rate
    except (OSError, EOFError, wave.Error):
        return None


def playback_elapsed_is_plausible(expected_duration: Optional[float], elapsed: float) -> bool:
    if expected_duration is None:
        return True
    return elapsed >= expected_duration * 0.65


def _resolve_configured_audio_dir(value: object) -> Path:
    if not value:
        return DEFAULT_AUDIO_DIR
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        Path(__file__).resolve().parent / path,
        Path(__file__).resolve().parents[1] / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


class AudioReporter:
    def __init__(self, config: dict, dry_run: bool = False):
        audio_cfg = config["audio"]
        network_cfg = config.get("network", {})
        self.enabled = bool(audio_cfg.get("enabled", True)) and not dry_run
        self.mode = str(audio_cfg.get("mode", "wav_first"))
        self.audio_dir = _resolve_configured_audio_dir(audio_cfg.get("audio_dir"))
        self.fallback_to_tts_on_audio_failure = bool(audio_cfg.get("fallback_to_tts_on_audio_failure", False))
        self.remote_host = str(audio_cfg.get("remote_host") or network_cfg.get("motion_ip") or "")
        self.remote_port = int(audio_cfg.get("remote_port", 43910))
        self.remote_timeout_seconds = float(audio_cfg.get("remote_timeout_seconds", 8.0))
        self.remote_retries = max(0, int(audio_cfg.get("remote_retries", 1)))
        self.remote_gain_db = float(audio_cfg.get("remote_gain_db", 3.0))
        self.prewarm_enabled = bool(audio_cfg.get("prewarm_enabled", True))
        self.prewarm_duration_s = float(audio_cfg.get("prewarm_duration_s", 0.8))
        if not math.isfinite(self.prewarm_duration_s) or not 0.1 <= self.prewarm_duration_s <= 3.0:
            raise ValueError("audio.prewarm_duration_s must be finite and within [0.1, 3.0]")
        if not math.isfinite(self.remote_gain_db) or not -12.0 <= self.remote_gain_db <= 6.0:
            raise ValueError("audio.remote_gain_db must be finite and within [-12, 6]")
        self.command = audio_cfg.get("command", "spd-say")
        self.args = list(audio_cfg.get("args", []))
        self.timeout_seconds = float(audio_cfg.get("timeout_seconds", 12.0))
        self.pulse_sink = str(audio_cfg.get("pulse_sink") or "")
        self.pulse_server = str(audio_cfg.get("pulse_server") or "")
        self.pulse_volume = str(audio_cfg.get("pulse_volume") or "")
        self.prepare_pulse = bool(audio_cfg.get("prepare_pulse", bool(self.pulse_sink)))
        self.pulse_setup_timeout_seconds = float(audio_cfg.get("pulse_setup_timeout_seconds", 3.0))
        self._pulse_prepared = False
        self.last_error: Optional[str] = None

    def say(self, text: str) -> Optional[str]:
        print(f"[播报] {text}")
        self.last_error = None
        if not self.enabled:
            return None
        env = self._audio_env()
        pulse_error = self._prepare_pulse_output(env)
        return self._say_tts(text, env, pulse_error)

    def say_result(self, letter: str, level: str, state: str) -> Optional[str]:
        text = build_announcement(letter, level, state)
        if text is None:
            self.last_error = f"unsupported inspection result: letter={letter} level={level} state={state}"
            print(f"[audio] speech_error: {self.last_error}")
            return self.last_error
        print(f"[播报] {text}")
        self.last_error = None
        if not self.enabled:
            return None

        if self.mode in {"remote_udp", "udp"}:
            clip_name = clip_name_for_result(letter, level, state)
            if clip_name is None:
                self.last_error = f"unsupported remote audio clip: letter={letter} level={level} state={state}"
                print(f"[audio] remote_audio_error: {self.last_error}")
                return self.last_error
            remote_error = self._play_remote_clip(clip_name)
            if remote_error:
                self.last_error = remote_error
                print(f"[audio] remote_audio_error: {self.last_error}")
                if self.fallback_to_tts_on_audio_failure:
                    env = self._audio_env()
                    pulse_error = self._prepare_pulse_output(env)
                    return self._say_tts(text, env, pulse_error)
                return self.last_error
            print(f"[audio] remote_audio_ok: {clip_name}")
            return None

        env = self._audio_env()
        pulse_error = self._prepare_pulse_output(env)
        if self.mode in {"tts", "tts_only"}:
            return self._say_tts(text, env, pulse_error)

        clip_path = resolve_audio_clip(letter, level, state, self.audio_dir)
        if clip_path is None:
            self.last_error = f"audio clip not found: letter={letter} level={level} state={state} dir={self.audio_dir}"
            print(f"[audio] audio_error: {self.last_error}")
            if self.fallback_to_tts_on_audio_failure:
                return self._say_tts(text, env, pulse_error)
            return self.last_error

        audio_error = self._play_audio_file(clip_path, env)
        if audio_error:
            self.last_error = audio_error
            print(f"[audio] audio_error: {self.last_error}")
            if self.fallback_to_tts_on_audio_failure:
                return self._say_tts(text, env, pulse_error)
            return self.last_error

        print(f"[audio] audio_file_ok: {clip_path}")
        return pulse_error

    def say_record(self, record) -> Optional[str]:
        return self.say_result(record.letter, record.level, record.state)

    def prewarm(self) -> Optional[str]:
        self.last_error = None
        if not self.enabled or not self.prewarm_enabled:
            print("[audio] prewarm_skipped", flush=True)
            return None
        if self.mode not in {"remote_udp", "udp"}:
            print(f"[audio] prewarm_skipped mode={self.mode}", flush=True)
            return None
        print(
            f"[audio] prewarm_start duration_s={self.prewarm_duration_s:.3f}",
            flush=True,
        )
        response, request_error = self._remote_request(
            "warmup",
            duration_s=self.prewarm_duration_s,
        )
        if request_error:
            self.last_error = request_error
        elif response is None or response.get("ok") is not True:
            self.last_error = str(
                (response or {}).get("error")
                or "remote audio server returned warmup failure"
            )
        if self.last_error:
            print(f"[audio] prewarm_error: {self.last_error}", flush=True)
            return self.last_error
        print("[audio] prewarm_ok", flush=True)
        return None

    def _say_tts(self, text: str, env: dict[str, str], pulse_error: Optional[str]) -> Optional[str]:
        self.last_error = None
        if shutil.which(self.command) is None:
            self.last_error = f"{self.command} not found"
            print(f"[audio] speech_error: {self.last_error}")
            return self.last_error
        command = [self.command, *self.args, text]
        try:
            completed = subprocess.run(
                command,
                check=False,
                timeout=self.timeout_seconds,
                capture_output=True,
                text=True,
                env=env,
            )
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[audio] speech_error: {self.last_error}")
            return self.last_error
        if completed.returncode != 0:
            self.last_error = f"{self.command} exited with {completed.returncode}"
            print(f"[audio] speech_error: {self.last_error}")
            if completed.stderr:
                print(f"[audio] stderr: {completed.stderr.strip()}")
        else:
            print(f"[audio] speech_command_ok: {' '.join(command[:-1])}")
        return self.last_error or pulse_error

    def _play_remote_clip(self, clip_name: str) -> Optional[str]:
        if not self.remote_host:
            return "remote audio host is not configured"
        response, request_error = self._remote_request(
            "play",
            clip=clip_name,
            gain_db=self.remote_gain_db,
        )
        if request_error:
            return request_error
        if response is None:
            return "remote audio response is missing"
        if response.get("ok") is not True:
            return str(response.get("error") or "remote audio server returned failure")
        applied_gain = response.get("applied_gain_db")
        if applied_gain is None:
            return "remote audio server did not confirm gain_db; upgrade the remote service"
        if abs(float(applied_gain) - self.remote_gain_db) > 1e-6:
            return (
                "remote audio gain mismatch: "
                f"requested={self.remote_gain_db:g}dB applied={float(applied_gain):g}dB"
            )
        return None

    def _remote_request(self, command: str, **fields) -> tuple[Optional[dict], Optional[str]]:
        if not self.remote_host:
            return None, "remote audio host is not configured"
        request_id = uuid.uuid4().hex
        request = {
            "command": command,
            "request_id": request_id,
            **fields,
        }
        payload = json.dumps(request, ensure_ascii=True).encode("utf-8")
        address = (self.remote_host, self.remote_port)
        attempts = self.remote_retries + 1
        last_error = ""
        for attempt in range(1, attempts + 1):
            request_time = datetime.now().astimezone().isoformat(timespec="milliseconds")
            print(
                f"[audio-service] request ts={request_time} "
                f"command={command} request_id={request_id} "
                f"attempt={attempt}/{attempts}",
                flush=True,
            )
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.settimeout(self.remote_timeout_seconds)
                    sock.sendto(payload, address)
                    response_payload, response_address = sock.recvfrom(4096)
                response = json.loads(response_payload.decode("utf-8"))
                response_time = datetime.now().astimezone().isoformat(timespec="milliseconds")
                print(
                    f"[audio-service] response ts={response_time} "
                    f"command={command} request_id={request_id} "
                    f"from={response_address[0]}:{response_address[1]} "
                    f"payload={json.dumps(response, ensure_ascii=True, sort_keys=True)}",
                    flush=True,
                )
                if not isinstance(response, dict):
                    return None, "remote audio response is not a JSON object"
                if str(response.get("request_id") or "") != request_id:
                    last_error = "remote audio response request_id mismatch"
                    continue
                return response, None
            except Exception as exc:
                last_error = f"attempt {attempt}/{attempts}: {exc}"
                failure_time = datetime.now().astimezone().isoformat(timespec="milliseconds")
                print(
                    f"[audio-service] no_response ts={failure_time} command={command} "
                    f"request_id={request_id} error={last_error}",
                    flush=True,
                )
        return None, last_error or "remote audio request failed"

    def _play_audio_file(self, path: Path, env: dict[str, str]) -> Optional[str]:
        expected_duration = wav_duration_seconds(path)
        timeout = self.timeout_seconds
        if expected_duration is not None:
            timeout = max(timeout, expected_duration + 3.0)

        commands: list[list[str]] = []
        if shutil.which("paplay") is not None:
            paplay = ["paplay"]
            if self.pulse_sink:
                paplay.append(f"--device={self.pulse_sink}")
            paplay.append(str(path))
            commands.append(paplay)
        if shutil.which("aplay") is not None:
            commands.append(["aplay", str(path)])
        if shutil.which("ffplay") is not None:
            commands.append(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)])
        if not commands:
            return "no usable audio player found; need paplay, aplay or ffplay"

        errors: list[str] = []
        for command in commands:
            started_at = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    timeout=timeout,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except Exception as exc:
                errors.append(f"{command[0]} error: {exc}")
                continue
            elapsed = time.monotonic() - started_at
            if completed.returncode == 0 and playback_elapsed_is_plausible(expected_duration, elapsed):
                print(f"[audio] audio_player_ok: {command[0]} elapsed={elapsed:.2f}s")
                return None
            details = f"{command[0]} exited with {completed.returncode} elapsed={elapsed:.2f}s"
            if completed.stderr.strip():
                details = f"{details}: {completed.stderr.strip()}"
            errors.append(details)
        return "; ".join(errors)

    def _audio_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.pulse_server:
            env["PULSE_SERVER"] = self.pulse_server
        return env

    def _prepare_pulse_output(self, env: dict[str, str]) -> Optional[str]:
        if not self.prepare_pulse or self._pulse_prepared:
            return None
        if not self.pulse_sink:
            return None
        if shutil.which("pactl") is None:
            error = "pactl not found"
            print(f"[audio] pulse_setup_error: {error}")
            return error

        commands = [
            ["pactl", "set-default-sink", self.pulse_sink],
            ["pactl", "set-sink-mute", self.pulse_sink, "0"],
        ]
        if self.pulse_volume:
            commands.append(["pactl", "set-sink-volume", self.pulse_sink, self.pulse_volume])

        for command in commands:
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    timeout=self.pulse_setup_timeout_seconds,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            except Exception as exc:
                error = str(exc)
                print(f"[audio] pulse_setup_error: {error}")
                return error
            if completed.returncode != 0:
                error = f"{' '.join(command)} exited with {completed.returncode}"
                print(f"[audio] pulse_setup_error: {error}")
                if completed.stderr:
                    print(f"[audio] stderr: {completed.stderr.strip()}")
                return error

        self._pulse_prepared = True
        volume = f" volume={self.pulse_volume}" if self.pulse_volume else ""
        print(f"[audio] pulse_output_ok: sink={self.pulse_sink}{volume}")
        return None
