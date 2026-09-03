#!/usr/bin/env python3
from __future__ import annotations

import argparse
import array
import json
import math
import socket
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any


VALID_CLIPS = {
    f"{letter}_{suffix}.wav"
    for letter in ("A", "B", "C", "D")
    for suffix in ("normal", "low", "high")
}
PROTOCOL_VERSION = 3
MIN_GAIN_DB = -12.0
MAX_GAIN_DB = 6.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UDP audio playback server for Lite3 body speaker")
    parser.add_argument("--host", default="0.0.0.0", help="UDP listen host")
    parser.add_argument("--port", type=int, default=43910, help="UDP listen port")
    parser.add_argument("--audio-dir", type=Path, default=Path("/tmp/inspection_audio_test"), help="Directory with WAV clips")
    parser.add_argument("--device", default="plughw:CARD=rockchipes8388c,DEV=0", help="ALSA playback device")
    parser.add_argument("--timeout", type=float, default=8.0, help="Maximum seconds for one clip playback")
    parser.add_argument(
        "--default-gain-db",
        type=float,
        default=0.0,
        help="Digital gain used when a request omits gain_db",
    )
    return parser


def validate_clip_name(clip_name: object) -> str:
    clip = str(clip_name or "")
    if clip not in VALID_CLIPS:
        raise ValueError(f"clip not allowed: {clip}")
    return clip


def validate_gain_db(value: object) -> float:
    try:
        gain_db = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("gain_db must be a number") from exc
    if not math.isfinite(gain_db):
        raise ValueError("gain_db must be finite")
    if not MIN_GAIN_DB <= gain_db <= MAX_GAIN_DB:
        raise ValueError(f"gain_db must be within [{MIN_GAIN_DB:g}, {MAX_GAIN_DB:g}]")
    return gain_db


def validate_warmup_duration(value: object) -> float:
    try:
        duration_s = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("warmup duration_s must be a number") from exc
    if not math.isfinite(duration_s) or not 0.1 <= duration_s <= 3.0:
        raise ValueError("warmup duration_s must be within [0.1, 3.0]")
    return duration_s


def play_warmup(device: str, timeout: float, duration_s: float = 0.8) -> None:
    duration_s = validate_warmup_duration(duration_s)
    sample_rate = 16000
    frame_count = int(round(sample_rate * duration_s))
    with tempfile.TemporaryDirectory(prefix="lite3-audio-warmup-") as directory:
        path = Path(directory) / "warmup_silence.wav"
        with wave.open(str(path), "wb") as output_wav:
            output_wav.setnchannels(1)
            output_wav.setsampwidth(2)
            output_wav.setframerate(sample_rate)
            output_wav.writeframes(bytes(frame_count * 2))
        subprocess.run(
            ["aplay", "-q", "-D", device, str(path)],
            check=True,
            timeout=max(float(timeout), duration_s + 1.0),
        )


def apply_gain_to_wav(source: Path, destination: Path, gain_db: float) -> int:
    gain_db = validate_gain_db(gain_db)
    with wave.open(str(source), "rb") as input_wav:
        params = input_wav.getparams()
        if input_wav.getcomptype() != "NONE" or input_wav.getsampwidth() != 2:
            raise ValueError("remote gain requires uncompressed 16-bit PCM WAV")
        samples = array.array("h", input_wav.readframes(input_wav.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    multiplier = 10.0 ** (gain_db / 20.0)
    clipped_samples = 0
    for index, sample in enumerate(samples):
        scaled = int(round(sample * multiplier))
        if scaled > 32767:
            scaled = 32767
            clipped_samples += 1
        elif scaled < -32768:
            scaled = -32768
            clipped_samples += 1
        samples[index] = scaled
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(destination), "wb") as output_wav:
        output_wav.setparams(params)
        output_wav.writeframes(samples.tobytes())
    return clipped_samples


def play_clip(audio_dir: Path, device: str, clip_name: str, timeout: float, gain_db: float = 0.0) -> int:
    clip_path = audio_dir / clip_name
    if not clip_path.is_file():
        raise FileNotFoundError(str(clip_path))
    gain_db = validate_gain_db(gain_db)
    clipped_samples = 0
    if abs(gain_db) <= 1e-12:
        playback_path = clip_path
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="lite3-audio-")
        playback_path = Path(temp_dir.name) / clip_name
        clipped_samples = apply_gain_to_wav(clip_path, playback_path, gain_db)
    try:
        subprocess.run(
            ["aplay", "-q", "-D", device, str(playback_path)],
            check=True,
            timeout=timeout,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
    return clipped_samples


def handle_request(
    payload: bytes,
    audio_dir: Path,
    device: str,
    timeout: float,
    default_gain_db: float = 0.0,
) -> dict[str, Any]:
    request_id = ""
    clip_name = ""
    command = ""
    gain_db = validate_gain_db(default_gain_db)
    try:
        request = json.loads(payload.decode("utf-8"))
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        request_id = str(request.get("request_id") or "")
        command = str(request.get("command") or "")
        if command == "warmup":
            duration_s = validate_warmup_duration(request.get("duration_s", 0.8))
            play_warmup(device, timeout, duration_s)
            return {
                "ok": True,
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "command": command,
                "warmup_duration_s": duration_s,
            }
        if command != "play":
            raise ValueError(f"unsupported command: {command}")
        clip_name = validate_clip_name(request.get("clip"))
        gain_db = validate_gain_db(request.get("gain_db", default_gain_db))
        clipped_samples = play_clip(audio_dir, device, clip_name, timeout, gain_db)
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "command": command,
            "clip": clip_name,
            "applied_gain_db": gain_db,
            "clipped_samples": clipped_samples,
        }
    except Exception as exc:
        return {
            "ok": False,
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "command": command,
            "clip": clip_name,
            "applied_gain_db": gain_db,
            "error": str(exc),
        }


def serve(host: str, port: int, audio_dir: Path, device: str, timeout: float, default_gain_db: float = 0.0) -> int:
    audio_dir = audio_dir.expanduser().resolve()
    default_gain_db = validate_gain_db(default_gain_db)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"[remote-audio] listening udp://{host}:{port}")
    print(f"[remote-audio] audio_dir={audio_dir}")
    print(f"[remote-audio] device={device}")
    print(f"[remote-audio] default_gain_db={default_gain_db:g}")
    while True:
        payload, address = sock.recvfrom(4096)
        response = handle_request(payload, audio_dir, device, timeout, default_gain_db)
        command = response.get("command") or "<unknown>"
        clip = response.get("clip") or "<none>"
        if response.get("ok"):
            print(
                f"[remote-audio] ok address={address[0]}:{address[1]} "
                f"command={command} clip={clip} "
                f"gain_db={response.get('applied_gain_db')}",
                flush=True,
            )
        else:
            print(
                f"[remote-audio] error address={address[0]}:{address[1]} "
                f"command={command} clip={clip} "
                f"error={response.get('error')}",
                flush=True,
            )
        sock.sendto(json.dumps(response, ensure_ascii=True).encode("utf-8"), address)


def main() -> int:
    args = build_parser().parse_args()
    return serve(args.host, args.port, args.audio_dir, args.device, args.timeout, args.default_gain_db)


if __name__ == "__main__":
    raise SystemExit(main())
