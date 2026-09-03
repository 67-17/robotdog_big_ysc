#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from mission_lite3 import lite3_speaker
from mission_lite3.audio import AudioReporter, build_announcement, resolve_audio_clip
from mission_lite3.config_loader import load_config


DEFAULT_TEXT = "A区域仪表盘显示偏低，状态异常"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe PulseAudio output and mission speech without moving the robot")
    parser.add_argument("--area", choices=("A", "B", "C", "D"), default="A", help="Area used to select a WAV clip")
    parser.add_argument(
        "--description",
        choices=("正常", "偏低", "偏高"),
        default="偏低",
        help="Meter description used to select a WAV clip",
    )
    parser.add_argument("--text", default=None, help="Text passed to the TTS path when --tts-only or --include-tts is used")
    parser.add_argument("--sink", default=None, help="PulseAudio sink; defaults to audio.pulse_sink in config")
    parser.add_argument("--pulse-server", default=None, help="PulseAudio server; defaults to audio.pulse_server in config")
    parser.add_argument("--volume", default=None, help="PulseAudio sink volume; defaults to audio.pulse_volume in config")
    parser.add_argument("--local-audio", action="store_true", help="Test local perception-host audio instead of remote UDP audio")
    parser.add_argument("--remote-host", default=None, help="Remote UDP audio server host; defaults to audio.remote_host")
    parser.add_argument("--remote-port", type=int, default=None, help="Remote UDP audio server port; defaults to audio.remote_port")
    parser.add_argument(
        "--remote-gain-db",
        type=float,
        default=None,
        help="Remote body-speaker digital gain in dB; defaults to audio.remote_gain_db",
    )
    parser.add_argument("--motion-host", default=None, help="Lite3 motion-host IP for the speaker enable command")
    parser.add_argument("--motion-port", type=int, default=None, help="Lite3 motion-host UDP port for the speaker enable command")
    parser.add_argument("--skip-lite3-speaker", action="store_true", help="Do not send the Lite3 speaker enable command before playback")
    parser.add_argument("--skip-beep", action="store_true", help="Skip direct local paplay beep test")
    parser.add_argument("--skip-speech", action="store_true", help="Skip mission speech test")
    parser.add_argument("--tts-only", action="store_true", help="Use spd-say TTS instead of the selected WAV clip")
    parser.add_argument("--include-tts", action="store_true", help="Also test spd-say TTS after the selected WAV clip")
    parser.add_argument("--skip-pulse-setup", action="store_true", help="Do not set default sink/mute/volume before testing")
    parser.add_argument("--diagnose", action="store_true", help="Print PulseAudio and ALSA device diagnostics")
    parser.add_argument("--alsa-device", default=None, help="Play the beep through this ALSA device, for example default, pulse, plughw:1,0")
    parser.add_argument("--test-alsa-devices", action="store_true", help="Interactively try common ALSA playback devices")
    parser.add_argument("--non-interactive", action="store_true", help="Do not ask whether sound was heard")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    audio_cfg = config["audio"]
    sink = args.sink or str(audio_cfg.get("pulse_sink") or "")
    pulse_server = args.pulse_server or str(audio_cfg.get("pulse_server") or "")
    volume = args.volume or str(audio_cfg.get("pulse_volume") or "")
    interactive = sys.stdin.isatty() and not args.non_interactive

    env = os.environ.copy()
    if pulse_server:
        env["PULSE_SERVER"] = pulse_server
    if sink:
        audio_cfg["pulse_sink"] = sink
    if pulse_server:
        audio_cfg["pulse_server"] = pulse_server
    if volume:
        audio_cfg["pulse_volume"] = volume
    audio_cfg["prepare_pulse"] = not args.skip_pulse_setup
    if args.remote_host:
        audio_cfg["remote_host"] = args.remote_host
    if args.remote_port is not None:
        audio_cfg["remote_port"] = args.remote_port
    if args.remote_gain_db is not None:
        if not math.isfinite(args.remote_gain_db) or not -12.0 <= args.remote_gain_db <= 6.0:
            raise SystemExit("--remote-gain-db must be between -12 and +6 dB")
        audio_cfg["remote_gain_db"] = args.remote_gain_db
    if args.local_audio:
        audio_cfg["mode"] = "wav_first"
    if args.tts_only:
        audio_cfg["mode"] = "tts_only"

    print(f"[audio-probe] mode={audio_cfg.get('mode')}")
    if args.local_audio or args.tts_only or args.include_tts:
        print(f"[audio-probe] sink={sink or '<default>'}")
        print(f"[audio-probe] pulse_server={pulse_server or '<env/default>'}")
        print(f"[audio-probe] volume={volume or '<unchanged>'}")
    print(
        "[audio-probe] remote="
        f"{audio_cfg.get('remote_host') or config.get('network', {}).get('motion_ip')}:{audio_cfg.get('remote_port')} "
        f"gain_db={float(audio_cfg.get('remote_gain_db', 3.0)):g}"
    )

    enable_lite3_speaker(args, config)

    if args.diagnose:
        diagnose_audio(env)

    if args.test_alsa_devices:
        if test_alsa_devices(env, interactive):
            return 0
        return 7

    needs_playback = args.local_audio and (not args.skip_beep or not args.skip_speech)
    if needs_playback and not args.skip_pulse_setup and sink:
        if not setup_pulse(sink, volume, env):
            return 2

    if args.local_audio and not args.skip_beep:
        if not play_beep(sink, env, alsa_device=args.alsa_device):
            return 3
        if interactive and not confirm("是否听到蜂鸣测试音"):
            print("[audio-probe] 蜂鸣无声：先不要跑全流程，检查扬声器接线、声卡输出口、PulseAudio sink 和音量。")
            return 4
    elif not args.local_audio and not args.skip_beep:
        print("[audio-probe] skip local beep in remote UDP mode; use --local-audio to test perception-host audio")

    if not args.skip_speech:
        reporter = AudioReporter(config)
        announcement = build_announcement(args.area, args.description, state_for_description(args.description)) or DEFAULT_TEXT
        if args.tts_only:
            print("[audio-probe] speech_mode=tts")
            speech_error = reporter.say(args.text or announcement)
        else:
            clip_path = resolve_audio_clip(args.area, args.description, state_for_description(args.description), reporter.audio_dir)
            if args.local_audio:
                print(f"[audio-probe] speech_mode=local_wav clip={clip_path or '<not found>'}")
            else:
                print(
                    "[audio-probe] speech_mode=remote_udp "
                    f"clip={args.area}_{'normal' if args.description == '正常' else 'low' if args.description == '偏低' else 'high'}.wav"
                )
            speech_error = reporter.say_result(args.area, args.description, state_for_description(args.description))
        if speech_error:
            print(f"[audio-probe] speech_error={speech_error}")
            return 5
        if args.include_tts and not args.tts_only:
            print("[audio-probe] speech_mode=tts")
            speech_error = reporter.say(args.text or announcement)
            if speech_error:
                print(f"[audio-probe] tts_error={speech_error}")
                return 5
        if interactive and not confirm("是否听到中文播报内容"):
            print("[audio-probe] 蜂鸣有声但中文播报无声：检查 WAV 播放器、Speech Dispatcher 或当前音频路由。")
            print("[audio-probe] 可分别测试：python3 run_speech_probe.py --skip-beep；python3 run_speech_probe.py --tts-only")
            return 6

    print("[audio-probe] audio probe passed")
    return 0


def state_for_description(description: str) -> str:
    return "正常" if description == "正常" else "异常"


def enable_lite3_speaker(args: argparse.Namespace, config: dict) -> None:
    if args.skip_lite3_speaker:
        print("[audio-probe] lite3_speaker=skipped")
        return
    if not lite3_speaker.speaker_control_enabled():
        print("[audio-probe] lite3_speaker=disabled_by_env")
        return
    network = config.get("network", {})
    host = args.motion_host or str(network.get("motion_ip") or lite3_speaker.infer_motion_host())
    raw_port = args.motion_port if args.motion_port is not None else network.get("motion_port")
    port = lite3_speaker.infer_motion_port() if raw_port is None else int(raw_port)
    try:
        sent = lite3_speaker.open_speaker(host=host, port=port)
    except (OSError, ValueError) as exc:
        print(f"[audio-probe] lite3_speaker_error={exc}")
        return
    print(f"[audio-probe] lite3_speaker_open=true target={host}:{port} bytes={sent}")
    print("[audio-probe] lite3_speaker_note=only opens the Lite3 speaker switch; playback still uses the configured audio path")


def setup_pulse(sink: str, volume: str, env: dict[str, str]) -> bool:
    if shutil.which("pactl") is None:
        print("[audio-probe] pactl not found")
        return False
    commands = [
        ["pactl", "set-default-sink", sink],
        ["pactl", "set-sink-mute", sink, "0"],
    ]
    if volume:
        commands.append(["pactl", "set-sink-volume", sink, volume])
    for command in commands:
        if not run(command, env=env, timeout=3.0):
            return False
    print("[audio-probe] pulse setup ok")
    return True


def play_beep(sink: str, env: dict[str, str], alsa_device: str | None = None) -> bool:
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "probe_beep.wav"
        write_beep_wav(wav_path)
        if alsa_device:
            if shutil.which("aplay") is None:
                print("[audio-probe] aplay not found")
                return False
            return run(["aplay", "-D", alsa_device, str(wav_path)], env=env, timeout=5.0)
        if shutil.which("paplay") is not None:
            command = ["paplay"]
            if sink:
                command.append(f"--device={sink}")
            command.append(str(wav_path))
            if run(command, env=env, timeout=5.0):
                return True
        if shutil.which("aplay") is not None:
            return run(["aplay", "-D", "pulse", str(wav_path)], env=env, timeout=5.0)
    print("[audio-probe] no usable player found; need paplay or aplay")
    return False


def diagnose_audio(env: dict[str, str]) -> None:
    print("[audio-probe] ==== PulseAudio / ALSA diagnose ====")
    diagnostic_commands = [
        ["pactl", "info"],
        ["pactl", "list", "short", "sinks"],
        ["pactl", "list", "sinks"],
        ["pactl", "list", "cards"],
        ["aplay", "-l"],
        ["aplay", "-L"],
        ["amixer", "-c", "0", "scontents"],
        ["amixer", "-c", "1", "scontents"],
    ]
    for command in diagnostic_commands:
        if shutil.which(command[0]) is None:
            print(f"[audio-probe] skip: {command[0]} not found")
            continue
        run(command, env=env, timeout=5.0)
    print("[audio-probe] ==== diagnose end ====")


def test_alsa_devices(env: dict[str, str], interactive: bool) -> bool:
    if shutil.which("aplay") is None:
        print("[audio-probe] aplay not found")
        return False
    candidates = [
        "default",
        "pulse",
        "plughw:0,0",
        "plughw:1,0",
        "hw:0,0",
        "hw:1,0",
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "probe_beep.wav"
        write_beep_wav(wav_path, seconds=1.0)
        for device in candidates:
            ok = run(["aplay", "-D", device, str(wav_path)], env=env, timeout=5.0)
            if not ok:
                continue
            if not interactive:
                print(f"[audio-probe] command accepted ALSA device: {device}")
                continue
            if confirm(f"ALSA 设备 {device} 是否听到蜂鸣"):
                print(f"[audio-probe] working_alsa_device={device}")
                print(f"[audio-probe] 可复测：python3 run_speech_probe.py --alsa-device {device} --skip-speech")
                return True
    print("[audio-probe] no audible ALSA device confirmed")
    return False


def write_beep_wav(path: Path, seconds: float = 1.2, frequency_hz: float = 880.0) -> None:
    sample_rate = 44100
    amplitude = 12000
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for index in range(frames):
            sample = int(amplitude * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate))
            wav_file.writeframes(struct.pack("<h", sample))


def run(command: list[str], env: dict[str, str], timeout: float) -> bool:
    print(f"[audio-probe] run: {' '.join(command)}")
    try:
        completed = subprocess.run(command, check=False, timeout=timeout, capture_output=True, text=True, env=env)
    except Exception as exc:
        print(f"[audio-probe] error: {exc}")
        return False
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())
    if completed.returncode != 0:
        print(f"[audio-probe] command exited with {completed.returncode}")
        return False
    return True


def confirm(question: str) -> bool:
    answer = input(f"[audio-probe] {question}? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
