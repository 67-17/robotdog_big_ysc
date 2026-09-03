"""Control the Lite3 motion-host speaker over UDP.

The Lite3 motion-host communication manual defines simple commands as three
little-endian uint32 values: command code, command value, and command type.
"""

import os
import socket
import struct
from urllib.parse import urlparse


DEFAULT_MOTION_HOST = "192.168.1.120"
DEFAULT_MOTION_PORT = 43893
SIMPLE_COMMAND_TYPE = 0

SPEAKER_COMMAND_CODE = 0x2101030D
SPEAKER_OFF = 0
SPEAKER_ON = 1
SPEAKER_QUERY = 2

MOTION_HOST_ENV_VARS = ("INSPECTION_MOTION_HOST", "LITE3_MOTION_HOST")
MOTION_PORT_ENV_VARS = ("INSPECTION_MOTION_PORT", "LITE3_MOTION_PORT")
SPEAKER_ENABLE_ENV_VARS = ("INSPECTION_LITE3_SPEAKER", "LITE3_SPEAKER")


def _first_env(names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def speaker_control_enabled():
    value = _first_env(SPEAKER_ENABLE_ENV_VARS)
    if value is None:
        return True
    return value.strip().lower() not in ("0", "false", "no", "off")


def infer_motion_host():
    explicit_host = _first_env(MOTION_HOST_ENV_VARS)
    if explicit_host:
        return explicit_host

    camera = os.environ.get("INSPECTION_CAMERA", "")
    if "://" in camera:
        parsed = urlparse(camera)
        if parsed.hostname:
            return parsed.hostname

    return DEFAULT_MOTION_HOST


def infer_motion_port():
    explicit_port = _first_env(MOTION_PORT_ENV_VARS)
    if explicit_port:
        try:
            return int(explicit_port)
        except ValueError as exc:
            raise ValueError(f"运动主机端口无效：{explicit_port!r}") from exc
    return DEFAULT_MOTION_PORT


def build_simple_command(code, value, command_type=SIMPLE_COMMAND_TYPE):
    return struct.pack(
        "<III",
        int(code) & 0xFFFFFFFF,
        int(value) & 0xFFFFFFFF,
        int(command_type) & 0xFFFFFFFF,
    )


def send_simple_command(code, value, host=None, port=None, timeout=0.2):
    target_host = host or infer_motion_host()
    target_port = infer_motion_port() if port is None else int(port)
    packet = build_simple_command(code, value)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        return sock.sendto(packet, (target_host, target_port))


def set_speaker(value, host=None, port=None, timeout=0.2):
    return send_simple_command(
        SPEAKER_COMMAND_CODE,
        value,
        host=host,
        port=port,
        timeout=timeout,
    )


def open_speaker(host=None, port=None, timeout=0.2):
    return set_speaker(SPEAKER_ON, host=host, port=port, timeout=timeout)


def close_speaker(host=None, port=None, timeout=0.2):
    return set_speaker(SPEAKER_OFF, host=host, port=port, timeout=timeout)


def query_speaker(host=None, port=None, timeout=0.2):
    return set_speaker(SPEAKER_QUERY, host=host, port=port, timeout=timeout)
