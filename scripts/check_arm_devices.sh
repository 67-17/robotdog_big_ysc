#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[arm-check] serial devices"
find /dev -maxdepth 1 \( -name 'ttyUSB*' -o -name 'ttyACM*' \) -print 2>/dev/null | sort || true

echo "[arm-check] stable serial aliases"
find /dev/serial/by-id -maxdepth 1 -type l -print 2>/dev/null | sort || true

echo "[arm-check] video devices"
find /dev -maxdepth 1 -name 'video*' -print 2>/dev/null | sort || true

echo "[arm-check] runtime preflight"
python3 -m mission_lite3.arm.run_arm_task "$@" preflight
