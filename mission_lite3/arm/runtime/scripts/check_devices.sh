#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/common.sh
source "$(dirname "$0")/common.sh"

echo "Serial device: ${ARM_PORT}"
ls -l "${ARM_PORT}"

echo "Camera device: ${ARM_CAMERA}"
ls -l "${ARM_CAMERA}"

echo "USB serial candidates:"
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true

echo "Video devices:"
ls -l /dev/video* 2>/dev/null || true

if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl --list-devices || true
  v4l2-ctl -d "${ARM_CAMERA}" --list-formats-ext || true
fi

python3 test.py --port "${ARM_PORT}" --baud "${ARM_BAUD}" --timeout "${ARM_TIMEOUT}" status
