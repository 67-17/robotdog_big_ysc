#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/common.sh
source "$(dirname "$0")/common.sh"

python3 strip_detector.py \
  --device "${ARM_CAMERA}" \
  --config "${ARM_CONFIG}" \
  --calibration "${ARM_CALIBRATION}" \
  --width "${ARM_WIDTH}" \
  --height "${ARM_HEIGHT}" \
  --fps "${ARM_FPS}" \
  --headless
