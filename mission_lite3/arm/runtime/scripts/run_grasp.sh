#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/common.sh
source "$(dirname "$0")/common.sh"

python3 arm_task.py \
  --port "${ARM_PORT}" \
  --device "${ARM_CAMERA}" \
  --calibration "${ARM_CALIBRATION}" \
  --reference "${ARM_GRASP_REFERENCE}" \
  --place-reference "${ARM_PLACE_REFERENCE}" \
  --config "${ARM_CONFIG}" \
  --width "${ARM_WIDTH}" \
  --height "${ARM_HEIGHT}" \
  --fps "${ARM_FPS}" \
  --baud "${ARM_BAUD}" \
  --timeout "${ARM_TIMEOUT}" \
  --json-result \
  --result-file logs/last_grasp_result.json \
  --run-log-dir "${ARM_RUN_LOG_DIR}" \
  grasp
