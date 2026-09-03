#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${RUNTIME_DIR}"

if [[ -f "${RUNTIME_DIR}/lite3_arm.env" ]]; then
  # shellcheck disable=SC1091
  source "${RUNTIME_DIR}/lite3_arm.env"
fi

export ARM_PORT="${ARM_PORT:-/dev/ttyUSB0}"
export ARM_CAMERA="${ARM_CAMERA:-/dev/video0}"
export ARM_WIDTH="${ARM_WIDTH:-1280}"
export ARM_HEIGHT="${ARM_HEIGHT:-720}"
export ARM_FPS="${ARM_FPS:-25}"
export ARM_BAUD="${ARM_BAUD:-115200}"
export ARM_TIMEOUT="${ARM_TIMEOUT:-2}"
export ARM_CONFIG="${ARM_CONFIG:-strip_detector_grasp_config.json}"
export ARM_CALIBRATION="${ARM_CALIBRATION:-camera_calibration.json}"
export ARM_GRASP_REFERENCE="${ARM_GRASP_REFERENCE:-grasp_reference_square_face.json}"
export ARM_PLACE_REFERENCE="${ARM_PLACE_REFERENCE:-place_reference.json}"
export ARM_RUN_LOG_DIR="${ARM_RUN_LOG_DIR:-grasp_runs}"

mkdir -p logs "${ARM_RUN_LOG_DIR}" strip_debug
