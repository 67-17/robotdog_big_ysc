#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/common.sh
source "$(dirname "$0")/common.sh"

python3 arm_task.py \
  --port "${ARM_PORT}" \
  --baud "${ARM_BAUD}" \
  --timeout "${ARM_TIMEOUT}" \
  --json-result \
  home
