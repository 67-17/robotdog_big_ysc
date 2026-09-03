#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/common.sh
source "$(dirname "$0")/common.sh"

slot="${1:-}"
if [[ -z "${slot}" ]]; then
  echo "Usage: bash scripts/run_place.sh A|B|C|D" >&2
  exit 2
fi

python3 arm_task.py \
  --port "${ARM_PORT}" \
  --place-reference "${ARM_PLACE_REFERENCE}" \
  --baud "${ARM_BAUD}" \
  --timeout "${ARM_TIMEOUT}" \
  --json-result \
  --result-file "logs/last_place_${slot}.json" \
  place --slot "${slot}" --object-held
