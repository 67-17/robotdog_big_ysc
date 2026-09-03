#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/foxy/setup.bash}"

cd "${ROOT_DIR}"
if [[ " ${*} " == *" --robot "* ]]; then
  set +u
  source "${ROS_SETUP}"
  set -u

  MOTION_LOCK="${LITE3_MOTION_LOCK:-/tmp/lite3_motion_test.lock}"
  MOTION_LOCK_WAIT_S="${LITE3_MOTION_LOCK_WAIT_S:-5}"
  exec 9>"${MOTION_LOCK}"
  if ! flock -w "${MOTION_LOCK_WAIT_S}" 9; then
    echo "[pickup-transfer] motion lock timeout: ${MOTION_LOCK}" >&2
    exit 3
  fi
fi

exec python3 -u -m mission_lite3.tools.pickup_transfer_mission "$@"
