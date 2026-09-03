#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/foxy/setup.bash}"

if [[ ! -r "${ROS_SETUP}" ]]; then
  echo "[full-mission] ROS2 setup is not readable: ${ROS_SETUP}" >&2
  exit 2
fi

cd "${ROOT_DIR}"
set +u
source "${ROS_SETUP}"
set -u

MOTION_LOCK="${LITE3_MOTION_LOCK:-/tmp/lite3_motion_test.lock}"
MOTION_LOCK_WAIT_S="${LITE3_MOTION_LOCK_WAIT_S:-5}"

exec 9>"${MOTION_LOCK}"
if ! flock -w "${MOTION_LOCK_WAIT_S}" 9; then
  echo "[full-mission] motion lock timeout: ${MOTION_LOCK}" >&2
  exit 3
fi

MISSION_LOG_DIR="${LITE3_MISSION_LOG_DIR:-${ROOT_DIR}/logs/full_mission}"
mkdir -p "${MISSION_LOG_DIR}"
MISSION_LOG_PATH="${MISSION_LOG_DIR}/mission_$(date +%Y%m%d_%H%M%S).log"
echo "[full-mission] log=${MISSION_LOG_PATH}" | tee "${MISSION_LOG_PATH}"

set +e
python3 -u -m mission_lite3.run_mission --robot "$@" 2>&1 | tee -a "${MISSION_LOG_PATH}"
PIPE_RESULTS=("${PIPESTATUS[@]}")
set -e
MISSION_STATUS=${PIPE_RESULTS[0]}
LOG_STATUS=${PIPE_RESULTS[1]}

if [[ "${LOG_STATUS}" -ne 0 ]]; then
  echo "[full-mission] task log write failed: ${MISSION_LOG_PATH}" >&2
  if [[ "${MISSION_STATUS}" -eq 0 ]]; then
    MISSION_STATUS=4
  fi
fi

echo "[full-mission] exit_status=${MISSION_STATUS} log=${MISSION_LOG_PATH}" | tee -a "${MISSION_LOG_PATH}"
exit "${MISSION_STATUS}"
