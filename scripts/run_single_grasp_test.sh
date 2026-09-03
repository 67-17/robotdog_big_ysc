#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/foxy/setup.bash}"

cd "${ROOT_DIR}"
if [[ " ${*} " == *" --robot "* ]]; then
  if [[ ! -r "${ROS_SETUP}" ]]; then
    echo "[single-grasp] ROS2 setup is not readable: ${ROS_SETUP}" >&2
    exit 2
  fi
  set +u
  source "${ROS_SETUP}"
  set -u

  MOTION_LOCK="${LITE3_MOTION_LOCK:-/tmp/lite3_motion_test.lock}"
  MOTION_LOCK_WAIT_S="${LITE3_MOTION_LOCK_WAIT_S:-5}"
  exec 9>"${MOTION_LOCK}"
  if ! flock -w "${MOTION_LOCK_WAIT_S}" 9; then
    echo "[single-grasp] motion lock timeout: ${MOTION_LOCK}" >&2
    exit 3
  fi
fi

LOG_DIR="${LITE3_SINGLE_GRASP_LOG_DIR:-${ROOT_DIR}/logs/single_grasp}"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/single_grasp_$(date +%Y%m%d_%H%M%S).log"
echo "[single-grasp] log=${LOG_PATH}" | tee "${LOG_PATH}"

set +e
python3 -u -m mission_lite3.tools.single_grasp_test "$@" 2>&1 | tee -a "${LOG_PATH}"
PIPE_RESULTS=("${PIPESTATUS[@]}")
set -e
TASK_STATUS=${PIPE_RESULTS[0]}
LOG_STATUS=${PIPE_RESULTS[1]}

if [[ "${LOG_STATUS}" -ne 0 ]]; then
  echo "[single-grasp] task log write failed: ${LOG_PATH}" >&2
  if [[ "${TASK_STATUS}" -eq 0 ]]; then
    TASK_STATUS=4
  fi
fi

echo "[single-grasp] exit_status=${TASK_STATUS} log=${LOG_PATH}" | tee -a "${LOG_PATH}"
exit "${TASK_STATUS}"
