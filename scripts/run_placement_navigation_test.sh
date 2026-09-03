#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/foxy/setup.bash}"

cd "${ROOT_DIR}"
if [[ " ${*} " == *" --robot "* ]]; then
  if [[ ! -r "${ROS_SETUP}" ]]; then
    echo "[placement-nav-test] ROS2 setup is not readable: ${ROS_SETUP}" >&2
    exit 2
  fi
  set +u
  source "${ROS_SETUP}"
  set -u

  MOTION_LOCK="${LITE3_MOTION_LOCK:-/tmp/lite3_motion_test.lock}"
  MOTION_LOCK_WAIT_S="${LITE3_MOTION_LOCK_WAIT_S:-5}"
  exec 9>"${MOTION_LOCK}"
  if ! flock -w "${MOTION_LOCK_WAIT_S}" 9; then
    echo "[placement-nav-test] motion lock timeout: ${MOTION_LOCK}" >&2
    exit 3
  fi
fi

LOG_DIR="${LITE3_PLACEMENT_NAV_TEST_LOG_DIR:-${ROOT_DIR}/logs/placement_navigation_test}"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/placement_nav_$(date +%Y%m%d_%H%M%S).log"
echo "[placement-nav-test] log=${LOG_PATH}" | tee "${LOG_PATH}"

set +e
python3 -u -m mission_lite3.tools.placement_navigation_test "$@" 2>&1 | tee -a "${LOG_PATH}"
PIPE_RESULTS=("${PIPESTATUS[@]}")
set -e
TEST_STATUS=${PIPE_RESULTS[0]}
LOG_STATUS=${PIPE_RESULTS[1]}

if [[ "${LOG_STATUS}" -ne 0 && "${TEST_STATUS}" -eq 0 ]]; then
  TEST_STATUS=4
fi
echo "[placement-nav-test] exit_status=${TEST_STATUS} log=${LOG_PATH}" | tee -a "${LOG_PATH}"
exit "${TEST_STATUS}"
