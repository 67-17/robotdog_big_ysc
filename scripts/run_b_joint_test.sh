#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${LITE3_B_JOINT_TEST_LOG_DIR:-${ROOT_DIR}/logs/b_joint_test}"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/b关节_$(date +%Y%m%d_%H%M%S).log"
echo "[b关节] 日志文件=${LOG_PATH}" | tee "${LOG_PATH}"

set +e
python3 -u -m mission_lite3.tools.b_joint_turn_test "$@" 2>&1 | tee -a "${LOG_PATH}"
PIPE_RESULTS=("${PIPESTATUS[@]}")
set -e
TEST_STATUS=${PIPE_RESULTS[0]}
LOG_STATUS=${PIPE_RESULTS[1]}
if [[ "${LOG_STATUS}" -ne 0 && "${TEST_STATUS}" -eq 0 ]]; then
  TEST_STATUS=4
fi
echo "[b关节] 退出状态=${TEST_STATUS}，日志文件=${LOG_PATH}" | tee -a "${LOG_PATH}"
exit "${TEST_STATUS}"
