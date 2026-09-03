#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

reset_args=(--reset-round)
for point in 1 2 3 4; do
  while true; do
    read -r -p "请将机器狗移动到巡检点 ${point}，停稳后按回车开始识别随机标签（Ctrl-C 退出）: "
    if python3 -u run_live_inspection.py \
      --no-window \
      --json-terminal \
      --no-speak \
      --exit-after-stable \
      --max-seconds 8 \
      --debug-frame-dir inspection_debug \
      "${reset_args[@]}"; then
      echo "[inspection-test] point=${point} complete"
      reset_args=()
      break
    fi
    reset_args=()
    echo "[inspection-test] point=${point} 未得到稳定结果，请调整位置后重试"
  done
done

echo "[inspection-test] points 1-4 complete"
python3 -m json.tool round_result.json
