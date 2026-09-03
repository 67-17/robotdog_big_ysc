#!/usr/bin/env bash
set -euo pipefail

MOTION_HOST="${MOTION_HOST:-192.168.1.120}"
MOTION_USER="${MOTION_USER:-ysc}"
REMOTE_APP_DIR="${REMOTE_APP_DIR:-/opt/robot_competition}"
SERVICE_NAME="${SERVICE_NAME:-lite3-remote-audio.service}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_SCRIPT="${ROOT_DIR}/tools/remote_audio_server.py"
SERVICE_FILE="${ROOT_DIR}/systemd/${SERVICE_NAME}"
TARGET="${MOTION_USER}@${MOTION_HOST}"

if [[ ! -f "${SERVER_SCRIPT}" ]]; then
  echo "missing server script: ${SERVER_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${SERVICE_FILE}" ]]; then
  echo "missing service file: ${SERVICE_FILE}" >&2
  exit 1
fi

scp "${SERVER_SCRIPT}" "${TARGET}:/tmp/remote_audio_server.py"
scp "${SERVICE_FILE}" "${TARGET}:/tmp/${SERVICE_NAME}"

ssh -tt "${TARGET}" "sudo sh -s" <<'REMOTE_SCRIPT'
set -eu
install -d -m 0755 /opt/robot_competition
install -m 0755 /tmp/remote_audio_server.py /opt/robot_competition/remote_audio_server.py
install -d -m 0755 /opt/robot_competition/inspection_audio_test
if [ -d /tmp/inspection_audio_test ]; then
  cp -a /tmp/inspection_audio_test/. /opt/robot_competition/inspection_audio_test/
fi
install -m 0644 /tmp/lite3-remote-audio.service /etc/systemd/system/lite3-remote-audio.service
systemctl daemon-reload
systemctl enable --now lite3-remote-audio.service
systemctl status --no-pager lite3-remote-audio.service
REMOTE_SCRIPT

echo
echo "Remote audio service installed."
echo "Check logs with:"
echo "  ssh ${TARGET} 'sudo journalctl -u lite3-remote-audio.service -f'"

