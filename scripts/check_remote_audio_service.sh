#!/usr/bin/env bash
set -euo pipefail

MOTION_HOST="${MOTION_HOST:-192.168.1.120}"
MOTION_USER="${MOTION_USER:-ysc}"
TARGET="${MOTION_USER}@${MOTION_HOST}"

ssh -tt "${TARGET}" 'systemctl is-active lite3-remote-audio.service; sudo systemctl status --no-pager lite3-remote-audio.service'

