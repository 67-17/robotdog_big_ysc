#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y \
  python3 \
  python3-opencv \
  python3-numpy \
  python3-serial \
  v4l-utils

echo "Dependencies installed."
