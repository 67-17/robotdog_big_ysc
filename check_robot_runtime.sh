#!/usr/bin/env bash
set -euo pipefail

python3 -m compileall mission_lite3 tests run_live_inspection.py run_speech_probe.py tools/remote_audio_server.py
python3 -m unittest discover -s tests -v
bash -n check_robot_runtime.sh scripts/*.sh
env -u DISPLAY python3 run_live_inspection.py --help >/dev/null
