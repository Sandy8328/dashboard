#!/usr/bin/env bash
# Optional speaker enrollment (Resemblyzer) on Kaggle — run after setup-kaggle.sh
set -euo pipefail
cd "$(dirname "$0")"

PIP() {
  if [ -x .venv/bin/python ]; then
    .venv/bin/python -m pip "$@"
  else
    python3 -m pip "$@"
  fi
}

RUN() {
  if [ -x .venv/bin/python ]; then
    .venv/bin/python "$@"
  else
    python3 "$@"
  fi
}

echo "==> setuptools <82 (pkg_resources for Resemblyzer; removed in 82+)"
PIP install -U "setuptools>=65.0.0,<82" wheel

echo "==> Resemblyzer + deps"
PIP install -r requirements-voice-id.txt

echo "==> Verify ffmpeg"
ffmpeg -version | head -n 1

echo "==> Verify VoiceEncoder"
RUN -c "
from voice_id import status
import json, sys
st = status()
print(json.dumps(st, indent=2))
if not st.get('ready'):
    sys.exit(1)
"

echo ""
echo "Voice ID ready. modelServer.py exposes POST /voice/enroll and /voice/identify"
