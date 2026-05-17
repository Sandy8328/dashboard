#!/usr/bin/env bash
# Start Python TTS on Kaggle after setup-kaggle.sh
set -euo pipefail
cd "$(dirname "$0")"

TTS_BACKEND="${TTS_BACKEND:-}"
if [ -z "$TTS_BACKEND" ] && [ -f .tts-backend ]; then
  TTS_BACKEND="$(cat .tts-backend)"
fi
TTS_BACKEND="${TTS_BACKEND:-auto}"

if [ -f .use-venv ] && [ "$(cat .use-venv)" = "1" ] && [ -x .venv/bin/python ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  exec env TTS_BACKEND="$TTS_BACKEND" GPU="${GPU:-1}" USE_TALKING_AVATAR="${USE_TALKING_AVATAR:-0}" \
    python modelServer.py
fi

PY="python3"
if [ -f .python-bin ]; then
  PY="$(cat .python-bin)"
fi
exec env TTS_BACKEND="$TTS_BACKEND" GPU="${GPU:-1}" USE_TALKING_AVATAR="${USE_TALKING_AVATAR:-0}" \
  "$PY" modelServer.py
