#!/usr/bin/env bash
# Start Python TTS on Kaggle after setup-kaggle.sh
set -euo pipefail
cd "$(dirname "$0")"

TTS_BACKEND="${TTS_BACKEND:-}"
if [ -z "$TTS_BACKEND" ] && [ -f .tts-backend ]; then
  TTS_BACKEND="$(cat .tts-backend)"
fi
TTS_BACKEND="${TTS_BACKEND:-auto}"
VOICE_BACKEND="${VOICE_BACKEND:-ecapa}"

# Per-backend cosine defaults (override with env)
if [ "$VOICE_BACKEND" = "mfcc" ]; then
  VOICE_MATCH_THRESHOLD="${VOICE_MATCH_THRESHOLD:-0.78}"
  VOICE_WEAK_MATCH="${VOICE_WEAK_MATCH:-0.72}"
  VOICE_MATCH_MARGIN="${VOICE_MATCH_MARGIN:-0.04}"
  VOICE_MULTI_MATCH_THRESHOLD="${VOICE_MULTI_MATCH_THRESHOLD:-0.82}"
  VOICE_MULTI_WEAK_MATCH="${VOICE_MULTI_WEAK_MATCH:-0.76}"
  VOICE_MULTI_MATCH_MARGIN="${VOICE_MULTI_MATCH_MARGIN:-0.06}"
elif [ "$VOICE_BACKEND" = "resemblyzer" ]; then
  VOICE_MATCH_THRESHOLD="${VOICE_MATCH_THRESHOLD:-0.74}"
  VOICE_WEAK_MATCH="${VOICE_WEAK_MATCH:-0.70}"
  VOICE_MATCH_MARGIN="${VOICE_MATCH_MARGIN:-0.06}"
  VOICE_MULTI_MATCH_THRESHOLD="${VOICE_MULTI_MATCH_THRESHOLD:-0.74}"
  VOICE_MULTI_WEAK_MATCH="${VOICE_MULTI_WEAK_MATCH:-0.70}"
  VOICE_MULTI_MATCH_MARGIN="${VOICE_MULTI_MATCH_MARGIN:-0.08}"
else
  VOICE_MATCH_THRESHOLD="${VOICE_MATCH_THRESHOLD:-0.62}"
  VOICE_WEAK_MATCH="${VOICE_WEAK_MATCH:-0.55}"
  VOICE_MATCH_MARGIN="${VOICE_MATCH_MARGIN:-0.05}"
  VOICE_MULTI_MATCH_THRESHOLD="${VOICE_MULTI_MATCH_THRESHOLD:-0.65}"
  VOICE_MULTI_WEAK_MATCH="${VOICE_MULTI_WEAK_MATCH:-0.58}"
  VOICE_MULTI_MATCH_MARGIN="${VOICE_MULTI_MATCH_MARGIN:-0.07}"
fi

# Optional: male Edge voice immediately (no Coqui download). Uncomment on Kaggle if /synthesize returns 503:
# export TTS_BACKEND=edge

pick_server_python() {
  local c
  for c in .venv/bin/python python3; do
    if [ -n "$c" ] && [ -x "$c" ] && "$c" -c "import numpy" 2>/dev/null; then
      echo "$c"
      return 0
    fi
  done
  if [ -f .python-bin ]; then
    c="$(tr -d '\n' < .python-bin)"
    if [ -n "$c" ] && [ -x "$c" ] && "$c" -c "import numpy" 2>/dev/null; then
      echo "$c"
      return 0
    fi
  fi
  echo "python3"
}

PY="$(pick_server_python)"
echo "[run-model-server] Using Python: $PY ($("$PY" -c 'import sys; print(sys.executable)' 2>/dev/null || echo unknown))"

if [ "$TTS_BACKEND" = "coqui" ] || [ "$TTS_BACKEND" = "auto" ]; then
  "$PY" -m pip install -q "edge-tts>=6.1.0" "gTTS>=2.5.0" 2>/dev/null || true
fi

# modelServer needs FastAPI/uvicorn (from setup-kaggle or kaggle image)
if ! "$PY" -c "import fastapi" 2>/dev/null; then
  echo "[run-model-server] Installing minimal API deps for $PY"
  "$PY" -m pip install -q "fastapi>=0.100" "uvicorn>=0.23" 2>/dev/null || true
fi

exec env TTS_BACKEND="$TTS_BACKEND" GPU="${GPU:-1}" USE_TALKING_AVATAR="${USE_TALKING_AVATAR:-0}" \
  TTS_MODEL="${TTS_MODEL:-tts_models/en/vctk/vits}" TTS_SPEAKER="${TTS_SPEAKER:-p229}" \
  EDGE_VOICE="${EDGE_VOICE:-en-US-GuyNeural}" \
  VOICE_BACKEND="$VOICE_BACKEND" \
  VOICE_MATCH_THRESHOLD="$VOICE_MATCH_THRESHOLD" \
  VOICE_WEAK_MATCH="$VOICE_WEAK_MATCH" \
  VOICE_MATCH_MARGIN="$VOICE_MATCH_MARGIN" \
  VOICE_MULTI_MATCH_THRESHOLD="$VOICE_MULTI_MATCH_THRESHOLD" \
  VOICE_MULTI_WEAK_MATCH="$VOICE_MULTI_WEAK_MATCH" \
  VOICE_MULTI_MATCH_MARGIN="$VOICE_MULTI_MATCH_MARGIN" \
  VOICE_MIN_ENROLL_SEC="${VOICE_MIN_ENROLL_SEC:-26}" \
  VOICE_MIN_ENROLL_SPEECH_SEC="${VOICE_MIN_ENROLL_SPEECH_SEC:-20}" \
  VOICE_MIN_IDENTIFY_SEC="${VOICE_MIN_IDENTIFY_SEC:-1.2}" \
  VOICE_MIN_IDENTIFY_SPEECH_SEC="${VOICE_MIN_IDENTIFY_SPEECH_SEC:-0.8}" \
  VOICE_PASSIVE_MIN_SEC="${VOICE_PASSIVE_MIN_SEC:-1.0}" \
  "$PY" modelServer.py
