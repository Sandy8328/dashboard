#!/usr/bin/env bash
# Speaker enrollment (SpeechBrain ECAPA) on Kaggle — run after setup-kaggle.sh
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

export VOICE_BACKEND="${VOICE_BACKEND:-ecapa}"

echo "==> SpeechBrain ECAPA voice ID (VOICE_BACKEND=$VOICE_BACKEND)"
PIP install -r requirements-voice-id.txt

TORCH_VER="$(RUN -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
echo "==> torch version: $TORCH_VER (Coqui venv often 2.0–2.2; voice_id patches torch.amp for SpeechBrain)"

if [ "$VOICE_BACKEND" = "resemblyzer" ]; then
  echo "==> Legacy Resemblyzer extras"
  PIP install -U "setuptools>=65.0.0,<82" wheel
  PIP install "resemblyzer>=0.1.1" "librosa>=0.9,<0.11"
fi

if [ "$VOICE_BACKEND" = "mfcc" ]; then
  echo "==> MFCC + cosine backend (librosa, CPU)"
  PIP install "librosa>=0.10,<0.12" "soundfile>=0.12.0"
fi

echo "==> Verify ffmpeg"
ffmpeg -version | head -n 1

echo "==> Verify voice encoder (first run may download ECAPA weights)"
RUN -c "
from voice_id import status
import json, sys
st = status()
print(json.dumps(st, indent=2))
if not st.get('ready'):
    sys.exit(1)
vad = st.get('vad') or {}
if not vad.get('ready'):
    print('WARN: Silero VAD not ready — passive gate uses passthrough.', vad.get('error'), file=sys.stderr)
"

echo ""
echo "Voice ID ready (VOICE_BACKEND=$VOICE_BACKEND). Re-enroll all officials on Mac after backend change."
echo "Endpoints: POST /voice/enroll, /voice/identify, /voice/passive"
