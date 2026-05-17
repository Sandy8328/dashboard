#!/usr/bin/env bash
# Run once per Kaggle session after unzipping dashboard-kaggle.zip
# Usage: cd /kaggle/working/dashboard && bash kaggle-restore.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> Node proxy (port 4000)"
npm install --omit=dev

echo "==> Python TTS (port 5000)"
cd server
bash setup-kaggle.sh
bash setup-voice-id.sh

BACKEND="auto"
if [ -f .tts-backend ]; then
  BACKEND="$(cat .tts-backend)"
fi

echo ""
echo "=============================================="
echo "Restore done. Backend: $BACKEND"
echo ""
echo "Terminal 1 (Python):"
echo "  cd $ROOT/server && bash run-model-server.sh"
echo "  # or: source .venv/bin/activate && USE_TALKING_AVATAR=0 TTS_BACKEND=coqui GPU=1 python modelServer.py"
echo ""
echo "Terminal 2 (Node):"
echo "  cd $ROOT && USE_GPU_MODEL=1 GPU_MODEL_URL=http://127.0.0.1:5000 npm run dev:gpu-server"
echo ""
if [ "$BACKEND" = "edge" ]; then
  echo "NOTE: Coqui failed — using Edge TTS (no GPU Coqui). For Coqui on Kaggle use GPU"
  echo "      runtime with Python 3.10 and re-run: bash kaggle-restore.sh"
fi
echo "=============================================="
