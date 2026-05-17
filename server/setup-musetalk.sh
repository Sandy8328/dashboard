#!/usr/bin/env bash
# One-time MuseTalk 1.5 setup on Kaggle GPU (use a separate venv OR after Coqui — heavy deps).
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Clone MuseTalk"
if [ ! -d MuseTalk/.git ]; then
  git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git MuseTalk
fi

cd MuseTalk

echo "==> Install MuseTalk requirements (this takes several minutes)"
pip install -U "huggingface_hub[cli]" gdown
pip install -r requirements.txt || {
  echo "WARN: full requirements.txt failed — install mmcv/mmpose manually if inference fails"
}

echo "==> MMLab (face landmarks) — required for inference"
pip install --no-cache-dir -U openmim || true
mim install mmengine || true
mim install "mmcv==2.0.1" || true
mim install "mmdet==3.1.0" || true
mim install "mmpose==1.1.0" || true

echo "==> Download model weights (~several GB)"
bash ./download_weights.sh

cd ..

echo "==> Verify unet checkpoint"
UNET="MuseTalk/models/musetalkV15/unet.pth"
if [ ! -f "$UNET" ]; then
  echo "ERROR: $UNET missing — re-run download_weights.sh inside MuseTalk/"
  exit 1
fi
echo "==> unet.pth $(du -h "$UNET" | cut -f1)"

export USE_TALKING_AVATAR=musetalk
export MUSETALK_DIR="$(pwd)/MuseTalk"
export AVATAR_FACE_PATH="$(cd .. && pwd)/public/assets/mr-brain-avatar.png"

python -c "from musetalk_runner import status; print(status())"
echo ""
echo "Done. Start Python TTS + MuseTalk:"
echo "  USE_TALKING_AVATAR=2 TTS_BACKEND=coqui GPU=1 python modelServer.py"
echo ""
echo "Mac .env.local (optional video mode):"
echo "  VITE_AVATAR_MODE=musetalk"
echo "Officials demo (same PNG, canvas lips): keep VITE_AVATAR_MODE=canvas"
