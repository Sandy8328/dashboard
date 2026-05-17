#!/usr/bin/env bash
# One-time Wav2Lip setup on Kaggle GPU (Python 3.10 venv active, Coqui already installed).
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Clone Wav2Lip"
if [ ! -d Wav2Lip/.git ]; then
  git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git Wav2Lip
fi

mkdir -p Wav2Lip/checkpoints
CKPT="Wav2Lip/checkpoints/wav2lip_gan.pth"
# GitHub release 404 as of 2025 — use Hugging Face mirror (~436 MB)
MIN_BYTES=350000000
CKPT_URL="${WAV2LIP_CKPT_URL:-https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth}"

need_ckpt=1
if [ -f "$CKPT" ]; then
  size=$(wc -c < "$CKPT" | tr -d ' ')
  if [ "$size" -ge "$MIN_BYTES" ]; then
    need_ckpt=0
    echo "==> Checkpoint OK ($(du -h "$CKPT" | cut -f1))"
  else
    echo "==> Checkpoint too small (${size} bytes) — re-downloading"
    rm -f "$CKPT"
  fi
fi
if [ "$need_ckpt" -eq 1 ]; then
  echo "==> Download wav2lip_gan.pth (~436MB from Hugging Face) — do not interrupt"
  wget -q --show-progress -O "$CKPT" "$CKPT_URL" || \
    wget -q --show-progress -O "$CKPT" \
      "https://huggingface.co/rippertnt/wav2lip/resolve/main/checkpoints/wav2lip_gan.pth"
  size=$(wc -c < "$CKPT" | tr -d ' ')
  if [ "$size" -lt "$MIN_BYTES" ]; then
    echo "ERROR: download failed or incomplete (${size} bytes). Run setup again."
    exit 1
  fi
  echo "==> Downloaded $(du -h "$CKPT" | cut -f1)"
fi

echo "==> Install Wav2Lip deps (opencv + librosa, numpy pinned for Coqui)"
pip install -r requirements-wav2lip.txt

echo "==> Force numpy 1.22.0 if pip upgraded it"
pip install "numpy==1.22.0" --force-reinstall --no-deps

echo "==> Patch Wav2Lip for librosa 0.10+"
python patch_wav2lip_librosa.py "$(pwd)/Wav2Lip"

echo "==> Verify"
python -c "
import cv2
import librosa
import numpy as np
print('opencv', cv2.__version__)
print('librosa', librosa.__version__)
print('numpy', np.__version__)
assert np.__version__.startswith('1.22'), 'numpy must stay 1.22.x for Coqui'
"

export USE_TALKING_AVATAR=1
export WAV2LIP_DIR="$(pwd)/Wav2Lip"
export WAV2LIP_CHECKPOINT="$(pwd)/$CKPT"
export AVATAR_FACE_PATH="$(cd .. && pwd)/public/assets/mr-brain-avatar.png"

python -c "from wav2lip_runner import status; print(status())"
echo ""
echo "Done. Start:"
echo "  USE_TALKING_AVATAR=1 TTS_BACKEND=coqui GPU=1 python modelServer.py"
