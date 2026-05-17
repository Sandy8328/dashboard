#!/usr/bin/env bash
# Build dashboard-kaggle.zip for upload to Kaggle (VS Code or notebook).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
OUT="dashboard-kaggle.zip"
rm -f "$OUT"

zip -r "$OUT" \
  kaggle-restore.sh \
  package.json \
  package-lock.json \
  server/run-model-server.sh \
  server/modelServer.py \
  server/index.js \
  server/talking_avatar.py \
  server/musetalk_runner.py \
  server/wav2lip_runner.py \
  server/patch_wav2lip_librosa.py \
  server/setup-kaggle.sh \
  server/setup-wav2lip.sh \
  server/setup-musetalk.sh \
  server/requirements.txt \
  server/requirements-gpu.txt \
  server/requirements-kaggle.txt \
  server/requirements-wav2lip.txt \
  server/requirements-musetalk.txt \
  shared/assistant.js \
  public/assets \
  .env.example \
  -x "*.pyc" "*__pycache__*" "*.DS_Store"

echo ""
echo "Created: $ROOT/$OUT ($(du -h "$OUT" | cut -f1))"
echo "Upload to Kaggle → unzip in /kaggle/working → bash kaggle-restore.sh"
