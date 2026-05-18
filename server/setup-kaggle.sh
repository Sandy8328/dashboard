#!/bin/bash
# Kaggle setup: Python 3.10/3.11 + Coqui GPU, else Python 3.12 + Edge TTS.
set -e
cd "$(dirname "$0")"

pick_python() {
  local cmd ver
  for cmd in python3.11 python3.10; do
    if command -v "$cmd" >/dev/null 2>&1; then
      ver="$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      echo "Found $cmd (Python $ver)" >&2
      echo "$cmd"
      return 0
    fi
  done
  if command -v python3 >/dev/null 2>&1; then
    ver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    echo "Using python3 (Python $ver)" >&2
    echo "python3"
    return 0
  fi
  echo "python3"
}

create_venv() {
  rm -rf .venv
  if "$PY" -m venv .venv 2>/dev/null; then
    return 0
  fi
  echo "venv failed for $PY — installing python${VER}-venv (Kaggle/Debian)..."
  if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq 2>/dev/null || apt-get update -qq 2>/dev/null || true
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
      "python${VER}-venv" "python${VER}-dev" "python${VER}-distutils" 2>/dev/null || \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      "python${VER}-venv" "python${VER}-dev" 2>/dev/null || true
  fi
  if "$PY" -m venv .venv 2>/dev/null; then
    return 0
  fi
  echo "Trying virtualenv package..."
  "$PY" -m pip install --user virtualenv
  "$PY" -m virtualenv .venv
}

PY="$(pick_python)"
VER="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Using $PY (Python $VER)"

PIP() {
  if [ "${USE_VENV:-1}" = "1" ] && [ -x ".venv/bin/python" ]; then
    .venv/bin/python -m pip "$@"
  else
    "$PY" -m pip "$@"
  fi
}

USE_VENV=1
if ! create_venv; then
  echo "Could not create .venv — will install with $PY --user (less isolated)"
  USE_VENV=0
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "venv python: $(python -c 'import sys; print(sys.executable, sys.version)')"
fi

PIP install -U pip wheel "setuptools>=65.0.0,<82"

if [ "$VER" = "3.10" ] || [ "$VER" = "3.11" ]; then
  echo "Installing GPU Coqui TTS (requirements-gpu.txt, numpy 1.22 for TTS)..."
  PIP install "numpy==1.22.0"
  PIP install -r requirements-gpu.txt
  echo "Edge TTS (male fallback) included in requirements-gpu.txt"
  TTS_BACKEND=coqui
else
  echo "Python $VER — Coqui needs 3.10/3.11. Using Edge TTS instead..."
  PIP install -r requirements-kaggle.txt
  TTS_BACKEND=edge
fi

echo "$PY" > .python-bin
echo "$TTS_BACKEND" > .tts-backend
echo "$USE_VENV" > .use-venv

echo ""
echo "Done (backend=$TTS_BACKEND, venv=$USE_VENV)."
echo "Start:"
if [ "$USE_VENV" = "1" ]; then
  echo "  source .venv/bin/activate && TTS_BACKEND=$TTS_BACKEND GPU=1 python modelServer.py"
else
  echo "  TTS_BACKEND=$TTS_BACKEND GPU=1 $PY modelServer.py"
fi
