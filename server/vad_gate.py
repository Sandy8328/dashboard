"""
Silero VAD gate for passive speaker logging (optional — falls back if import fails).
Uses the pip package `silero-vad` — do NOT name this file silero_vad.py (shadows the package).
Audio is loaded via soundfile (not silero read_audio) to avoid torchaudio deprecation noise.
"""
from __future__ import annotations

import os
import warnings

_VAD_MODEL = None
_VAD_READY = False
_VAD_ERROR = None

MIN_SPEECH_SEC = float(os.environ.get("VAD_MIN_SPEECH_SEC", "0.45"))
SAMPLE_RATE = 16000

# Silero still uses torch internally; silence torchaudio 2.9 migration warnings on Kaggle.
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio(\.|$)")
warnings.filterwarnings("ignore", category=UserWarning, module=r"silero_vad(\.|$)")


def _load_audio_tensor_16k(wav_path: str):
    """16 kHz mono float32 tensor for Silero — no silero_vad.read_audio / torchaudio.sox."""
    import numpy as np
    import torch
    import soundfile as sf

    data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = arr.flatten()
    if int(sr) != SAMPLE_RATE and arr.size > 0:
        try:
            import librosa

            arr = librosa.resample(arr, orig_sr=int(sr), target_sr=SAMPLE_RATE)
        except Exception:
            pass
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def _lazy_init() -> bool:
    global _VAD_MODEL, _VAD_READY, _VAD_ERROR
    if _VAD_READY:
        return True
    if _VAD_ERROR:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            from silero_vad import load_silero_vad

            _VAD_MODEL = load_silero_vad()
        _VAD_READY = True
        print("Silero VAD loaded for passive speech gate.")
        return True
    except Exception as exc:
        _VAD_ERROR = str(exc)
        print("Silero VAD not available (passive gate disabled):", exc)
        return False


def status() -> dict:
    ok = _lazy_init()
    return {
        "ready": ok,
        "backend": "silero" if ok else None,
        "min_speech_sec": MIN_SPEECH_SEC,
        "error": None if ok else _VAD_ERROR,
    }


def analyze_wav_16k(wav_path: str) -> dict:
    """Return speech_detected, speech_seconds, segment count for a 16 kHz wav file."""
    if not _lazy_init():
        return {
            "speech_detected": True,
            "speech_seconds": 0.0,
            "segments": 0,
            "vad_backend": "passthrough",
            "note": _VAD_ERROR or "vad_unavailable",
        }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            from silero_vad import get_speech_timestamps

            wav = _load_audio_tensor_16k(wav_path)
            timestamps = get_speech_timestamps(
                wav,
                _VAD_MODEL,
                sampling_rate=SAMPLE_RATE,
                return_seconds=True,
            )
        speech_seconds = 0.0
        for seg in timestamps or []:
            speech_seconds += float(seg.get("end", 0)) - float(seg.get("start", 0))
        return {
            "speech_detected": speech_seconds >= MIN_SPEECH_SEC,
            "speech_seconds": round(speech_seconds, 3),
            "segments": len(timestamps or []),
            "vad_backend": "silero",
        }
    except Exception as exc:
        return {
            "speech_detected": True,
            "speech_seconds": 0.0,
            "segments": 0,
            "vad_backend": "error_passthrough",
            "note": str(exc),
        }
