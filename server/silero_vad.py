"""
Silero VAD gate for passive speaker logging (optional — falls back if import fails).
https://github.com/snakers4/silero-vad
"""
from __future__ import annotations

import os

_VAD_MODEL = None
_VAD_READY = False
_VAD_ERROR = None

MIN_SPEECH_SEC = float(os.environ.get("VAD_MIN_SPEECH_SEC", "0.45"))
SAMPLE_RATE = 16000


def _lazy_init() -> bool:
    global _VAD_MODEL, _VAD_READY, _VAD_ERROR
    if _VAD_READY:
        return True
    if _VAD_ERROR:
        return False
    try:
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
        from silero_vad import get_speech_timestamps, read_audio

        wav = read_audio(wav_path, sampling_rate=SAMPLE_RATE)
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
