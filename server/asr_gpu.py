"""Optional GPU ASR (faster-whisper) for command transcript when browser STT fails."""
from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from typing import Any

_MODEL = None
_READY = False
_ERROR: str | None = None


def status() -> dict[str, Any]:
    global _READY, _ERROR
    if _READY:
        return {"ready": True, "backend": "faster-whisper", "model": os.environ.get("ASR_MODEL", "base")}
    if _ERROR:
        return {"ready": False, "error": _ERROR}
    return {"ready": False, "hint": "pip install faster-whisper or set ASR disabled"}


def _lazy_init() -> bool:
    global _MODEL, _READY, _ERROR
    if _READY:
        return True
    if _ERROR and "not installed" in _ERROR:
        return False
    try:
        from faster_whisper import WhisperModel

        model_size = os.environ.get("ASR_MODEL", "base")
        device = "cuda" if os.environ.get("GPU", "1") in ("1", "true", "yes") else "cpu"
        compute = os.environ.get("ASR_COMPUTE", "float16" if device == "cuda" else "int8")
        print(f"[asr] Loading faster-whisper {model_size} on {device}...")
        _MODEL = WhisperModel(model_size, device=device, compute_type=compute)
        _READY = True
        _ERROR = None
        print("[asr] Whisper ASR ready")
        return True
    except ImportError:
        _ERROR = "faster-whisper not installed (pip install faster-whisper)"
        print("[asr]", _ERROR)
        return False
    except Exception as exc:
        _ERROR = str(exc)
        print("[asr] init failed:", exc)
        return False


def _to_wav_16k(src_path: str, dst_path: str) -> None:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            src_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            dst_path,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr or proc.stdout}")


def transcribe_base64(audio_b64: str, mime: str = "audio/webm") -> dict[str, Any]:
    if not audio_b64:
        return {"ok": False, "reason": "no_audio", "text": ""}
    if not _lazy_init():
        return {"ok": False, "reason": "asr_unavailable", "error": _ERROR, "text": ""}
    ext = ".webm" if "webm" in (mime or "") else ".bin"
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in" + ext)
        wav = os.path.join(tmp, "audio.wav")
        with open(src, "wb") as f:
            f.write(base64.b64decode(audio_b64))
        _to_wav_16k(src, wav)
        segments, info = _MODEL.transcribe(
            wav,
            language=os.environ.get("ASR_LANGUAGE", "en"),
            beam_size=int(os.environ.get("ASR_BEAM_SIZE", "1")),
        )
        parts = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
        text = " ".join(parts).strip()
        return {
            "ok": bool(text),
            "text": text,
            "language": getattr(info, "language", None),
            "backend": "faster-whisper",
        }
