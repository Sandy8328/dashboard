"""
Speaker embedding (Resemblyzer) — stateless: browser stores profiles, GPU computes vectors.
"""
from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from typing import Any

import numpy as np

_RESEMBLYZER = None
_ENCODER = None
_READY = False
_INIT_ERROR = None

MATCH_THRESHOLD = float(os.environ.get("VOICE_MATCH_THRESHOLD", "0.75"))
MIN_ENROLL_SEC = float(os.environ.get("VOICE_MIN_ENROLL_SEC", "4"))


def _lazy_init() -> bool:
    global _RESEMBLYZER, _ENCODER, _READY, _INIT_ERROR
    if _READY:
        return True
    if _INIT_ERROR:
        return False
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav

        _RESEMBLYZER = {"VoiceEncoder": VoiceEncoder, "preprocess_wav": preprocess_wav}
        _ENCODER = VoiceEncoder()
        _READY = True
        print("VoiceEncoder loaded for speaker ID.")
        return True
    except Exception as exc:
        _INIT_ERROR = str(exc)
        print("VoiceEncoder not available:", exc)
        return False


def status() -> dict:
    ok = _lazy_init()
    return {
        "ready": ok,
        "backend": "resemblyzer" if ok else None,
        "match_threshold": MATCH_THRESHOLD,
        "error": None if ok else _INIT_ERROR,
    }


def _write_b64(path: str, b64: str) -> None:
    raw = base64.b64decode(b64)
    with open(path, "wb") as f:
        f.write(raw)


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


def _load_wav(path: str) -> np.ndarray:
    if not _lazy_init():
        raise RuntimeError(_INIT_ERROR or "VoiceEncoder not ready")
    preprocess_wav = _RESEMBLYZER["preprocess_wav"]
    return preprocess_wav(path)


def _embed_path(wav_path: str) -> list[float]:
    wav = _load_wav(wav_path)
    emb = _ENCODER.embed_utterance(wav)
    return [float(x) for x in np.asarray(emb).flatten()]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def embed_from_base64(audio_b64: str, mime: str = "audio/webm") -> dict[str, Any]:
    if not audio_b64:
        raise ValueError("audio_base64 required")
    with tempfile.TemporaryDirectory() as tmp:
        ext = ".webm" if "webm" in (mime or "") else ".bin"
        src = os.path.join(tmp, "in" + ext)
        wav = os.path.join(tmp, "audio.wav")
        _write_b64(src, audio_b64)
        _to_wav_16k(src, wav)
        dur = _wav_duration_sec(wav)
        if dur < MIN_ENROLL_SEC:
            raise ValueError(f"audio too short ({dur:.1f}s); speak at least {MIN_ENROLL_SEC:.0f}s")
        embedding = _embed_path(wav)
        return {"embedding": embedding, "duration_ms": int(dur * 1000), "dims": len(embedding)}


def identify_from_base64(
    audio_b64: str, profiles: list[dict], mime: str = "audio/webm"
) -> dict[str, Any]:
    if not profiles:
        return {"matched": False, "reason": "no_profiles"}
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "probe.webm")
        wav = os.path.join(tmp, "probe.wav")
        _write_b64(src, audio_b64)
        _to_wav_16k(src, wav)
        probe = np.asarray(_embed_path(wav), dtype=np.float64)
        best = None
        best_score = -1.0
        for p in profiles:
            emb = p.get("embedding")
            if not emb or len(emb) < 8:
                continue
            score = _cosine(probe, np.asarray(emb, dtype=np.float64))
            if score > best_score:
                best_score = score
                best = p
        if best is None or best_score < MATCH_THRESHOLD:
            return {
                "matched": False,
                "confidence": round(max(0.0, best_score), 3),
                "threshold": MATCH_THRESHOLD,
            }
        return {
            "matched": True,
            "id": best.get("id"),
            "name": best.get("name"),
            "confidence": round(best_score, 3),
            "threshold": MATCH_THRESHOLD,
        }


def _wav_duration_sec(path: str) -> float:
    import wave

    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())
