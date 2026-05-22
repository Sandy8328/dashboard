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

MATCH_THRESHOLD = float(os.environ.get("VOICE_MATCH_THRESHOLD", "0.64"))
WEAK_MATCH = float(os.environ.get("VOICE_WEAK_MATCH", "0.62"))
MATCH_MARGIN = float(os.environ.get("VOICE_MATCH_MARGIN", "0.07"))
MULTI_MATCH_THRESHOLD = float(os.environ.get("VOICE_MULTI_MATCH_THRESHOLD", "0.60"))
MULTI_WEAK_MATCH = float(os.environ.get("VOICE_MULTI_WEAK_MATCH", "0.56"))
MULTI_MATCH_MARGIN = float(os.environ.get("VOICE_MULTI_MATCH_MARGIN", "0.08"))
MIN_ENROLL_SEC = float(os.environ.get("VOICE_MIN_ENROLL_SEC", "4"))
PASSIVE_MIN_SEC = float(os.environ.get("VOICE_PASSIVE_MIN_SEC", "0.35"))


def _thresholds_for_profiles(num_profiles: int) -> tuple[float, float, float]:
    """Stricter matching when 2+ enrolled voices (reduces Sandy/Tharun swaps)."""
    if num_profiles >= 2:
        return MULTI_MATCH_THRESHOLD, MULTI_WEAK_MATCH, MULTI_MATCH_MARGIN
    return MATCH_THRESHOLD, WEAK_MATCH, MATCH_MARGIN
 

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
    vad = {"ready": False}
    try:
        from vad_gate import status as vad_status

        vad = vad_status()
    except Exception:
        vad = {"ready": False}
    return {
        "ready": ok,
        "backend": "resemblyzer" if ok else None,
        "match_threshold": MATCH_THRESHOLD,
        "weak_match": WEAK_MATCH,
        "match_margin": MATCH_MARGIN,
        "vad": vad,
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


def _identify_wav_path(wav_path: str, profiles: list[dict]) -> dict[str, Any]:
    if not profiles:
        return {"matched": False, "reason": "no_profiles"}
    probe = np.asarray(_embed_path(wav_path), dtype=np.float64)
    ranked: list[tuple[float, dict]] = []
    for p in profiles:
        emb = p.get("embedding")
        if not emb or len(emb) < 8:
            continue
        score = _cosine(probe, np.asarray(emb, dtype=np.float64))
        ranked.append((score, p))
    if not ranked:
        return {
            "matched": False,
            "confidence": 0.0,
            "threshold": MATCH_THRESHOLD,
            "reason": "no_embeddings",
        }
    ranked.sort(key=lambda x: x[0], reverse=True)
    best_score, best = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    second_profile = ranked[1][1] if len(ranked) > 1 else None
    margin = float(best_score - second_score)
    conf = round(max(0.0, best_score), 3)
    th, weak, margin_min = _thresholds_for_profiles(len(profiles))
    base = {
        "confidence": conf,
        "second_best": round(max(0.0, second_score), 3),
        "second_best_name": (second_profile or {}).get("name"),
        "margin": round(max(0.0, margin), 3),
        "threshold": th,
        "weak_threshold": weak,
        "match_margin_min": margin_min,
        "profile_count": len(profiles),
    }
    matched = False
    match_mode = None
    if best_score >= th:
        matched = True
        match_mode = "threshold"
    elif best_score >= weak and margin >= margin_min:
        matched = True
        match_mode = "weak_margin"
    if matched and best is not None:
        return {
            **base,
            "matched": True,
            "id": best.get("id"),
            "name": best.get("name"),
            "match_mode": match_mode,
        }
    return {**base, "matched": False, "reason": "no_match"}


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
        return _identify_wav_path(wav, profiles)


def passive_from_base64(
    audio_b64: str,
    profiles: list[dict],
    mime: str = "audio/webm",
    transcript: str = "",
) -> dict[str, Any]:
    """VAD gate + speaker ID for passive logging (no LLM)."""
    if not audio_b64:
        return {"matched": False, "reason": "no_audio", "speech_detected": False}
    try:
        from vad_gate import analyze_wav_16k
    except Exception:
        analyze_wav_16k = None  # type: ignore

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "passive.webm")
        wav = os.path.join(tmp, "passive.wav")
        _write_b64(src, audio_b64)
        _to_wav_16k(src, wav)
        dur = _wav_duration_sec(wav)
        vad_info = (
            analyze_wav_16k(wav)
            if analyze_wav_16k
            else {"speech_detected": True, "vad_backend": "passthrough"}
        )
        if not vad_info.get("speech_detected"):
            return {
                "matched": False,
                "reason": "no_speech",
                "speech_detected": False,
                "vad": vad_info,
                "transcript": (transcript or "").strip(),
            }
        if dur < PASSIVE_MIN_SEC:
            return {
                "matched": False,
                "reason": "audio_too_short",
                "speech_detected": True,
                "vad": vad_info,
                "duration_ms": int(dur * 1000),
                "transcript": (transcript or "").strip(),
            }
        if not profiles:
            return {
                "matched": False,
                "reason": "no_profiles",
                "speech_detected": True,
                "vad": vad_info,
                "transcript": (transcript or "").strip(),
            }
        out = _identify_wav_path(wav, profiles)
        out["speech_detected"] = True
        out["vad"] = vad_info
        out["duration_ms"] = int(dur * 1000)
        out["transcript"] = (transcript or "").strip()
        out["passive"] = True
        return out


def _wav_duration_sec(path: str) -> float:
    import wave

    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())
