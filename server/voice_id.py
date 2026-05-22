"""
Speaker embedding — stateless: browser stores profiles, GPU computes vectors.
Default backend: SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb).
Optional: VOICE_BACKEND=resemblyzer for legacy Resemblyzer.
"""
from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from typing import Any

import numpy as np

VOICE_BACKEND = os.environ.get("VOICE_BACKEND", "ecapa").strip().lower()
ECAPA_MODEL = os.environ.get(
    "VOICE_ECAPA_MODEL", "speechbrain/spkrec-ecapa-voxceleb"
)
ECAPA_SAVEDIR = os.environ.get(
    "VOICE_ECAPA_SAVEDIR",
    os.path.join(os.path.dirname(__file__), "pretrained_models", "spkrec-ecapa-voxceleb"),
)

BACKEND_ID = (
    "speechbrain-ecapa-voxceleb"
    if VOICE_BACKEND in ("ecapa", "speechbrain", "ecapa-tdnn")
    else "resemblyzer"
)

if BACKEND_ID.startswith("speechbrain"):
    _DEF_MATCH = 0.70
    _DEF_WEAK = 0.65
    _DEF_MARGIN = 0.05
    _DEF_MULTI_MATCH = 0.68
    _DEF_MULTI_WEAK = 0.63
    _DEF_MULTI_MARGIN = 0.06
else:
    _DEF_MATCH = 0.64
    _DEF_WEAK = 0.62
    _DEF_MARGIN = 0.07
    _DEF_MULTI_MATCH = 0.60
    _DEF_MULTI_WEAK = 0.56
    _DEF_MULTI_MARGIN = 0.08

MATCH_THRESHOLD = float(os.environ.get("VOICE_MATCH_THRESHOLD", str(_DEF_MATCH)))
WEAK_MATCH = float(os.environ.get("VOICE_WEAK_MATCH", str(_DEF_WEAK)))
MATCH_MARGIN = float(os.environ.get("VOICE_MATCH_MARGIN", str(_DEF_MARGIN)))
MULTI_MATCH_THRESHOLD = float(
    os.environ.get("VOICE_MULTI_MATCH_THRESHOLD", str(_DEF_MULTI_MATCH))
)
MULTI_WEAK_MATCH = float(os.environ.get("VOICE_MULTI_WEAK_MATCH", str(_DEF_MULTI_WEAK)))
MULTI_MATCH_MARGIN = float(
    os.environ.get("VOICE_MULTI_MATCH_MARGIN", str(_DEF_MULTI_MARGIN))
)
MIN_ENROLL_SEC = float(os.environ.get("VOICE_MIN_ENROLL_SEC", "4"))
PASSIVE_MIN_SEC = float(os.environ.get("VOICE_PASSIVE_MIN_SEC", "0.35"))

_ENCODER = None
_READY = False
_INIT_ERROR = None


def _patch_torch_amp_compat() -> None:
    """
    SpeechBrain 1.x uses torch.amp.custom_fwd (PyTorch>=2.4).
    Coqui Kaggle venv often has torch 2.0–2.2 — map to torch.cuda.amp.
    """
    try:
        import torch
    except Exception:
        return
    amp = getattr(torch, "amp", None)
    if amp is None:
        return
    if hasattr(amp, "custom_fwd") and hasattr(amp, "custom_bwd"):
        return
    cuda_amp = getattr(torch.cuda, "amp", None)
    if cuda_amp is None or not hasattr(cuda_amp, "custom_fwd"):
        return

    def _wrap(cuda_dec: Any) -> Any:
        def compat_dec(*args: Any, **kwargs: Any) -> Any:
            kwargs.pop("device_type", None)
            if args and callable(args[0]):
                return cuda_dec(**kwargs)(args[0])
            return lambda fn: cuda_dec(**kwargs)(fn)

        return compat_dec

    if not hasattr(amp, "custom_fwd"):
        amp.custom_fwd = _wrap(cuda_amp.custom_fwd)  # type: ignore[attr-defined]
    if not hasattr(amp, "custom_bwd"):
        amp.custom_bwd = _wrap(cuda_amp.custom_bwd)  # type: ignore[attr-defined]
    print(
        "[voice_id] Patched torch.amp.custom_fwd/bwd for SpeechBrain "
        f"(torch {getattr(torch, '__version__', '?')})."
    )


def _patch_torchaudio_compat() -> None:
    """SpeechBrain 1.x may call torchaudio.list_audio_backends (removed in torchaudio 2.9+)."""
    try:
        import torchaudio
    except Exception:
        return
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]


if BACKEND_ID.startswith("speechbrain"):
    _patch_torch_amp_compat()
    _patch_torchaudio_compat()


def _thresholds_for_profiles(num_profiles: int) -> tuple[float, float, float]:
    if num_profiles >= 2:
        return MULTI_MATCH_THRESHOLD, MULTI_WEAK_MATCH, MULTI_MATCH_MARGIN
    return MATCH_THRESHOLD, WEAK_MATCH, MATCH_MARGIN


def _lazy_init() -> bool:
    global _ENCODER, _READY, _INIT_ERROR
    if _READY:
        return True
    if _INIT_ERROR:
        return False
    try:
        if BACKEND_ID.startswith("speechbrain"):
            _ENCODER = _load_ecapa_encoder()
        else:
            _ENCODER = _load_resemblyzer_encoder()
        _READY = True
        print(f"Voice ID ready: {BACKEND_ID}")
        return True
    except Exception as exc:
        _INIT_ERROR = str(exc)
        print(f"Voice ID init failed ({BACKEND_ID}):", exc)
        return False


def _load_ecapa_encoder() -> Any:
    _patch_torch_amp_compat()
    _patch_torchaudio_compat()
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SpeechBrain ECAPA ({ECAPA_MODEL}) on {device}...")
    classifier = EncoderClassifier.from_hparams(
        source=ECAPA_MODEL,
        savedir=ECAPA_SAVEDIR,
        run_opts={"device": device},
    )
    if hasattr(classifier, "eval"):
        classifier.eval()
    return {"kind": "ecapa", "model": classifier, "device": device}


def _load_resemblyzer_encoder() -> Any:
    from resemblyzer import VoiceEncoder, preprocess_wav

    enc = VoiceEncoder()
    return {"kind": "resemblyzer", "encoder": enc, "preprocess_wav": preprocess_wav}


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
        "backend": BACKEND_ID if ok else None,
        "voice_backend": VOICE_BACKEND,
        "ecapa_model": ECAPA_MODEL if BACKEND_ID.startswith("speechbrain") else None,
        "match_threshold": MATCH_THRESHOLD,
        "weak_match": WEAK_MATCH,
        "match_margin": MATCH_MARGIN,
        "multi_match_threshold": MULTI_MATCH_THRESHOLD,
        "vad": vad,
        "error": None if ok else _INIT_ERROR,
        "re_enroll_required": (
            "Old Resemblyzer profiles must be re-enrolled after switching to ECAPA."
            if BACKEND_ID.startswith("speechbrain")
            else None
        ),
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


def _load_wav_tensor_16k_mono(wav_path: str, classifier: Any) -> Any:
    """Load 16 kHz mono waveform as [batch=1, time] for encode_batch."""
    import torch

    signal = None
    fs = 16000
    if hasattr(classifier, "load_audio"):
        try:
            loaded = classifier.load_audio(wav_path)
            if isinstance(loaded, tuple):
                signal, fs = loaded[0], int(loaded[1]) if len(loaded) > 1 else 16000
            else:
                signal = loaded
        except Exception:
            signal = None

    if signal is None:
        try:
            import torchaudio

            signal, fs = torchaudio.load(wav_path)
        except Exception:
            import soundfile as sf

            data, fs = sf.read(wav_path, dtype="float32")
            arr = np.asarray(data, dtype=np.float32)
            signal = torch.from_numpy(arr.T if arr.ndim == 2 else arr)

    if not isinstance(signal, torch.Tensor):
        signal = torch.tensor(signal, dtype=torch.float32)

    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
    elif signal.dim() == 2 and signal.shape[0] > signal.shape[1]:
        signal = signal.transpose(0, 1)

    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    fs = int(fs) if fs else 16000
    if fs != 16000:
        import torchaudio

        signal = torchaudio.functional.resample(signal, fs, 16000)

    return signal


def _ecapa_embed_path(classifier: Any, wav_path: str) -> np.ndarray:
    """SpeechBrain ECAPA — encode_batch (encode_file not in all versions)."""
    import torch

    batch = _load_wav_tensor_16k_mono(wav_path, classifier)
    try:
        dev = next(classifier.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    batch = batch.to(dev)
    with torch.no_grad():
        emb = classifier.encode_batch(batch)
    return np.asarray(emb.detach().cpu().numpy()).squeeze().astype(np.float64)


def _embed_path(wav_path: str) -> list[float]:
    if not _lazy_init():
        raise RuntimeError(_INIT_ERROR or "Voice encoder not ready")
    if _ENCODER["kind"] == "ecapa":
        vec = _ecapa_embed_path(_ENCODER["model"], wav_path)
    else:
        wav = _ENCODER["preprocess_wav"](wav_path)
        vec = np.asarray(_ENCODER["encoder"].embed_utterance(wav)).flatten().astype(
            np.float64
        )
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    return [float(x) for x in vec]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    if a.shape != b.shape:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _compatible_profiles(profiles: list[dict]) -> list[dict]:
    out: list[dict] = []
    for p in profiles:
        pb = (p.get("embedding_backend") or p.get("embeddingBackend") or "").strip()
        if not pb:
            # Legacy Resemblyzer profiles — skip when using ECAPA
            if BACKEND_ID.startswith("speechbrain"):
                continue
            out.append(p)
            continue
        if pb == BACKEND_ID:
            out.append(p)
    return out


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
            raise ValueError(
                f"audio too short ({dur:.1f}s); speak at least {MIN_ENROLL_SEC:.0f}s"
            )
        embedding = _embed_path(wav)
        return {
            "embedding": embedding,
            "duration_ms": int(dur * 1000),
            "dims": len(embedding),
            "embedding_backend": BACKEND_ID,
        }


def _identify_wav_path(wav_path: str, profiles: list[dict]) -> dict[str, Any]:
    compatible = _compatible_profiles(profiles)
    if not profiles:
        return {"matched": False, "reason": "no_profiles", "embedding_backend": BACKEND_ID}
    if not compatible:
        return {
            "matched": False,
            "reason": "stale_embeddings",
            "embedding_backend": BACKEND_ID,
            "profile_count": len(profiles),
            "compatible_count": 0,
            "hint": "Re-enroll all voices after SpeechBrain upgrade (clear profiles, register again).",
        }
    probe = np.asarray(_embed_path(wav_path), dtype=np.float64)
    ranked: list[tuple[float, dict]] = []
    for p in compatible:
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
            "embedding_backend": BACKEND_ID,
        }
    ranked.sort(key=lambda x: x[0], reverse=True)
    best_score, best = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    second_profile = ranked[1][1] if len(ranked) > 1 else None
    margin = float(best_score - second_score)
    conf = round(max(0.0, best_score), 3)
    th, weak, margin_min = _thresholds_for_profiles(len(compatible))
    base = {
        "confidence": conf,
        "second_best": round(max(0.0, second_score), 3),
        "second_best_name": (second_profile or {}).get("name"),
        "margin": round(max(0.0, margin), 3),
        "threshold": th,
        "weak_threshold": weak,
        "match_margin_min": margin_min,
        "profile_count": len(profiles),
        "compatible_count": len(compatible),
        "embedding_backend": BACKEND_ID,
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
        return {"matched": False, "reason": "no_profiles", "embedding_backend": BACKEND_ID}
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
