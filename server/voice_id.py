"""
Speaker embedding — stateless: browser stores profiles, GPU computes vectors.
Default backend: SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb).
Optional: VOICE_BACKEND=mfcc (MFCC mean/std + cosine similarity, CPU).
Optional: VOICE_BACKEND=resemblyzer for legacy Resemblyzer.

Profiles must be re-enrolled when VOICE_BACKEND / embedding space changes.
"""
from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import threading
import time
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

def _resolve_backend_id() -> str:
    if VOICE_BACKEND in ("ecapa", "speechbrain", "ecapa-tdnn"):
        return "speechbrain-ecapa-voxceleb"
    if VOICE_BACKEND in ("mfcc", "mfcc-cosine"):
        return "mfcc-cosine"
    if VOICE_BACKEND in ("resemblyzer", "resembly"):
        return "resemblyzer"
    raise ValueError(
        f"Unknown VOICE_BACKEND={VOICE_BACKEND!r}. "
        "Use ecapa, mfcc, or resemblyzer."
    )


BACKEND_ID = _resolve_backend_id()

def _env_float(name: str, default: float) -> float:
    return float(os.environ[name]) if name in os.environ else default


# Per-backend cosine thresholds (browser mic; tune via env)
if BACKEND_ID.startswith("speechbrain"):
    _DEF_MATCH, _DEF_WEAK, _DEF_MARGIN = 0.62, 0.55, 0.05
    _DEF_MULTI_MATCH, _DEF_MULTI_WEAK, _DEF_MULTI_MARGIN = 0.65, 0.58, 0.07
elif BACKEND_ID == "mfcc-cosine":
    _DEF_MATCH, _DEF_WEAK, _DEF_MARGIN = 0.78, 0.72, 0.04
    _DEF_MULTI_MATCH, _DEF_MULTI_WEAK, _DEF_MULTI_MARGIN = 0.82, 0.76, 0.06
else:
    _DEF_MATCH, _DEF_WEAK, _DEF_MARGIN = 0.74, 0.70, 0.06
    _DEF_MULTI_MATCH, _DEF_MULTI_WEAK, _DEF_MULTI_MARGIN = 0.74, 0.70, 0.08

MATCH_THRESHOLD = _env_float("VOICE_MATCH_THRESHOLD", _DEF_MATCH)
WEAK_MATCH = _env_float("VOICE_WEAK_MATCH", _DEF_WEAK)
MATCH_MARGIN = _env_float("VOICE_MATCH_MARGIN", _DEF_MARGIN)
MULTI_MATCH_THRESHOLD = _env_float("VOICE_MULTI_MATCH_THRESHOLD", _DEF_MULTI_MATCH)
MULTI_WEAK_MATCH = _env_float("VOICE_MULTI_WEAK_MATCH", _DEF_MULTI_WEAK)
MULTI_MATCH_MARGIN = _env_float("VOICE_MULTI_MATCH_MARGIN", _DEF_MULTI_MARGIN)

MIN_ENROLL_SEC = float(os.environ.get("VOICE_MIN_ENROLL_SEC", "26"))
MIN_ENROLL_SPEECH_SEC = float(os.environ.get("VOICE_MIN_ENROLL_SPEECH_SEC", "20"))
MIN_IDENTIFY_SEC = float(os.environ.get("VOICE_MIN_IDENTIFY_SEC", "1.2"))
MIN_IDENTIFY_SPEECH_SEC = float(os.environ.get("VOICE_MIN_IDENTIFY_SPEECH_SEC", "0.8"))
PASSIVE_MIN_SEC = float(os.environ.get("VOICE_PASSIVE_MIN_SEC", "1.0"))

MIN_RMS = float(os.environ.get("VOICE_MIN_RMS", "0.008"))
MIN_PEAK = float(os.environ.get("VOICE_MIN_PEAK", "0.02"))
MAX_CLIP_RATIO = float(os.environ.get("VOICE_MAX_CLIP_RATIO", "0.35"))

_ENCODER = None
_READY = False
_INIT_ERROR = None
_INIT_ERROR_AT = 0.0
_INIT_LOCK = threading.Lock()
_INIT_RETRY_COOLDOWN_SEC = float(os.environ.get("VOICE_INIT_RETRY_COOLDOWN_SEC", "90"))


def _patch_torch_amp_compat() -> None:
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
    try:
        import torchaudio
    except Exception:
        return
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]


if BACKEND_ID.startswith("speechbrain"):
    _patch_torch_amp_compat()
    _patch_torchaudio_compat()


def _reject(reason: str, **details: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "matched": False,
        "reason": reason,
        "details": details,
        "embedding_backend": BACKEND_ID,
    }


def _thresholds_for_profiles(num_profiles: int) -> tuple[float, float, float]:
    if num_profiles >= 2:
        return MULTI_MATCH_THRESHOLD, MULTI_WEAK_MATCH, MULTI_MATCH_MARGIN
    return MATCH_THRESHOLD, WEAK_MATCH, MATCH_MARGIN


def _lazy_init() -> bool:
    global _ENCODER, _READY, _INIT_ERROR, _INIT_ERROR_AT
    if _READY:
        return True
    with _INIT_LOCK:
        if _READY:
            return True
        if _INIT_ERROR:
            if time.time() - _INIT_ERROR_AT < _INIT_RETRY_COOLDOWN_SEC:
                return False
            print(
                f"[voice_id] Retrying init after cooldown ({BACKEND_ID})…"
            )
            _INIT_ERROR = None
        try:
            if BACKEND_ID.startswith("speechbrain"):
                _ENCODER = _load_ecapa_encoder()
            elif BACKEND_ID == "mfcc-cosine":
                _ENCODER = _load_mfcc_encoder()
            else:
                _ENCODER = _load_resemblyzer_encoder()
            _READY = True
            _INIT_ERROR = None
            print(f"Voice ID ready: {BACKEND_ID}")
            return True
        except Exception as exc:
            _INIT_ERROR = str(exc)
            _INIT_ERROR_AT = time.time()
            print(f"Voice ID init failed ({BACKEND_ID}):", exc)
            return False


def _resolve_torch_device_str() -> str:
    """SpeechBrain run_opts expects 'cuda:0', not bare 'cuda' (parser needs type:index)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _load_ecapa_encoder() -> Any:
    _patch_torch_amp_compat()
    _patch_torchaudio_compat()
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    device = _resolve_torch_device_str()
    print(f"Loading SpeechBrain ECAPA ({ECAPA_MODEL}) on {device}...")
    classifier = EncoderClassifier.from_hparams(
        source=ECAPA_MODEL,
        savedir=ECAPA_SAVEDIR,
        run_opts={"device": device},
    )
    if hasattr(classifier, "eval"):
        classifier.eval()
    if hasattr(classifier, "device"):
        try:
            classifier.device = device
        except Exception:
            pass
    print(f"[voice_id] ECAPA loaded on {device} (cuda available={torch.cuda.is_available()})")
    return {"kind": "ecapa", "model": classifier, "device": device}


def _load_resemblyzer_encoder() -> Any:
    from resemblyzer import VoiceEncoder, preprocess_wav

    enc = VoiceEncoder()
    return {"kind": "resemblyzer", "encoder": enc, "preprocess_wav": preprocess_wav}


def _load_mfcc_encoder() -> dict[str, Any]:
    """Lightweight CPU speaker features: MFCC mean+std, L2-normalized for cosine match."""
    try:
        import librosa  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "MFCC backend requires librosa. pip install 'librosa>=0.10,<0.12'"
        ) from exc
    n_mfcc = int(os.environ.get("VOICE_MFCC_N_COEFF", "20"))
    n_fft = int(os.environ.get("VOICE_MFCC_N_FFT", "512"))
    hop = int(os.environ.get("VOICE_MFCC_HOP", "160"))
    print(
        f"[voice_id] MFCC encoder ready n_mfcc={n_mfcc} "
        f"n_fft={n_fft} hop={hop} dims={n_mfcc * 2}"
    )
    return {
        "kind": "mfcc",
        "n_mfcc": n_mfcc,
        "n_fft": n_fft,
        "hop_length": hop,
        "dims": n_mfcc * 2,
    }


def _mfcc_embed_path(wav_path: str, cfg: dict[str, Any]) -> np.ndarray:
    import librosa

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if y.size < sr * 0.3:
        raise ValueError("audio too short for MFCC")
    n_mfcc = int(cfg.get("n_mfcc", 20))
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=int(cfg.get("n_fft", 512)),
        hop_length=int(cfg.get("hop_length", 160)),
    )
    # Utterance-level CMVN
    mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
    feat = np.concatenate(
        [np.mean(mfcc, axis=1), np.std(mfcc, axis=1)],
        dtype=np.float64,
    )
    return feat


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
        "voice_backend_env": VOICE_BACKEND,
        "embedding_backend": BACKEND_ID,
        "ecapa_model": ECAPA_MODEL if BACKEND_ID.startswith("speechbrain") else None,
        "mfcc_n_coeff": (
            int(os.environ.get("VOICE_MFCC_N_COEFF", "20"))
            if BACKEND_ID == "mfcc-cosine"
            else None
        ),
        "mfcc_embedding_dims": (
            (_ENCODER or {}).get("dims") if BACKEND_ID == "mfcc-cosine" and _ENCODER else None
        ),
        "match_threshold": MATCH_THRESHOLD,
        "weak_match": WEAK_MATCH,
        "match_margin": MATCH_MARGIN,
        "multi_match_threshold": MULTI_MATCH_THRESHOLD,
        "multi_match_margin": MULTI_MATCH_MARGIN,
        "min_enroll_sec": MIN_ENROLL_SEC,
        "min_identify_sec": MIN_IDENTIFY_SEC,
        "vad": vad,
        "error": None if ok else _INIT_ERROR,
        "re_enroll_required": (
            f"Re-enroll all voices when embedding backend changes (active: {BACKEND_ID})."
        ),
    }


def _write_b64(path: str, b64: str) -> None:
    raw = base64.b64decode(b64)
    if not raw or len(raw) < 32:
        raise ValueError("audio_base64 empty or too small")
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


def _wav_duration_sec(path: str) -> float:
    import wave

    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _read_wav_mono_float(path: str) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        sw = wf.getsampwidth()
    if sw == 2:
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
    elif sw == 4:
        samples = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported wav sample width: {sw}")
    return samples, rate


def _energy_speech_seconds(samples: np.ndarray, rate: int) -> float:
    """Frame RMS speech estimate when Silero VAD is unavailable."""
    frame = max(1, int(rate * 0.03))
    thresh = max(MIN_RMS * 2.5, 0.012)
    speech_frames = 0
    total_frames = 0
    for i in range(0, len(samples) - frame, frame):
        chunk = samples[i : i + frame]
        if float(np.sqrt(np.mean(chunk * chunk))) >= thresh:
            speech_frames += 1
        total_frames += 1
    if total_frames < 1:
        return 0.0
    return (speech_frames * frame) / float(rate)


def _vad_for_wav(wav_path: str) -> dict[str, Any]:
    try:
        from vad_gate import analyze_wav_16k

        return analyze_wav_16k(wav_path)
    except Exception as exc:
        return {
            "speech_detected": True,
            "speech_seconds": 0.0,
            "vad_backend": "passthrough",
            "note": str(exc),
        }


def _analyze_audio_quality(
    wav_path: str,
    *,
    min_duration_sec: float,
    min_speech_sec: float,
) -> dict[str, Any]:
    """Gate silence/clipped/short clips before embedding."""
    try:
        samples, rate = _read_wav_mono_float(wav_path)
    except Exception as exc:
        return _reject("invalid_audio", error=str(exc))

    if samples.size < 8 or rate < 8000:
        return _reject("invalid_audio", samples=len(samples), sample_rate=rate)

    duration_sec = float(samples.size) / float(rate)
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    clip_ratio = float(np.mean(np.abs(samples) > 0.98))

    vad = _vad_for_wav(wav_path)
    speech_sec = float(vad.get("speech_seconds") or 0.0)
    vad_backend = str(vad.get("vad_backend") or "")
    if speech_sec < min_speech_sec and vad_backend in (
        "passthrough",
        "error_passthrough",
    ):
        energy_sec = _energy_speech_seconds(samples, rate)
        if energy_sec > speech_sec:
            speech_sec = energy_sec
            vad = {
                **vad,
                "speech_seconds_energy": round(energy_sec, 3),
                "vad_backend": vad_backend + "+energy",
            }

    details = {
        "duration_sec": round(duration_sec, 3),
        "speech_sec": round(speech_sec, 3),
        "rms": round(rms, 5),
        "peak": round(peak, 5),
        "clip_ratio": round(clip_ratio, 4),
        "min_duration_sec": min_duration_sec,
        "min_speech_sec": min_speech_sec,
        "vad": vad,
    }

    if duration_sec < min_duration_sec:
        return _reject("audio_too_short", **details)
    if speech_sec < min_speech_sec:
        return _reject("insufficient_speech", **details)
    if rms < MIN_RMS:
        return _reject("rms_too_low", **details)
    if peak < MIN_PEAK:
        return _reject("peak_too_low", **details)
    if clip_ratio > MAX_CLIP_RATIO:
        return _reject("clipping_too_high", **details)

    return {"ok": True, "details": details}


def _load_wav_tensor_16k_mono(wav_path: str, classifier: Any) -> Any:
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


def _ecapa_infer_device(classifier: Any) -> Any:
    import torch

    dev = getattr(classifier, "device", None)
    if dev is not None:
        if isinstance(dev, str) and dev.strip() == "cuda":
            dev = "cuda:0"
        return torch.device(dev) if isinstance(dev, str) else dev
    mods = getattr(classifier, "mods", None)
    if mods is not None:
        for mod in (mods.values() if hasattr(mods, "values") else []):
            params_fn = getattr(mod, "parameters", None)
            if not callable(params_fn):
                continue
            try:
                return next(params_fn()).device
            except StopIteration:
                continue
    if isinstance(_ENCODER, dict) and _ENCODER.get("device"):
        d = _ENCODER["device"]
        if isinstance(d, str) and d.strip() == "cuda":
            d = "cuda:0"
        return torch.device(d) if isinstance(d, str) else d
    return torch.device(_resolve_torch_device_str())


def _ecapa_embed_path(classifier: Any, wav_path: str) -> np.ndarray:
    import torch

    batch = _load_wav_tensor_16k_mono(wav_path, classifier)
    dev = _ecapa_infer_device(classifier)
    batch = batch.to(dev)
    with torch.no_grad():
        emb = classifier.encode_batch(batch)
    return np.asarray(emb.detach().cpu().numpy()).squeeze().astype(np.float64)


def _embed_path(wav_path: str) -> list[float]:
    if not _lazy_init():
        raise RuntimeError(_INIT_ERROR or "Voice encoder not ready")
    kind = _ENCODER["kind"]
    if kind == "ecapa":
        vec = _ecapa_embed_path(_ENCODER["model"], wav_path)
    elif kind == "mfcc":
        vec = _mfcc_embed_path(wav_path, _ENCODER)
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
            continue
        if pb == BACKEND_ID:
            out.append(p)
    return out


def _decode_to_wav(audio_b64: str, mime: str, tmp: str) -> str:
    ext = ".webm" if "webm" in (mime or "") else ".bin"
    src = os.path.join(tmp, "in" + ext)
    wav = os.path.join(tmp, "audio.wav")
    _write_b64(src, audio_b64)
    _to_wav_16k(src, wav)
    return wav


def embed_from_base64(audio_b64: str, mime: str = "audio/webm") -> dict[str, Any]:
    if not audio_b64:
        return _reject("no_audio")
    with tempfile.TemporaryDirectory() as tmp:
        wav = _decode_to_wav(audio_b64, mime, tmp)
        quality = _analyze_audio_quality(
            wav,
            min_duration_sec=MIN_ENROLL_SEC,
            min_speech_sec=MIN_ENROLL_SPEECH_SEC,
        )
        if not quality.get("ok"):
            return quality
        embedding = _embed_path(wav)
        dur = _wav_duration_sec(wav)
        return {
            "ok": True,
            "embedding": embedding,
            "duration_ms": int(dur * 1000),
            "dims": len(embedding),
            "embedding_backend": BACKEND_ID,
            "audio_quality": quality.get("details"),
        }


def _identify_wav_path(wav_path: str, profiles: list[dict]) -> dict[str, Any]:
    quality = _analyze_audio_quality(
        wav_path,
        min_duration_sec=MIN_IDENTIFY_SEC,
        min_speech_sec=MIN_IDENTIFY_SPEECH_SEC,
    )
    if not quality.get("ok"):
        out = dict(quality)
        out["accepted"] = False
        return out

    compatible = _compatible_profiles(profiles)
    if not profiles:
        return {**_reject("no_profiles"), "accepted": False}
    if not compatible:
        return {
            **_reject(
                "stale_embeddings",
                profile_count=len(profiles),
                compatible_count=0,
                hint="Re-enroll all voices after backend change.",
            ),
            "accepted": False,
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
            **_reject("no_embeddings", compatible_count=len(compatible)),
            "accepted": False,
        }

    ranked.sort(key=lambda x: x[0], reverse=True)
    best_score, best = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    second_profile = ranked[1][1] if len(ranked) > 1 else None
    margin = float(best_score - second_score)
    conf = round(max(0.0, best_score), 3)
    th, _weak, margin_min = _thresholds_for_profiles(len(compatible))

    base: dict[str, Any] = {
        "ok": True,
        "confidence": conf,
        "second_best": round(max(0.0, second_score), 3),
        "second_best_name": (second_profile or {}).get("name"),
        "margin": round(max(0.0, margin), 3),
        "threshold": th,
        "match_margin_min": margin_min,
        "profile_count": len(profiles),
        "compatible_count": len(compatible),
        "embedding_backend": BACKEND_ID,
        "audio_quality": quality.get("details"),
        "accepted": False,
    }

    def _log_identify(out: dict[str, Any]) -> dict[str, Any]:
        print(
            "[voice_id] identify",
            f"backend={BACKEND_ID}",
            f"top={out.get('confidence')}",
            f"second={out.get('second_best')}",
            f"margin={out.get('margin')}",
            f"threshold={out.get('threshold')}",
            f"margin_min={out.get('match_margin_min')}",
            f"accepted={out.get('accepted')}",
            f"matched={out.get('matched')}",
            f"reason={out.get('reason', '')}",
            f"name={out.get('name', '')}",
        )
        return out

    if len(compatible) == 1:
        if best_score >= th:
            base.update(
                {
                    "matched": True,
                    "accepted": True,
                    "id": best.get("id"),
                    "name": best.get("name"),
                    "match_mode": "threshold",
                }
            )
            return _log_identify(base)
        base["reason"] = "uncertain" if best_score >= WEAK_MATCH else "no_match"
        base["matched"] = False
        return _log_identify(base)

    if best_score >= th and margin >= margin_min:
        base.update(
            {
                "matched": True,
                "accepted": True,
                "id": best.get("id"),
                "name": best.get("name"),
                "match_mode": "threshold_margin",
            }
        )
        return _log_identify(base)

    base["matched"] = False
    if best_score >= WEAK_MATCH and margin < margin_min:
        base["reason"] = "uncertain"
    elif best_score >= WEAK_MATCH:
        base["reason"] = "uncertain"
    else:
        base["reason"] = "no_match"
    return _log_identify(base)


def identify_from_base64(
    audio_b64: str, profiles: list[dict], mime: str = "audio/webm"
) -> dict[str, Any]:
    if not audio_b64:
        return {**_reject("no_audio"), "accepted": False}
    if not profiles:
        return {**_reject("no_profiles"), "accepted": False}
    with tempfile.TemporaryDirectory() as tmp:
        wav = _decode_to_wav(audio_b64, mime, tmp)
        return _identify_wav_path(wav, profiles)


def passive_from_base64(
    audio_b64: str,
    profiles: list[dict],
    mime: str = "audio/webm",
    transcript: str = "",
) -> dict[str, Any]:
    if not audio_b64:
        return {"matched": False, "reason": "no_audio", "speech_detected": False, "ok": False}
    with tempfile.TemporaryDirectory() as tmp:
        wav = _decode_to_wav(audio_b64, mime, tmp)
        dur = _wav_duration_sec(wav)
        vad_info = _vad_for_wav(wav)
        if not vad_info.get("speech_detected"):
            return {
                "ok": False,
                "matched": False,
                "reason": "no_speech",
                "speech_detected": False,
                "vad": vad_info,
                "transcript": (transcript or "").strip(),
            }
        if dur < PASSIVE_MIN_SEC:
            return {
                "ok": False,
                "matched": False,
                "reason": "audio_too_short",
                "speech_detected": True,
                "vad": vad_info,
                "duration_ms": int(dur * 1000),
                "transcript": (transcript or "").strip(),
            }
        if not profiles:
            return {
                "ok": False,
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
