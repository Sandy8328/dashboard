"""TTS server (port 5000). Coqui GPU on Py 3.10/11, Edge TTS fallback on Py 3.12 (Kaggle)."""
import asyncio
import base64
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL_NAME = os.environ.get("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC")
USE_GPU = os.environ.get("GPU", "1").lower() in ("1", "true", "yes")
SAMPLE_RATE = 22050
LIP_FPS = int(os.environ.get("LIP_FPS", "50"))
TTS_BACKEND = os.environ.get("TTS_BACKEND", "auto").lower()
EDGE_VOICE = os.environ.get("EDGE_VOICE", "en-US-GuyNeural")

try:
    from TTS.api import TTS

    HAS_COQUI = True
except ImportError:
    HAS_COQUI = False

try:
    import edge_tts

    HAS_EDGE = True
except ImportError:
    HAS_EDGE = False

try:
    from gtts import gTTS

    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

TTS_ENGINE = None
ACTIVE_BACKEND = None
TTS_STARTUP_HINT = None


class TextRequest(BaseModel):
    text: str


class VoiceAudioRequest(BaseModel):
    audio_base64: str
    mime: str = "audio/webm"


class VoiceIdentifyRequest(BaseModel):
    audio_base64: str
    mime: str = "audio/webm"
    profiles: list = []


def _cuda_available():
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _compute_lip_frames(wav: np.ndarray, sample_rate: int, fps: int = LIP_FPS) -> tuple:
    """GPU/CPU: jaw envelope from the real waveform (50 fps) for browser lip sync."""
    wav = np.asarray(wav, dtype=np.float32).flatten()
    if wav.size == 0:
        return [], 0
    peak = float(np.max(np.abs(wav)))
    norm = wav / max(peak, 1e-6)
    samples_per_frame = max(1, int(sample_rate / max(fps, 20)))
    n_frames = int(np.ceil(wav.size / samples_per_frame))
    raw = []
    for i in range(n_frames):
        chunk = norm[i * samples_per_frame : (i + 1) * samples_per_frame]
        if chunk.size < 4:
            raw.append(0.05)
            continue
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        # Human jaw: soft closure, wider vowels
        o = min(0.99, max(0.04, (rms * 1.5) ** 0.48))
        raw.append(o)
    smoothed = raw[:]
    for i in range(1, len(smoothed) - 1):
        smoothed[i] = raw[i - 1] * 0.12 + raw[i] * 0.76 + raw[i + 1] * 0.12
    frame_ms = 1000.0 / fps
    frames = [
        {"t": int(round(i * frame_ms)), "o": round(float(smoothed[i]), 3)}
        for i in range(len(smoothed))
    ]
    duration_ms = int(1000.0 * wav.size / sample_rate)
    return frames, duration_ms


def _audio_payload(wav: np.ndarray, sample_rate: int, backend: str, peak=None) -> dict:
    from talking_avatar import attach_talking_video, status as avatar_status

    wav = np.asarray(wav, dtype=np.float32).flatten()
    wav = np.clip(wav, -1.0, 1.0)
    lip_frames, duration_ms = _compute_lip_frames(wav, sample_rate)
    pcm = (wav * 32767).astype(np.int16)
    payload = {
        "audio_base64": base64.b64encode(pcm.tobytes()).decode("utf-8"),
        "sample_rate": sample_rate,
        "backend": backend,
        "duration_ms": duration_ms,
        "lip_frames": lip_frames,
        "talking_avatar": avatar_status(),
        "avatar_mode": avatar_status().get("mode", "off"),
    }
    if peak is not None:
        payload["peak"] = peak
    return attach_talking_video(payload, wav, sample_rate)


def _mp3_to_pcm_response(mp3_path: str, backend: str) -> dict:
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, "speech.raw")
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                mp3_path,
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-f",
                "s16le",
                raw_path,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {proc.stderr or proc.stdout}")

        pcm = np.fromfile(raw_path, dtype=np.int16)
        if pcm.size < 200:
            raise RuntimeError("decoded audio too short")
        peak = int(np.max(np.abs(pcm)))
        if peak < 32:
            raise RuntimeError(f"decoded audio is silent (peak={peak})")

        floats = pcm.astype(np.float32) / 32768.0
        return _audio_payload(floats, SAMPLE_RATE, backend, peak=peak)


def _synthesize_coqui(text: str) -> dict:
    if not TTS_ENGINE:
        raise RuntimeError("Coqui TTS not loaded")
    wav = TTS_ENGINE.tts(text=text)
    backend = "coqui-gpu" if USE_GPU and _cuda_available() else "coqui-cpu"
    return _audio_payload(np.asarray(wav, dtype=np.float32), SAMPLE_RATE, backend)


async def _synthesize_edge_async(text: str) -> dict:
    import tempfile

    if not HAS_EDGE:
        raise RuntimeError("edge-tts not installed")

    communicate = edge_tts.Communicate(text, EDGE_VOICE)
    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = os.path.join(tmp, "speech.mp3")
        await communicate.save(mp3_path)
        mp3_size = os.path.getsize(mp3_path) if os.path.exists(mp3_path) else 0
        if mp3_size < 128:
            raise RuntimeError(f"edge-tts mp3 too small ({mp3_size} bytes)")
        print(f"edge-tts mp3 size={mp3_size} bytes")
        return _mp3_to_pcm_response(mp3_path, "edge-tts")


def _synthesize_gtts(text: str) -> dict:
    import tempfile

    if not HAS_GTTS:
        raise RuntimeError("gTTS not installed (pip install gTTS)")
    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = os.path.join(tmp, "speech.mp3")
        gTTS(text=text, lang="en").save(mp3_path)
        mp3_size = os.path.getsize(mp3_path)
        print(f"gTTS mp3 size={mp3_size} bytes")
        return _mp3_to_pcm_response(mp3_path, "gtts")


def _synthesize_edge(text: str) -> dict:
    try:
        return asyncio.run(_synthesize_edge_async(text))
    except Exception as edge_err:
        print(f"edge-tts failed: {edge_err}")
        if HAS_GTTS:
            print("Trying gTTS fallback...")
            return _synthesize_gtts(text)
        raise


def _pick_backend():
    global ACTIVE_BACKEND
    if TTS_BACKEND == "coqui":
        ACTIVE_BACKEND = "coqui" if HAS_COQUI else None
    elif TTS_BACKEND == "edge":
        ACTIVE_BACKEND = "edge" if HAS_EDGE else None
    else:
        if HAS_COQUI:
            ACTIVE_BACKEND = "coqui"
        elif HAS_EDGE:
            ACTIVE_BACKEND = "edge"
        else:
            ACTIVE_BACKEND = None
    return ACTIVE_BACKEND


@asynccontextmanager
async def lifespan(app: FastAPI):
    global TTS_ENGINE, ACTIVE_BACKEND, TTS_STARTUP_HINT
    backend = _pick_backend()
    if backend == "coqui":
        try:
            print(f"Loading Coqui TTS {MODEL_NAME} gpu={USE_GPU}")
            TTS_ENGINE = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=USE_GPU)
            print("Coqui model loaded.")
            TTS_STARTUP_HINT = None
        except Exception as exc:
            print(f"Coqui load failed: {exc}")
            if HAS_EDGE:
                print("Falling back to edge-tts.")
                ACTIVE_BACKEND = "edge"
                TTS_ENGINE = "edge"
                TTS_STARTUP_HINT = f"Coqui failed ({exc}); using edge-tts."
            else:
                ACTIVE_BACKEND = None
                TTS_ENGINE = None
                TTS_STARTUP_HINT = (
                    f"Coqui failed ({exc}) and edge-tts is not installed. "
                    "Run: bash setup-kaggle.sh"
                )
    elif backend == "edge":
        print(f"Using edge-tts voice={EDGE_VOICE} (Python 3.12 / no Coqui).")
        TTS_ENGINE = "edge"
        TTS_STARTUP_HINT = None
    else:
        TTS_STARTUP_HINT = (
            "No TTS packages in this Python. "
            "cd server && bash setup-kaggle.sh && bash run-model-server.sh"
        )
        print(TTS_STARTUP_HINT)
        print(
            "  Py3.10/11: requirements-gpu.txt (Coqui)\n"
            "  Py3.12: requirements-kaggle.txt (edge-tts)"
        )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return JSONResponse(
        {
            "message": "TTS model server is running",
            "health_url": "/health",
            "backend": ACTIVE_BACKEND,
        }
    )


@app.get("/health")
def health():
    ready = TTS_ENGINE is not None
    model = MODEL_NAME if ACTIVE_BACKEND == "coqui" else EDGE_VOICE
    try:
        from talking_avatar import status as avatar_status

        avatar = avatar_status()
    except ImportError:
        avatar = {"enabled": False, "ready": False, "mode": "off"}

    try:
        from voice_id import status as voice_status

        voice = voice_status()
    except ImportError:
        voice = {"ready": False}

    return {
        "status": "ok" if ready else "loading",
        "gpu": _cuda_available(),
        "model": model,
        "ready": ready,
        "backend": ACTIVE_BACKEND,
        "has_coqui": HAS_COQUI,
        "has_edge": HAS_EDGE,
        "hint": TTS_STARTUP_HINT,
        "talking_avatar": avatar,
        "voice_id": voice,
    }


@app.post("/synthesize")
def synthesize(req: TextRequest):
    if not TTS_ENGINE:
        raise HTTPException(status_code=503, detail="TTS not ready")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    try:
        if ACTIVE_BACKEND == "coqui":
            return _synthesize_coqui(text)
        if ACTIVE_BACKEND == "edge":
            return _synthesize_edge(text)
        raise HTTPException(status_code=503, detail="No TTS backend active")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/voice/enroll")
def voice_enroll(req: VoiceAudioRequest):
    try:
        from voice_id import embed_from_base64, status as voice_status

        st = voice_status()
        if not st.get("ready"):
            raise HTTPException(
                status_code=503,
                detail=st.get("error") or "Voice ID not ready. Run: bash setup-voice-id.sh",
            )
        return embed_from_base64(req.audio_base64, req.mime)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/voice/identify")
def voice_identify(req: VoiceIdentifyRequest):
    try:
        from voice_id import identify_from_base64, status as voice_status

        st = voice_status()
        if not st.get("ready"):
            raise HTTPException(
                status_code=503,
                detail=st.get("error") or "Voice ID not ready. Run: bash setup-voice-id.sh",
            )
        return identify_from_base64(req.audio_base64, req.profiles or [], req.mime)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/assistant")
def assistant(req: TextRequest):
    text = (req.text or "").strip()
    if not text:
        return {"reply": "How can I help, boss?"}
    return {"reply": f"Yes boss! I heard: {text}"}


if __name__ == "__main__":
    import uvicorn

    # 0.0.0.0 = listen on all interfaces (Kaggle, localhost, dev tunnel). Prefer this.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    uvicorn.run(app, host=host, port=port)
