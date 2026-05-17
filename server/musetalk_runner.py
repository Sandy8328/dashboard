"""
MuseTalk 1.5 lip-sync video for Mr. Brain portrait (USE_TALKING_AVATAR=musetalk or 2).
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MUSETALK_DIR = os.environ.get("MUSETALK_DIR", os.path.join(os.path.dirname(__file__), "MuseTalk"))
FACE_PATH = os.environ.get(
    "AVATAR_FACE_PATH",
    os.path.join(ROOT, "public", "assets", "mr-brain-avatar.png"),
)
MUSETALK_VERSION = os.environ.get("MUSETALK_VERSION", "v15").strip().lower()
UNET_MODEL = os.environ.get(
    "MUSETALK_UNET_PATH",
    os.path.join(MUSETALK_DIR, "models", "musetalkV15", "unet.pth"),
)
UNET_CONFIG = os.environ.get(
    "MUSETALK_UNET_CONFIG",
    os.path.join(MUSETALK_DIR, "models", "musetalkV15", "musetalk.json"),
)
WHISPER_DIR = os.environ.get(
    "MUSETALK_WHISPER_DIR",
    os.path.join(MUSETALK_DIR, "models", "whisper"),
)
INFERENCE = os.path.join(MUSETALK_DIR, "scripts", "inference.py")
MIN_UNET_BYTES = int(os.environ.get("MUSETALK_MIN_UNET_BYTES", str(300 * 1024 * 1024)))


def _mode_enabled() -> bool:
    try:
        from talking_avatar import avatar_mode

        return avatar_mode() == "musetalk"
    except ImportError:
        return os.environ.get("USE_TALKING_AVATAR", "0").strip().lower() in (
            "2",
            "musetalk",
            "muse",
            "mt",
        )


def _unet_ok() -> bool:
    return os.path.isfile(UNET_MODEL) and os.path.getsize(UNET_MODEL) >= MIN_UNET_BYTES


def enabled() -> bool:
    return _mode_enabled()


def is_ready() -> bool:
    return (
        enabled()
        and os.path.isdir(MUSETALK_DIR)
        and os.path.isfile(INFERENCE)
        and _unet_ok()
        and os.path.isfile(UNET_CONFIG)
        and os.path.isfile(FACE_PATH)
    )


def status() -> dict:
    unet_bytes = os.path.getsize(UNET_MODEL) if os.path.isfile(UNET_MODEL) else 0
    return {
        "enabled": enabled(),
        "ready": is_ready(),
        "backend": "musetalk",
        "version": MUSETALK_VERSION,
        "face": FACE_PATH,
        "musetalk_dir": MUSETALK_DIR,
        "unet_model": UNET_MODEL,
        "unet_bytes": unet_bytes,
        "unet_ok": _unet_ok(),
        "whisper_dir": WHISPER_DIR,
    }


def _write_wav(path: str, wav: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(np.asarray(wav, dtype=np.float32).flatten(), -1.0, 1.0) * 32767).astype(
        np.int16
    )
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def _write_inference_yaml(path: str, video_path: str, audio_path: str) -> None:
    # MuseTalk reads task_0 keys from OmegaConf yaml
    content = (
        "task_0:\n"
        f'  video_path: "{video_path}"\n'
        f'  audio_path: "{audio_path}"\n'
        '  result_name: "avatar.mp4"\n'
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _version_args() -> tuple[str, str]:
    if MUSETALK_VERSION in ("v1", "1.0", "1"):
        return (
            "v1",
            os.environ.get(
                "MUSETALK_UNET_PATH",
                os.path.join(MUSETALK_DIR, "models", "musetalk", "pytorch_model.bin"),
            ),
        )
    return "v15", UNET_MODEL


def _run_inference(
    inference_yaml: str,
    result_dir: str,
    timeout: int,
) -> tuple[bool, str, str | None]:
    version_arg, unet_path = _version_args()
    py = sys.executable
    cmd = [
        py,
        "-m",
        "scripts.inference",
        "--inference_config",
        inference_yaml,
        "--result_dir",
        result_dir,
        "--unet_model_path",
        unet_path,
        "--unet_config",
        UNET_CONFIG,
        "--whisper_dir",
        WHISPER_DIR,
        "--version",
        version_arg,
        "--fps",
        os.environ.get("MUSETALK_FPS", "25"),
        "--batch_size",
        os.environ.get("MUSETALK_BATCH_SIZE", "4"),
        "--saved_coord",
    ]
    if os.environ.get("MUSETALK_USE_FLOAT16", "1").lower() in ("1", "true", "yes"):
        cmd.append("--use_float16")
    bbox_shift = os.environ.get("MUSETALK_BBOX_SHIFT")
    if bbox_shift is not None and version_arg == "v1":
        cmd.extend(["--bbox_shift", str(bbox_shift)])

    ffmpeg_path = os.environ.get("MUSETALK_FFMPEG_PATH", "")
    if ffmpeg_path:
        cmd.extend(["--ffmpeg_path", ffmpeg_path])

    print("MuseTalk:", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=MUSETALK_DIR,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": MUSETALK_DIR},
    )
    tail = (proc.stderr or proc.stdout or "")[-4000:]
    if proc.returncode != 0:
        return False, tail, None

    out_mp4 = os.path.join(result_dir, version_arg, "avatar.mp4")
    if os.path.isfile(out_mp4) and os.path.getsize(out_mp4) > 256:
        return True, tail, out_mp4

    # Fallback: newest mp4 under version folder
    version_dir = os.path.join(result_dir, version_arg)
    if os.path.isdir(version_dir):
        candidates = [
            os.path.join(version_dir, name)
            for name in os.listdir(version_dir)
            if name.endswith(".mp4")
        ]
        candidates.sort(key=os.path.getmtime, reverse=True)
        for path in candidates:
            if os.path.getsize(path) > 256:
                return True, tail, path

    return False, tail or "output mp4 not found", None


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def generate_video_base64(wav: np.ndarray, sample_rate: int) -> str | None:
    if not is_ready():
        print("MuseTalk not ready:", status())
        return None

    wav = np.asarray(wav, dtype=np.float32).flatten()
    if wav.size < sample_rate // 4:
        return None

    timeout = int(os.environ.get("MUSETALK_TIMEOUT", "600"))

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "speech.wav")
        yaml_path = os.path.join(tmp, "infer.yaml")
        result_dir = os.path.join(tmp, "results")
        os.makedirs(result_dir, exist_ok=True)
        _write_wav(audio_path, wav, sample_rate)
        _write_inference_yaml(yaml_path, os.path.abspath(FACE_PATH), os.path.abspath(audio_path))

        try:
            ok, err, out_mp4 = _run_inference(yaml_path, result_dir, timeout)
        except subprocess.TimeoutExpired:
            print("MuseTalk timed out")
            return None
        finally:
            _clear_cuda_cache()

        if not ok or not out_mp4:
            print("MuseTalk failed:", err)
            return None

        print(f"MuseTalk ok: {out_mp4} ({os.path.getsize(out_mp4)} bytes)")
        with open(out_mp4, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def attach_talking_video(payload: dict, wav: np.ndarray, sample_rate: int) -> dict:
    st = status()
    ta = payload.get("talking_avatar")
    if isinstance(ta, dict):
        ta["musetalk"] = st
        ta["ready"] = bool(st.get("ready"))
    else:
        payload["talking_avatar"] = st
    if not enabled():
        payload["musetalk_note"] = "Set USE_TALKING_AVATAR=2 or musetalk on Kaggle Python server"
        return payload
    if not st["ready"]:
        payload["musetalk_note"] = "Run on Kaggle: cd server && bash setup-musetalk.sh"
        print("MuseTalk not ready:", st)
        return payload
    try:
        video_b64 = generate_video_base64(wav, sample_rate)
        if video_b64:
            payload["video_base64"] = video_b64
            payload["video_mime"] = "video/mp4"
            payload["avatar_backend"] = f"musetalk-{MUSETALK_VERSION}"
            print(f"MuseTalk ok, mp4 b64 chars={len(video_b64)}")
        else:
            payload["musetalk_note"] = "MuseTalk inference failed — check Python terminal logs"
    except subprocess.TimeoutExpired:
        payload["musetalk_note"] = "MuseTalk timed out (increase MUSETALK_TIMEOUT)"
        print("MuseTalk timed out")
    except Exception as exc:
        payload["musetalk_note"] = f"MuseTalk error: {exc}"
        print(f"MuseTalk error: {exc}")
    return payload
