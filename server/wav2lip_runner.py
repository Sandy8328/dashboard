"""
Wav2Lip talking-head video: crop Mr. Brain face → lip-sync → composite back on portrait.
Avoids full-frame GAN replace (which morphs stylized art into a realistic "old woman" face).
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
WAV2LIP_DIR = os.environ.get("WAV2LIP_DIR", os.path.join(os.path.dirname(__file__), "Wav2Lip"))
CHECKPOINT = os.environ.get(
    "WAV2LIP_CHECKPOINT",
    os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth"),
)
FACE_PATH = os.environ.get(
    "AVATAR_FACE_PATH",
    os.path.join(ROOT, "public", "assets", "mr-brain-avatar.png"),
)
INFERENCE = os.path.join(WAV2LIP_DIR, "inference.py")
MIN_CHECKPOINT_BYTES = int(os.environ.get("WAV2LIP_MIN_CKPT_BYTES", str(350 * 1024 * 1024)))
# Face crop on portrait (y1, y2, x1, x2) as fractions — lower face of Mr. Brain only
FACE_CROP = os.environ.get("WAV2LIP_FACE_CROP", "0.42,0.96,0.20,0.80")
USE_COMPOSITE = os.environ.get("WAV2LIP_COMPOSITE", "1").lower() in ("1", "true", "yes")


def _checkpoint_valid() -> bool:
    if not os.path.isfile(CHECKPOINT):
        return False
    return os.path.getsize(CHECKPOINT) >= MIN_CHECKPOINT_BYTES


def enabled() -> bool:
    try:
        from talking_avatar import avatar_mode

        return avatar_mode() == "wav2lip"
    except ImportError:
        return os.environ.get("USE_TALKING_AVATAR", "0").lower() in ("1", "true", "yes")


def is_ready() -> bool:
    return (
        enabled()
        and os.path.isdir(WAV2LIP_DIR)
        and os.path.isfile(INFERENCE)
        and _checkpoint_valid()
        and os.path.isfile(FACE_PATH)
    )


def status() -> dict:
    ckpt_bytes = os.path.getsize(CHECKPOINT) if os.path.isfile(CHECKPOINT) else 0
    return {
        "enabled": enabled(),
        "ready": is_ready(),
        "face": FACE_PATH,
        "checkpoint": CHECKPOINT,
        "checkpoint_bytes": ckpt_bytes,
        "checkpoint_ok": _checkpoint_valid(),
        "composite": USE_COMPOSITE,
        "face_crop": FACE_CROP,
        "wav2lip_dir": WAV2LIP_DIR,
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


def _ensure_wav2lip_patched() -> None:
    try:
        from patch_wav2lip_librosa import patch_audio_py

        patch_audio_py(WAV2LIP_DIR)
    except Exception as exc:
        print("Wav2Lip librosa patch skipped:", exc)


def _parse_crop() -> tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in FACE_CROP.split(",")]
    if len(parts) != 4:
        raise ValueError("WAV2LIP_FACE_CROP must be y1,y2,x1,x2 fractions")
    return parts[0], parts[1], parts[2], parts[3]


def _crop_face_bgr(portrait_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    import cv2

    y1r, y2r, x1r, x2r = _parse_crop()
    h, w = portrait_bgr.shape[:2]
    y1, y2 = int(h * y1r), int(h * y2r)
    x1, x2 = int(w * x1r), int(w * x2r)
    y2 = min(h, max(y1 + 32, y2))
    x2 = min(w, max(x1 + 32, x2))
    crop = portrait_bgr[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2, y2)


def _run_wav2lip(
    face_input: str,
    audio_path: str,
    out_path: str,
    box: list[int] | None,
    timeout: int,
) -> tuple[bool, str]:
    pads = os.environ.get("WAV2LIP_PADS", "0 8 0 0")
    resize = os.environ.get("WAV2LIP_RESIZE", "1")
    py = sys.executable
    cmd = [
        py,
        INFERENCE,
        "--checkpoint_path",
        CHECKPOINT,
        "--face",
        face_input,
        "--audio",
        audio_path,
        "--outfile",
        out_path,
        "--pads",
        *pads.split(),
        "--resize_factor",
        str(resize),
        "--nosmooth",
    ]
    if box is not None:
        cmd.extend(["--box", *[str(v) for v in box]])
    print("Wav2Lip:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=WAV2LIP_DIR, capture_output=True, text=True, timeout=timeout)
    err = proc.stderr or proc.stdout or ""
    if proc.returncode != 0:
        return False, err
    if not os.path.isfile(out_path) or os.path.getsize(out_path) < 256:
        return False, "output missing or too small"
    return True, ""


def _composite_on_portrait(
    portrait_path: str,
    lip_video_path: str,
    box: tuple[int, int, int, int],
    audio_path: str,
    out_path: str,
) -> bool:
    import cv2

    base = cv2.imread(portrait_path)
    if base is None:
        print("composite: cannot read portrait", portrait_path)
        return False
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1

    cap = cv2.VideoCapture(lip_video_path)
    if not cap.isOpened():
        print("composite: cannot open lip video")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0

    temp_avi = lip_video_path + ".composite.avi"
    h, w = base.shape[:2]
    writer = cv2.VideoWriter(
        temp_avi, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        cap.release()
        print("composite: VideoWriter failed")
        return False

    n = 0
    while True:
        ok, lip_frame = cap.read()
        if not ok:
            break
        canvas = base.copy()
        lip_resized = cv2.resize(lip_frame, (bw, bh), interpolation=cv2.INTER_LINEAR)
        orig = base[y1:y2, x1:x2]
        # Blend AI lips with original pixels so Mr. Brain style stays visible
        blend = float(os.environ.get("WAV2LIP_BLEND", "0.42"))
        blend = max(0.15, min(0.65, blend))
        merged = cv2.addWeighted(lip_resized, blend, orig, 1.0 - blend, 0)
        mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.ellipse(mask, (bw // 2, bh // 2), (bw // 2 - 4, bh // 2 - 4), 0, 0, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        center = (x1 + bw // 2, y1 + bh // 2)
        try:
            canvas = cv2.seamlessClone(merged, canvas, mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            canvas[y1:y2, x1:x2] = merged
        writer.write(canvas)
        n += 1
    cap.release()
    writer.release()
    if n < 1:
        print("composite: no frames")
        return False

    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            temp_avi,
            "-i",
            audio_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            out_path,
        ],
        capture_output=True,
        text=True,
    )
    try:
        os.remove(temp_avi)
    except OSError:
        pass
    if proc.returncode != 0:
        print("composite ffmpeg failed:", proc.stderr or proc.stdout)
        return False
    return os.path.isfile(out_path) and os.path.getsize(out_path) > 256


def generate_video_base64(wav: np.ndarray, sample_rate: int) -> str | None:
    if not is_ready():
        print("Wav2Lip not ready:", status())
        return None

    _ensure_wav2lip_patched()

    wav = np.asarray(wav, dtype=np.float32).flatten()
    if wav.size < sample_rate // 4:
        return None

    timeout = int(os.environ.get("WAV2LIP_TIMEOUT", "300"))

    try:
        import cv2
    except ImportError:
        print("Wav2Lip composite needs opencv (pip install opencv-python-headless)")
        return None

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "speech.wav")
        lip_path = os.path.join(tmp, "lips.mp4")
        final_path = os.path.join(tmp, "avatar.mp4")
        crop_path = os.path.join(tmp, "face_crop.png")
        _write_wav(audio_path, wav, sample_rate)

        portrait = cv2.imread(FACE_PATH)
        if portrait is None:
            print("Wav2Lip: cannot read", FACE_PATH)
            return None

        if USE_COMPOSITE:
            crop, box = _crop_face_bgr(portrait)
            cv2.imwrite(crop_path, crop)
            ch, cw = crop.shape[:2]
            # Fixed box on crop = skip bad face detector on stylized art
            ok, err = _run_wav2lip(crop_path, audio_path, lip_path, [0, ch, 0, cw], timeout)
            if not ok:
                print("Wav2Lip failed:", err)
                return None
            if not _composite_on_portrait(FACE_PATH, lip_path, box, audio_path, final_path):
                print("Wav2Lip composite failed")
                return None
            out_file = final_path
            print(f"Wav2Lip composite ok, box={box}")
        else:
            ok, err = _run_wav2lip(FACE_PATH, audio_path, final_path, None, timeout)
            if not ok:
                print("Wav2Lip failed:", err)
                return None
            out_file = final_path

        with open(out_file, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def attach_talking_video(payload: dict, wav: np.ndarray, sample_rate: int) -> dict:
    st = status()
    payload["talking_avatar"] = st
    if not enabled():
        payload["wav2lip_note"] = "Set USE_TALKING_AVATAR=1 or wav2lip on Kaggle Python server"
        return payload
    if not st["ready"]:
        payload["wav2lip_note"] = "Run on Kaggle: cd server && bash setup-wav2lip.sh"
        print("Wav2Lip not ready:", st)
        return payload
    try:
        video_b64 = generate_video_base64(wav, sample_rate)
        if video_b64:
            payload["video_base64"] = video_b64
            payload["video_mime"] = "video/mp4"
            payload["avatar_backend"] = "wav2lip-composite" if USE_COMPOSITE else "wav2lip-gan"
            print(f"Wav2Lip ok, mp4 b64 chars={len(video_b64)}")
        else:
            payload["wav2lip_note"] = "Wav2Lip inference failed — check Python terminal logs"
    except subprocess.TimeoutExpired:
        payload["wav2lip_note"] = "Wav2Lip timed out (increase WAV2LIP_TIMEOUT)"
        print("Wav2Lip timed out")
    except Exception as exc:
        payload["wav2lip_note"] = f"Wav2Lip error: {exc}"
        print(f"Wav2Lip error: {exc}")
    return payload
