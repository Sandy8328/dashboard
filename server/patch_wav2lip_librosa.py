"""Patch Wav2Lip audio.py for librosa >= 0.10 (mel filter API change)."""
from __future__ import annotations

import os
import sys


def patch_audio_py(wav2lip_dir: str) -> bool:
    path = os.path.join(wav2lip_dir, "audio.py")
    if not os.path.isfile(path):
        print("patch_wav2lip: audio.py not found at", path)
        return False

    text = open(path, encoding="utf-8").read()
    original = text

    if "librosa.filters.mel(sr=" not in text:
        text = text.replace(
            "return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,",
            "return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,",
        )

    text = text.replace("librosa.core.load", "librosa.load")

    if text == original:
        if "librosa.filters.mel(sr=" in text:
            print("patch_wav2lip: already patched")
            return True
        print("patch_wav2lip: no changes applied")
        return False

    open(path, "w", encoding="utf-8").write(text)
    print("patch_wav2lip: patched audio.py for librosa 0.10+")
    return True


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "Wav2Lip")
    ok = patch_audio_py(d)
    sys.exit(0 if ok else 1)
