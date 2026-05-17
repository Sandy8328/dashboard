"""
Talking-head backend selection: off | wav2lip | musetalk (USE_TALKING_AVATAR).
"""
from __future__ import annotations

import os


def avatar_mode() -> str:
    raw = os.environ.get("USE_TALKING_AVATAR", "0").strip().lower()
    if raw in ("0", "false", "no", "off", "canvas", ""):
        return "off"
    if raw in ("2", "musetalk", "muse", "mt"):
        return "musetalk"
    if raw in ("1", "true", "yes", "wav2lip", "w2l"):
        return "wav2lip"
    return "off"


def status() -> dict:
    mode = avatar_mode()
    out = {
        "mode": mode,
        "enabled": mode != "off",
        "ready": False,
        "env": os.environ.get("USE_TALKING_AVATAR", "0"),
    }
    if mode == "musetalk":
        try:
            from musetalk_runner import status as musetalk_status

            ms = musetalk_status()
            out["musetalk"] = ms
            out["ready"] = bool(ms.get("ready"))
        except ImportError as exc:
            out["musetalk"] = {"ready": False, "error": str(exc)}
    elif mode == "wav2lip":
        try:
            from wav2lip_runner import status as wav2lip_status

            ws = wav2lip_status()
            out["wav2lip"] = ws
            out["ready"] = bool(ws.get("ready"))
        except ImportError as exc:
            out["wav2lip"] = {"ready": False, "error": str(exc)}
    return out


def attach_talking_video(payload: dict, wav, sample_rate: int) -> dict:
    mode = avatar_mode()
    st = status()
    payload["talking_avatar"] = st
    payload["avatar_mode"] = mode

    if mode == "off":
        payload["talking_avatar_note"] = (
            "Canvas lips only. Set USE_TALKING_AVATAR=2 (musetalk) or 1 (wav2lip) on Kaggle."
        )
        return payload

    if mode == "musetalk":
        from musetalk_runner import attach_talking_video as musetalk_attach

        return musetalk_attach(payload, wav, sample_rate)

    from wav2lip_runner import attach_talking_video as wav2lip_attach

    return wav2lip_attach(payload, wav, sample_rate)
