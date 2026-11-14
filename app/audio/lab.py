"""The three things the audio page does: make a marked clip, check a clip, and
test which edits each kind of mark survives."""
from __future__ import annotations

import base64

import numpy as np

from . import codec, container, mark

MAKE_SECONDS = 30          # clips are cut to this before marking, so the page stays quick


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def make(x: np.ndarray, label: bool, mark_on: bool, key: str, payload: int, strength: str) -> dict:
    """Mark a clip. Returns the before and after sound, the mark on its own
    (amplified until it can be heard), and a picture of each."""
    x = x[:MAKE_SECONDS * codec.RATE]
    push = mark.plan(x, payload, key, strength) if mark_on else np.zeros(mark.CELLS)
    y = mark.embed(x, payload, key, strength) if mark_on else x
    wav = codec.wav_bytes(y)
    if label:
        wav = container.add_label(wav)

    added = y - x * (np.dot(y, x) / np.dot(x, x))                 # what the filter added, loudness matched
    below = float(10 * np.log10((x ** 2).sum() / max((added ** 2).sum(), 1e-20))) if mark_on else None
    gain = min(0.5 / max(np.abs(added).max(), 1e-9), 10 ** (40 / 20)) if mark_on else 1.0
    pic_before, top = codec.spectrogram(x)
    return {"before": _b64(codec.wav_bytes(x)), "after": _b64(wav), "mark": _b64(codec.wav_bytes(added * gain)),
            "pic_before": pic_before, "pic_after": codec.spectrogram(y, top)[0],
            "pic_mark": codec.spectrogram(added * gain, top)[0],
            "mark_gain_db": round(float(20 * np.log10(gain)), 1), "mark_below_db": None if below is None else round(below, 1),
            "bands_pushed": round(float((push != 0).mean()), 2), "mean_push_db": round(float(np.abs(push).mean()), 2),
            "largest_push_db": round(float(np.abs(push).max()), 1), "bands": mark.CELLS,
            "seconds": round(len(x) / codec.RATE, 1), "bytes": len(wav), "strength": strength,
            "marks": {"label": label, "mark": mark_on}, "payload_hex": f"{payload:04X}" if mark_on else None}


def check(data: bytes, key: str) -> dict:
    """Everything this project can read from one audio file."""
    a = codec.decode(data)
    x = a["samples"]
    return {"file": {"seconds": round(len(x) / codec.RATE, 1), "codec": a.get("codec"), "container": a.get("container"),
                     "source_rate": a.get("source_rate"), "channels": a.get("channels"), "cut_short": a["cut_short"],
                     "bytes": len(data)},
            "labels": container.labels(data), "mark": mark.detect(x, key),
            "picture": codec.spectrogram(x)[0], "picture_max_hz": 12000, "key": key}


def _noise(x: np.ndarray) -> np.ndarray:
    return x + np.random.default_rng(0).normal(0, x.std() / 10, len(x))        # noise 20 dB below the sound


# name -> ("codec", encoder, bitrate, format) | ("filter", ffmpeg filter) | ("samples", function)
EDITS = {
    "MP3, 128 kbps": ("codec", "libmp3lame", "128k", "mp3"),
    "MP3, 64 kbps": ("codec", "libmp3lame", "64k", "mp3"),
    "AAC, 96 kbps": ("codec", "aac", "96k", "adts"),
    "Opus, 24 kbps (voice chat)": ("codec", "libopus", "24k", "ogg"),
    "Phone line (8 kHz sample rate)": ("filter", "aresample=8000,aresample=44100"),
    "Background noise added": ("samples", _noise),
    "Volume down to a quarter": ("samples", lambda x: x * 0.25),
    "Trim to the last third": ("samples", lambda x: x[-len(x) // 3:]),
    "First 0.37 s cut off": ("samples", lambda x: x[int(0.37 * codec.RATE):]),
    "Tempo up 5%, same pitch": ("filter", "atempo=1.05"),
    "Speed up 5% (pitch rises too)": ("filter", f"asetrate={codec.RATE}*1.05,aresample={codec.RATE}"),
    "Pitch up 1%, same tempo": ("filter", f"asetrate={codec.RATE}*1.01,aresample={codec.RATE},atempo=1.01"),
    "Echo of a small room": ("filter", "aecho=0.8:0.7:40:0.3"),
}


def survive(data: bytes, key: str) -> dict:
    """Apply each edit, write the result as a new file, and re-run both checks
    on that file."""
    x = codec.decode(data)["samples"]

    def row(name: str, file: bytes, samples: np.ndarray, base_id: str | None) -> dict:
        d = mark.detect(samples, key)
        return {"edit": name, "label": container.labels(file)["verdict"] == "detected", "z": round(d["z"], 1),
                "found": d["verdict"] == "detected", "id": d.get("payload_hex"),
                "id_ok": bool(base_id) and d.get("payload_hex") == base_id}

    first = row("No edit", data, x, None)
    first["id_ok"] = first["found"]
    rows = [first]
    for name, (how, *args) in EDITS.items():
        if how == "codec":
            file = codec.lossy(x, *args)
            y = codec.decode(file)["samples"]
        else:
            y = codec.effect(x, args[0]) if how == "filter" else args[0](x)
            file = codec.wav_bytes(y)
        rows.append(row(name, file, y, first["id"]))
    return {"rows": rows, "threshold": mark.DETECT_Z}
