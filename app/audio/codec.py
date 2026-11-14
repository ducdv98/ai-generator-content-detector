"""Audio files to samples and back.

Everything in this step works on one canonical form: mono, 44,100 samples per
second, floating point in -1..1. Any file is brought to that form first, the
same way step two brought every image to a 512 x 512 grid.

ffmpeg does the decoding, the lossy encoding (MP3, AAC, Opus) and the effects
used by the edit tests. Without ffmpeg only plain 16-bit WAV files can be read.
"""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
from PIL import Image

RATE = 44100
MAX_SECONDS = 120          # longest stretch of a file that is read


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def _run(args: list[str], data: bytes) -> bytes:
    exe = ffmpeg_path()
    if exe is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH; it is needed for this step.")
    done = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", *args], input=data, capture_output=True)
    if done.returncode != 0:
        raise ValueError("ffmpeg could not process the audio: " + done.stderr.decode("utf-8", "replace")[-300:])
    return done.stdout


_RAW = ["-f", "f32le", "-ar", str(RATE), "-ac", "1"]


def _info(path: str) -> dict:
    exe = shutil.which("ffprobe")
    if exe is None:
        return {}
    done = subprocess.run([exe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                           "stream=codec_name,sample_rate,channels:format=format_name", "-of", "json", path],
                          capture_output=True, text=True)
    try:
        j = json.loads(done.stdout)
        st = (j.get("streams") or [{}])[0]
        return {"codec": st.get("codec_name"), "source_rate": int(st.get("sample_rate") or 0),
                "channels": st.get("channels"), "container": (j.get("format") or {}).get("format_name")}
    except Exception:
        return {}


def decode(data: bytes, max_seconds: float = MAX_SECONDS) -> dict:
    """Read an audio file (or the sound track of a video file) as canonical samples."""
    if ffmpeg_path() is None:
        return _decode_wav(data, max_seconds)
    fd, path = tempfile.mkstemp(suffix=".aud")
    os.close(fd)
    try:
        with open(path, "wb") as fh:
            fh.write(data)
        info = _info(path)
        raw = _run(["-i", path, "-vn", "-t", str(max_seconds + 0.5), *_RAW, "-"], b"")
    finally:
        os.unlink(path)
    x = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    if x.size < RATE // 10:
        raise ValueError("No sound could be read from this file.")
    cut = x.size > max_seconds * RATE
    return {"samples": x[:int(max_seconds * RATE)], "cut_short": cut, **info}


def _decode_wav(data: bytes, max_seconds: float) -> dict:
    try:
        with wave.open(io.BytesIO(data)) as w:
            n, ch, rate, width = w.getnframes(), w.getnchannels(), w.getframerate(), w.getsampwidth()
            raw = w.readframes(min(n, int(max_seconds * rate)))
    except Exception:
        raise ValueError("Without ffmpeg only 16-bit WAV files can be read.")
    if width != 2:
        raise ValueError("Without ffmpeg only 16-bit WAV files can be read.")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64).reshape(-1, ch).mean(axis=1) / 32768.0
    if rate != RATE:
        x = np.interp(np.arange(0, len(x), rate / RATE), np.arange(len(x)), x)
    return {"samples": x, "cut_short": n > max_seconds * rate, "codec": "pcm_s16le", "source_rate": rate,
            "channels": ch, "container": "wav"}


def wav_bytes(x: np.ndarray) -> bytes:
    """Canonical samples as a 16-bit mono WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def lossy(x: np.ndarray, codec: str, bitrate: str, fmt: str) -> bytes:
    """Compress samples with a lossy codec; returns the compressed file."""
    return _run([*_RAW, "-i", "-", "-c:a", codec, "-b:a", bitrate, "-f", fmt, "-"], x.astype("<f4").tobytes())


def effect(x: np.ndarray, audio_filter: str) -> np.ndarray:
    """Run samples through an ffmpeg audio filter chain."""
    raw = _run([*_RAW, "-i", "-", "-af", audio_filter, *_RAW, "-"], x.astype("<f4").tobytes())
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


# ------------------------------------------------------------- pictures ----

_DARK, _LIGHT = np.array([13, 27, 42.0]), np.array([205, 226, 251.0])     # one hue, dark to light


def spectrogram(x: np.ndarray, top_db: float | None = None, width: int = 720, height: int = 150,
                max_hz: float = 12000.0) -> tuple[str, float]:
    """A picture of the sound: time left to right, pitch bottom to top,
    brightness = loudness over a 70 dB range. Returns (PNG as base64, the level
    shown as brightest), so two pictures can share one scale."""
    n, hop = 1024, 512
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    db = 10 * np.log10(power[:, :int(max_hz * n / RATE)] + 1e-12).T[::-1]      # rows: high pitch first
    if top_db is None:
        top_db = float(np.percentile(db, 99.9))
    level = np.clip((db - (top_db - 70)) / 70, 0, 1)[..., None]
    rgb = (_DARK + (_LIGHT - _DARK) * level).astype(np.uint8)
    img = Image.fromarray(rgb).resize((width, height), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode(), top_db
