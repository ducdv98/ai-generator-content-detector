"""What an audio file is made of, and the label stored beside the sound.

    WAV   a row of chunks: `fmt ` (sample rate, channels), `data` (the samples),
          and optional extras: `LIST` (title, software), `_PMX` (XMP metadata),
          `C2PA` (Content Credentials)
    MP3   an ID3 tag block at the front (title, software, comments, and where
          XMP or C2PA data goes), then the compressed sound frames
    M4A   the same boxes as an MP4 video (see app/video/container.py)

The pattern is the one from images and video: labels are separate blocks beside
the sound, exact when present, and left behind whenever a service decodes the
sound and writes a new file.
"""
from __future__ import annotations

import re
import struct

from ..image.lab import XMP_AI
from ..video import container as mp4

SOUND, LABEL, OTHER = "pixels", "label", "other"       # "pixels" is the shared name for "the content itself"
AI_TOOLS = ["suno", "udio", "elevenlabs", "lyria", "musicgen", "audiogen", "stable audio", "musicfx",
            "openai", "resemble", "notebooklm", "voicebox", "bark"]

_WAV = {"fmt ": ("Format (sample rate, channels)", OTHER), "data": ("The sound samples", SOUND),
        "LIST": ("Text tags (title, software)", LABEL), "_PMX": ("XMP metadata (IPTC tags live here)", LABEL),
        "C2PA": ("C2PA Content Credentials", LABEL), "bext": ("Broadcast description", LABEL),
        "id3 ": ("ID3 tags", LABEL), "ID3 ": ("ID3 tags", LABEL), "fact": ("Sample count", OTHER)}


def kind(data: bytes) -> str:
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:3] == b"ID3" or (len(data) > 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return "mp3"
    if mp4.is_mp4(data):
        return "mp4"
    return "other"


def parts(data: bytes) -> list[dict]:
    """The file's blocks in order."""
    k = kind(data)
    if k == "mp4":
        return mp4.parts(data)
    out: list[dict] = []
    if k == "wav":
        out.append({"name": "File header", "kind": OTHER, "bytes": 12})
        pos = 12
        while pos + 8 <= len(data):
            cid = data[pos:pos + 4].decode("latin-1")
            size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
            total = min(8 + size + (size & 1), len(data) - pos)
            name, cat = _WAV.get(cid, (f"Other chunk ({cid.strip()})", OTHER))
            out.append({"name": name, "kind": cat, "bytes": total})
            pos += total
    elif k == "mp3":
        pos = 0
        if data[:3] == b"ID3" and len(data) >= 10:
            size = 10 + ((data[6] & 0x7F) << 21 | (data[7] & 0x7F) << 14 | (data[8] & 0x7F) << 7 | (data[9] & 0x7F))
            out.append({"name": "ID3 tags (title, software, comments)", "kind": LABEL, "bytes": min(size, len(data))})
            pos = min(size, len(data))
        tail = 128 if data[-128:-125] == b"TAG" else 0
        out.append({"name": "Compressed sound frames", "kind": SOUND, "bytes": len(data) - pos - tail})
        if tail:
            out.append({"name": "ID3 tags (old style)", "kind": LABEL, "bytes": 128})
    else:
        out.append({"name": "Whole file (only WAV, MP3 and M4A files are broken down here)", "kind": OTHER,
                    "bytes": len(data)})
    return out


def add_label(wav: bytes) -> bytes:
    """Append the unsigned "made by AI" tag to a WAV: an XMP chunk holding IPTC DigitalSourceType."""
    body = XMP_AI.encode("utf-8")
    body += b"\x00" * (len(body) & 1)
    out = wav + b"_PMX" + struct.pack("<I", len(body)) + body
    return out[:4] + struct.pack("<I", len(out) - 8) + out[8:]


def _tool(data: bytes) -> str | None:
    """The name of the program that wrote the file, which most encoders leave in a tag."""
    k = kind(data)
    if k == "mp4":
        return mp4._tool_tag(data)
    for pattern in (rb"ISFT....([\x20-\x7e]{2,60})", rb"TSSE.{6,7}([\x20-\x7e]{2,60})"):
        m = re.search(pattern, data[:200_000] + data[-4096:], re.S)
        if m:
            return m.group(1).decode("latin-1").strip()
    return None


def labels(data: bytes) -> dict:
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "mp4": "audio/mp4"}.get(kind(data), "audio/wav")
    return mp4.labels(data, mime=mime, tool=_tool(data), blocks=parts(data), ai_tools=AI_TOOLS)
