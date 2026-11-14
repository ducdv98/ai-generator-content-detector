"""What a video file is made of, and the label stored beside the frames.

An MP4 (or MOV) is a row of "boxes", each with a size and a four-letter name:

    ftyp  which flavour of MP4 this is
    moov  the index: where every frame sits in the file, timing, titles, tool names
    mdat  the compressed frames and sound themselves
    uuid  an extension box; C2PA Content Credentials and XMP metadata both live in one

As with images, every "made by AI" label that is not a pixel watermark is a box
beside the picture data. Re-encoding a video writes a new file from the frames
alone, so the label does not come along.
"""
from __future__ import annotations

import io
import json
import re
import struct

from ..detectors import provenance
from ..image.lab import XMP_AI

PIXELS, LABEL, OTHER = "pixels", "label", "other"
XMP_UUID = bytes.fromhex("BE7ACFCB97A942E89C71999491E3AFAC")      # XMP packet in an MP4 (Adobe XMP spec, part 3)
C2PA_UUID = bytes.fromhex("D8FEC3D61B0E483C92975828877EC481")     # C2PA manifest store in an MP4

_NAMES = {
    "ftyp": ("File type", OTHER), "moov": ("Index of frames, timing and tags", OTHER),
    "mdat": ("Compressed frames and sound", PIXELS), "moof": ("Index of frames, timing and tags", OTHER),
    "free": ("Padding", OTHER), "skip": ("Padding", OTHER), "wide": ("Padding", OTHER),
    "meta": ("Metadata box", LABEL), "udta": ("User data (titles, tool names)", LABEL),
}


def is_mp4(data: bytes) -> bool:
    return len(data) >= 12 and data[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"wide", b"skip")


def _boxes(data: bytes):
    pos, n = 0, len(data)
    while pos + 8 <= n:
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        head = 8
        if size == 1 and pos + 16 <= n:
            size, head = struct.unpack(">Q", data[pos + 8:pos + 16])[0], 16
        elif size == 0:
            size = n - pos
        if size < head or pos + size > n:
            yield "????", pos, n - pos, head
            return
        yield kind.decode("latin-1"), pos, size, head
        pos += size


def parts(data: bytes) -> list[dict]:
    """Top-level boxes in file order."""
    if not is_mp4(data):
        return [{"name": "Whole file (only MP4 / MOV files are broken down here)", "kind": OTHER, "bytes": len(data)}]
    out = []
    for kind, pos, size, head in _boxes(data):
        if kind == "uuid":
            ext = data[pos + head:pos + head + 16]
            name, cat = (("XMP metadata (IPTC tags live here)", LABEL) if ext == XMP_UUID else
                         ("C2PA Content Credentials", LABEL) if ext == C2PA_UUID else ("Extension box", OTHER))
        else:
            name, cat = _NAMES.get(kind, (f"Other box ({kind})", OTHER))
        out.append({"name": name, "kind": cat, "bytes": size})
    return out


def add_label(mp4: bytes) -> bytes:
    """Append the unsigned "made by AI" tag: an XMP box holding IPTC DigitalSourceType."""
    body = XMP_UUID + XMP_AI.encode("utf-8")
    return mp4 + struct.pack(">I", 8 + len(body)) + b"uuid" + body


def _tool_tag(data: bytes) -> str | None:
    """The writing tool's name, which most encoders leave in the index (the ©too tag)."""
    i = data.find(b"\xa9too")
    if i < 0:
        return None
    j = data.find(b"data", i, i + 64)
    if j < 0:
        return None
    size = struct.unpack(">I", data[j - 4:j])[0]
    return data[j + 12:j - 4 + size].decode("utf-8", "replace").strip("\x00 ") or None


def labels(data: bytes, mime: str = "video/mp4", tool: str | None = "auto", blocks: list[dict] | None = None,
           ai_tools: list[str] | None = None) -> dict:
    """Read every label this project understands from a media file. The audio
    step calls this too, passing its own file breakdown and tool tag."""
    findings = []
    ai_tools = ai_tools or provenance.AI_SOFTWARE_KEYWORDS
    try:                                              # signed Content Credentials, if any
        import c2pa
        store = json.loads(c2pa.Reader(mime, io.BytesIO(data)).json())
        active = store.get("manifests", {}).get(store.get("active_manifest", ""), {})
        types = sorted({str(d["digitalSourceType"]) for m in store.get("manifests", {}).values()
                        for d in provenance._walk(m) if "digitalSourceType" in d})
        ai = any(t.rsplit("/", 1)[-1].lower() in provenance.AI_SOURCE_TYPES for t in types)
        issuer = (active.get("signature_info") or {}).get("issuer") or "unknown signer"
        findings.append({"check": "C2PA Content Credentials", "points_to": "ai" if ai else "neutral",
                         "strength": "declared" if ai else "info",
                         "finding": f"Content Credentials (signed by {issuer}) "
                                    + ("say this file was made with generative AI." if ai
                                       else "are present but do not state an AI source."),
                         "details": {"digital_source_types": types, "validation_state": store.get("validation_state"),
                                     "claim_generator": active.get("claim_generator")}})
    except Exception:
        pass

    found = {m.decode().lower() for p in provenance._XMP_RE.findall(data)
             for m in provenance._SOURCE_TYPE_RE.findall(p)}
    for key in sorted(found):
        if key in provenance.AI_SOURCE_TYPES:
            findings.append({"check": "IPTC DigitalSourceType", "points_to": "ai", "strength": "strong",
                             "finding": f"The metadata tag DigitalSourceType = {key}: "
                                        f"{provenance.AI_SOURCE_TYPES[key]}. (An unsigned tag: reliable when "
                                        "present, but anyone can add or remove it.)",
                             "details": {"digital_source_type": key}})
    if tool == "auto":
        tool = _tool_tag(data) if is_mp4(data) else None
    if tool:
        hit = next((k for k in ai_tools if re.search(rf"\b{re.escape(k)}\b", tool.lower())), None)
        findings.append({"check": "Tool tag", "points_to": "ai" if hit else "neutral",
                         "strength": "strong" if hit else "info",
                         "finding": (f"The file names an AI generator as the tool that wrote it: {tool}." if hit else
                                     f"Written by \"{tool}\". This names the encoder, not what made the content."),
                         "details": {"tool": tool}})
    if not any(f["points_to"] == "ai" for f in findings):
        findings.append({"check": "Labels", "points_to": "neutral", "strength": "info", "details": {},
                         "finding": "No AI label found. Most shared files have none: platforms re-encode every "
                                    "upload, which drops these blocks."})
    blocks = parts(data) if blocks is None else blocks
    ai = [f for f in findings if f["points_to"] == "ai"]
    return {"verdict": "detected" if ai else "none", "headline": (ai or findings)[0]["finding"],
            "findings": findings, "parts": blocks,
            "label_bytes": sum(p["bytes"] for p in blocks if p["kind"] == LABEL), "total_bytes": len(data)}
