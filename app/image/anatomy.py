"""What an image file is made of.

A JPEG, PNG or WebP is a list of labelled blocks. A few hold the picture; the
rest hold notes about it. Every "this was made by AI" label that is not a pixel
watermark (C2PA Content Credentials, IPTC tags, EXIF, generator parameters)
lives in one of the note blocks. That is why such labels are exact and easy to
read, and also why a screenshot or a re-save by a chat app removes them: those
copy the picture and leave the notes behind.
"""
from __future__ import annotations

import struct

PIXELS, LABEL, OTHER = "pixels", "label", "other"


def _jpeg(data: bytes) -> list[tuple[str, str, int]]:
    out, pos, n = [("Start of file", OTHER, 2)], 2, len(data)
    c2pa = False
    while pos + 2 <= n and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker == 0xD9:
            out.append(("End of file", OTHER, 2))
            pos += 2
            break
        if pos + 4 > n:
            break
        size = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        body = data[pos + 4:pos + 2 + size]
        end = pos + 2 + size
        if marker == 0xDA:                       # image data runs to the next real marker
            while True:
                end = data.find(b"\xff", end)
                if end < 0 or end + 1 >= n:
                    end = n
                    break
                if data[end + 1] == 0x00 or 0xD0 <= data[end + 1] <= 0xD7 or data[end + 1] == 0xFF:
                    end += 1 if data[end + 1] == 0xFF else 2
                    continue
                break
            out.append(("Picture data", PIXELS, end - pos))
        elif marker == 0xE1 and body.startswith(b"Exif"):
            out.append(("EXIF (camera and software fields)", LABEL, end - pos))
        elif marker == 0xE1 and body.startswith(b"http://ns.adobe.com/xap/1.0/"):
            out.append(("XMP metadata (IPTC tags live here)", LABEL, end - pos))
        elif marker == 0xEB:
            c2pa = c2pa or b"c2pa" in body[:64]     # a manifest is split over several of these
            out.append(("C2PA Content Credentials" if c2pa else "JUMBF box", LABEL, end - pos))
        elif marker == 0xED:
            out.append(("Photoshop / IPTC block", LABEL, end - pos))
        elif marker == 0xFE:
            out.append(("Comment", LABEL, end - pos))
        elif marker == 0xE2 and body.startswith(b"ICC_PROFILE"):
            out.append(("Colour profile", OTHER, end - pos))
        elif 0xE0 <= marker <= 0xEF:
            out.append(("App header", OTHER, end - pos))
        else:                                    # quantisation / Huffman tables, frame header
            out.append(("Picture header", PIXELS, end - pos))
        pos = end
    if pos < n:
        out.append(("Data after the end marker", OTHER, n - pos))
    return out


def _png(data: bytes) -> list[tuple[str, str, int]]:
    out, pos, n = [("Start of file", OTHER, 8)], 8, len(data)
    while pos + 12 <= n:
        size = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8].decode("latin-1")
        body = data[pos + 8:pos + 8 + min(size, 80)]
        total = size + 12
        if kind in ("IHDR", "PLTE", "IDAT", "IEND", "tRNS", "acTL", "fcTL", "fdAT"):
            out.append(("Picture data" if kind in ("IDAT", "fdAT") else "Picture header", PIXELS, total))
        elif kind in ("tEXt", "zTXt", "iTXt"):
            key = body.split(b"\x00", 1)[0].decode("latin-1")
            name = "XMP metadata (IPTC tags live here)" if key == "XML:com.adobe.xmp" else f"Text note \"{key}\""
            out.append((name, LABEL, total))
        elif kind == "caBX":
            out.append(("C2PA Content Credentials", LABEL, total))
        elif kind == "eXIf":
            out.append(("EXIF (camera and software fields)", LABEL, total))
        elif kind == "iCCP":
            out.append(("Colour profile", OTHER, total))
        else:
            out.append((f"Other chunk ({kind})", OTHER, total))
        pos += total
    return out


def _webp(data: bytes) -> list[tuple[str, str, int]]:
    out, pos, n = [("Start of file", OTHER, 12)], 12, len(data)
    names = {"EXIF": ("EXIF (camera and software fields)", LABEL), "XMP ": ("XMP metadata (IPTC tags live here)", LABEL),
             "C2PA": ("C2PA Content Credentials", LABEL), "ICCP": ("Colour profile", OTHER)}
    while pos + 8 <= n:
        kind = data[pos:pos + 4].decode("latin-1")
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        total = 8 + size + (size & 1)
        name, cat = names.get(kind, ("Picture data" if kind.startswith("VP8") or kind == "ALPH"
                                     else f"Other chunk ({kind.strip()})",
                                     PIXELS if kind.startswith("VP8") or kind in ("ALPH", "ANIM", "ANMF") else OTHER))
        out.append((name, cat, total))
        pos += total
    return out


def parts(data: bytes) -> list[dict]:
    """The file's blocks in order, neighbours of the same kind merged."""
    try:
        if data[:2] == b"\xff\xd8":
            raw = _jpeg(data)
        elif data[:8] == b"\x89PNG\r\n\x1a\n":
            raw = _png(data)
        elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            raw = _webp(data)
        else:
            raw = [("Whole file (format not broken down here)", OTHER, len(data))]
    except Exception:
        raw = [("Whole file (could not be broken down)", OTHER, len(data))]
    merged: list[dict] = []
    for name, kind, size in raw:
        if merged and merged[-1]["name"] == name:
            merged[-1]["bytes"] += size
        elif merged and kind == PIXELS and merged[-1]["kind"] == PIXELS:
            merged[-1]["bytes"] += size
            merged[-1]["name"] = "Picture data"
        else:
            merged.append({"name": name, "kind": kind, "bytes": size})
    return merged
