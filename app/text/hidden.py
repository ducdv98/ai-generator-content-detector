"""Scan for characters that take no visible space.

This is the one check that works on text from any source with no key. It is
also the weakest kind of mark: Google, Anthropic and OpenAI all state that
their text watermarks add no hidden characters, and anything found here is
removed by retyping or a find-and-replace. Finding some does not show a text is
AI-written (copying from web pages, PDFs and chat apps leaves them too); it
shows the text carries something a reader cannot see.
"""
from __future__ import annotations

import unicodedata

# kind: "invisible" draws nothing; "space" looks like a normal space but is not U+0020.
_NAMED = {
    0x200B: ("Zero-width space", "invisible", ""),
    0x200C: ("Zero-width non-joiner", "invisible", "Normal in Persian, Arabic and Indic scripts."),
    0x200D: ("Zero-width joiner", "invisible", "Normal inside emoji sequences and Indic scripts."),
    0x2060: ("Word joiner", "invisible", ""),
    0xFEFF: ("Zero-width no-break space (BOM)", "invisible", "Normal as the very first character of a file."),
    0x180E: ("Mongolian vowel separator", "invisible", ""),
    0x00AD: ("Soft hyphen", "invisible", "Often left behind by PDFs and word processors."),
    0x034F: ("Combining grapheme joiner", "invisible", ""),
    0x00A0: ("No-break space", "space", "Common in ordinary typography, e.g. French punctuation."),
    0x202F: ("Narrow no-break space", "space", "Some model outputs used this in place of normal spaces."),
    0x2009: ("Thin space", "space", ""),
    0x200A: ("Hair space", "space", ""),
    0x205F: ("Medium mathematical space", "space", ""),
    0x3000: ("Ideographic space", "space", "Normal in Chinese and Japanese text."),
}
_RANGES = [
    (0x2000, 0x2008, "Fixed-width space", "space", ""),
    (0x200E, 0x200F, "Direction mark", "invisible", "Normal in right-to-left text."),
    (0x202A, 0x202E, "Direction override", "invisible", ""),
    (0x2061, 0x2064, "Invisible math operator", "invisible", ""),
    (0x2066, 0x2069, "Direction isolate", "invisible", ""),
    (0xFE00, 0xFE0F, "Variation selector", "invisible", "Normal after emoji; a run of them can hide data."),
    (0xE0100, 0xE01EF, "Variation selector (supplement)", "invisible", "Can hide arbitrary bytes."),
    (0xE0000, 0xE007F, "Tag character", "invisible", "Each one hides an ASCII character."),
]


def classify(ch: str) -> tuple[str, str, str] | None:
    """(name, kind, note) if `ch` is a character worth flagging, else None."""
    cp = ord(ch)
    if cp in _NAMED:
        return _NAMED[cp]
    for lo, hi, name, kind, note in _RANGES:
        if lo <= cp <= hi:
            return name, kind, note
    if unicodedata.category(ch) == "Cf":
        return unicodedata.name(ch, "Format character").title(), "invisible", ""
    return None


def scan(text: str) -> dict:
    found: dict[int, dict] = {}
    tag_payload = []
    for pos, ch in enumerate(text):
        info = classify(ch)
        if info is None:
            continue
        cp = ord(ch)
        if cp == 0xFEFF and pos == 0:
            continue
        entry = found.setdefault(cp, {"codepoint": f"U+{cp:04X}", "name": info[0], "kind": info[1],
                                      "note": info[2], "count": 0, "positions": []})
        entry["count"] += 1
        if len(entry["positions"]) < 20:
            entry["positions"].append(pos)
        if 0xE0020 <= cp <= 0xE007E:
            tag_payload.append(chr(cp - 0xE0000))
    items = sorted(found.values(), key=lambda e: (e["kind"] != "invisible", -e["count"]))
    return {"items": items,
            "invisible": sum(e["count"] for e in items if e["kind"] == "invisible"),
            "spaces": sum(e["count"] for e in items if e["kind"] == "space"),
            "tag_message": "".join(tag_payload) or None}
