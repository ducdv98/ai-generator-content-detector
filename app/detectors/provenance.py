"""Layer 1: what the file says about itself.

Checks, in order of trustworthiness:
  * C2PA Content Credentials (cryptographically signed manifest)
  * IPTC DigitalSourceType in XMP metadata
  * Generator-specific metadata (Stable Diffusion / ComfyUI PNG chunks, EXIF Software)
  * Camera EXIF (weak hint of a real capture; trivially forged)
"""
from __future__ import annotations

import io
import json
import re
from typing import Any

from PIL import ExifTags, Image

from ..report import AI, DECLARED, INFO, NEUTRAL, NOT_AI, STRONG, WEAK, Evidence

LAYER = "provenance"

# https://cv.iptc.org/newscodes/digitalsourcetype/
AI_SOURCE_TYPES = {
    "trainedalgorithmicmedia": "created entirely by a generative AI model",
    "compositewithtrainedalgorithmicmedia": "edited or composited using generative AI",
    "compositesynthetic": "a composite that includes synthetic elements",
}
CAPTURE_SOURCE_TYPES = {
    "digitalcapture": "captured by a digital camera",
    "computationalcapture": "captured by a camera with computational processing",
    "negativefilm": "digitised from film",
    "positivefilm": "digitised from film",
    "print": "digitised from a print",
}

AI_SOFTWARE_KEYWORDS = [
    "midjourney", "dall-e", "dall·e", "dalle", "stable diffusion", "stablediffusion",
    "firefly", "imagen", "novelai", "comfyui", "invokeai", "automatic1111", "ideogram",
    "leonardo", "bing image creator", "flux", "sora", "grok", "gpt-4o", "chatgpt",
    "openai", "runway", "fooocus", "playground",
]

# PNG text-chunk keys written by popular local generators.
PNG_GENERATOR_KEYS = {
    "parameters": "Stable Diffusion WebUI (AUTOMATIC1111 / Forge) generation parameters",
    "prompt": "ComfyUI prompt graph",
    "workflow": "ComfyUI workflow",
    "invokeai_metadata": "InvokeAI metadata",
    "sd-metadata": "InvokeAI (legacy) metadata",
    "dream": "InvokeAI (legacy) metadata",
    "fooocus_scheme": "Fooocus metadata",
}

_SOURCE_TYPE_RE = re.compile(rb"digitalsourcetype/([A-Za-z]+)", re.I)
_XMP_RE = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.S)


def analyze(data: bytes, img: Image.Image | None) -> list[Evidence]:
    out: list[Evidence] = []
    out += _check_c2pa(data, img)
    out += _check_iptc(data, img)
    out += _check_generator_metadata(img)
    out += _check_camera_exif(img)
    if not any(e.points_to != NEUTRAL for e in out):
        out.append(Evidence(LAYER, "Metadata", "No provenance metadata found. Most uploads "
                            "have none, because screenshots and re-encodes strip it."))
    return out


# --------------------------------------------------------------------------- C2PA

def _mime(img: Image.Image | None) -> str | None:
    if img is None or not img.format:
        return None
    return Image.MIME.get(img.format.upper())


def _source_type_meaning(value: str) -> tuple[str, str] | None:
    key = value.rsplit("/", 1)[-1].lower()
    if key in AI_SOURCE_TYPES:
        return AI, AI_SOURCE_TYPES[key]
    if key in CAPTURE_SOURCE_TYPES:
        return NOT_AI, CAPTURE_SOURCE_TYPES[key]
    return None


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _check_c2pa(data: bytes, img: Image.Image | None) -> list[Evidence]:
    has_marker = b"c2pa" in data and (b"jumb" in data or b"jumd" in data)
    try:
        import c2pa
    except Exception:
        if has_marker:
            return [Evidence(LAYER, "C2PA Content Credentials",
                             "The file appears to contain a C2PA manifest, but the c2pa-python "
                             "library is not installed, so it could not be read or verified.")]
        return []

    mime = _mime(img)
    if mime is None:
        return []
    try:
        reader = c2pa.Reader(mime, io.BytesIO(data))
        store = json.loads(reader.json())
    except Exception as exc:  # no manifest is the normal case
        if has_marker:
            return [Evidence(LAYER, "C2PA Content Credentials",
                             f"A C2PA manifest seems to be present but could not be read ({exc}).")]
        return []

    active = store.get("manifests", {}).get(store.get("active_manifest", ""), {})
    generator = active.get("claim_generator") or ", ".join(
        g.get("name", "") for g in active.get("claim_generator_info", []) if isinstance(g, dict))
    issuer = (active.get("signature_info") or {}).get("issuer")
    state = store.get("validation_state")  # "Trusted" | "Valid" | "Invalid"
    failures = [f.get("code") for f in ((store.get("validation_results") or {})
                                         .get("activeManifest") or {}).get("failure", [])
                if isinstance(f, dict)]
    details = {"claim_generator": generator, "signed_by": issuer,
               "validation_state": state, "validation_failures": failures}

    source_types = sorted({str(d["digitalSourceType"]) for m in store.get("manifests", {}).values()
                           for d in _walk(m) if "digitalSourceType" in d})
    details["digital_source_types"] = source_types
    signer = f"signed by {issuer}" if issuer else "signer unknown"
    trusted = state == "Trusted"
    valid = state in ("Valid", "Trusted")
    if trusted:
        caveat = ""
    elif valid:
        # Intact, but the signer was not checked against a C2PA trust list.
        caveat = " The manifest is intact; the signer's identity is not verified against a trust list."
    else:
        caveat = " Warning: the manifest failed validation, so the file was altered after signing."

    for st in source_types:
        meaning = _source_type_meaning(st)
        if meaning and meaning[0] == AI:
            return [Evidence(LAYER, "C2PA Content Credentials",
                             f"Content Credentials ({signer}) say this content was {meaning[1]}."
                             + caveat, AI, DECLARED if valid else STRONG, details)]
    if any(k in (generator or "").lower() for k in AI_SOFTWARE_KEYWORDS):
        return [Evidence(LAYER, "C2PA Content Credentials",
                         f"Content Credentials ({signer}) name an AI tool as the producer: "
                         f"{generator}." + caveat, AI, STRONG, details)]
    for st in source_types:
        meaning = _source_type_meaning(st)
        if meaning and meaning[0] == NOT_AI:
            return [Evidence(LAYER, "C2PA Content Credentials",
                             f"Content Credentials ({signer}) say this content was {meaning[1]}."
                             + caveat, NOT_AI, STRONG if trusted else WEAK, details)]
    return [Evidence(LAYER, "C2PA Content Credentials",
                     f"Content Credentials are present ({signer}, produced with "
                     f"{generator or 'an unnamed tool'}) but do not state an AI source.",
                     NEUTRAL, INFO, details)]


# --------------------------------------------------------------------------- IPTC / XMP

def _xmp_packets(data: bytes, img: Image.Image | None) -> list[bytes]:
    packets = _XMP_RE.findall(data)
    if img is not None:  # PNG can store XMP in a compressed iTXt chunk
        for key in ("XML:com.adobe.xmp", "xmp"):
            v = img.info.get(key)
            if isinstance(v, str):
                packets.append(v.encode("utf-8", "ignore"))
            elif isinstance(v, bytes):
                packets.append(v)
    return packets


def _check_iptc(data: bytes, img: Image.Image | None) -> list[Evidence]:
    found = {m.decode().lower() for p in _xmp_packets(data, img)
             for m in _SOURCE_TYPE_RE.findall(p)}
    for key in found:
        if key in AI_SOURCE_TYPES:
            return [Evidence(LAYER, "IPTC DigitalSourceType",
                             f"The metadata tag DigitalSourceType = {key}: "
                             f"{AI_SOURCE_TYPES[key]}. (An unsigned tag: reliable when present, "
                             "but anyone can add or remove it.)",
                             AI, STRONG, {"digital_source_type": key})]
    for key in found:
        if key in CAPTURE_SOURCE_TYPES:
            return [Evidence(LAYER, "IPTC DigitalSourceType",
                             f"The metadata tag DigitalSourceType = {key}: "
                             f"{CAPTURE_SOURCE_TYPES[key]}. Unsigned, so only a weak hint.",
                             NOT_AI, WEAK, {"digital_source_type": key})]
    return []


# --------------------------------------------------------------------------- generator metadata

def _exif_fields(img: Image.Image | None) -> dict[str, Any]:
    if img is None:
        return {}
    try:
        exif = img.getexif()
    except Exception:
        return {}
    fields: dict[str, Any] = {}
    for tag, value in exif.items():
        fields[ExifTags.TAGS.get(tag, str(tag))] = value
    try:
        for tag, value in exif.get_ifd(ExifTags.IFD.Exif).items():
            fields[ExifTags.TAGS.get(tag, str(tag))] = value
    except Exception:
        pass
    return fields


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        for enc in ("utf-8", "utf-16-be", "utf-16-le", "latin-1"):
            try:
                return value.decode(enc).replace("\x00", "")
            except Exception:
                continue
    return str(value)


def _check_generator_metadata(img: Image.Image | None) -> list[Evidence]:
    if img is None:
        return []
    out: list[Evidence] = []

    text_chunks = {str(k): _text(v) for k, v in (getattr(img, "text", None) or {}).items()}
    for key, value in text_chunks.items():
        label = PNG_GENERATOR_KEYS.get(key.lower())
        if key.lower() == "prompt" and not value.lstrip().startswith("{"):
            label = None  # ComfyUI stores JSON here; a plain "prompt" string proves nothing
        if label:
            out.append(Evidence(LAYER, "Generator metadata",
                                f"The PNG carries a '{key}' text chunk: {label}.",
                                AI, STRONG, {"key": key, "excerpt": value[:300]}))
            break

    exif = _exif_fields(img)
    candidates = {"Software": text_chunks.get("Software") or exif.get("Software"),
                  "Make": exif.get("Make"), "Artist": exif.get("Artist"),
                  "ImageDescription": exif.get("ImageDescription"),
                  "UserComment": exif.get("UserComment")}
    if not out:
        for field_name, raw in candidates.items():
            if raw is None:
                continue
            value = _text(raw)
            low = value.lower()
            hit = next((k for k in AI_SOFTWARE_KEYWORDS if k in low), None)
            if field_name == "UserComment" and "steps:" in low and "sampler:" in low:
                hit = "stable diffusion parameters"
            if hit:
                out.append(Evidence(LAYER, "Generator metadata",
                                    f"The {field_name} metadata field names an AI generator "
                                    f"(matched '{hit}').",
                                    AI, STRONG, {"field": field_name, "excerpt": value[:300]}))
                break
    return out


def _check_camera_exif(img: Image.Image | None) -> list[Evidence]:
    exif = _exif_fields(img)
    if exif.get("Make") and exif.get("Model") and (
            exif.get("ExposureTime") or exif.get("FNumber") or exif.get("ISOSpeedRatings")):
        make, model = _text(exif["Make"]).strip(), _text(exif["Model"]).strip()
        if any(k in f"{make} {model}".lower() for k in AI_SOFTWARE_KEYWORDS):
            return []
        return [Evidence(LAYER, "Camera EXIF",
                         f"Camera capture data present ({make} {model}). This is a weak hint of "
                         "a real photo: EXIF is trivial to copy onto any image.",
                         NOT_AI, WEAK, {"make": make, "model": model})]
    return []
