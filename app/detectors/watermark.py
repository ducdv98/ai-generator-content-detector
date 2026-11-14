"""Layer 2: invisible watermarks we know how to read.

  * The DWT-DCT watermark that Stable Diffusion's reference code and the
    diffusers SDXL pipeline embed (via the open-source `invisible-watermark`
    package). The payloads are public constants, so anyone can check for them.
  * This service's own keyed watermark (see app/signing.py).

A watermark detector can only find schemes it knows. Proprietary ones
(e.g. Google SynthID for images) are not checkable here.
"""
from __future__ import annotations

import math

import numpy as np

from .. import signing
from ..report import AI, INFO, NEUTRAL, STRONG, Evidence

LAYER = "watermark"

# Public payloads. SD 1.x / 2.x: scripts/txt2img.py in the reference repos.
# SDXL: WATERMARK_MESSAGE in diffusers' StableDiffusionXLWatermarker.
_SDXL_MESSAGE = 0b101100111110110010010000011110111011000110011110


def _bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


KNOWN_DWTDCT = [
    ("Stable Diffusion 1.x", _bytes_to_bits(b"StableDiffusionV1")),
    ("Stable Diffusion XL", np.array([int(c) for c in bin(_SDXL_MESSAGE)[2:]], dtype=np.uint8)),
    ("Stable Diffusion 2.x", _bytes_to_bits(b"SDV2")),
]
# The DWT-DCT scheme is lossy even on untouched images (often only 75-100% of
# bits come back), so instead of demanding an exact match we ask how likely the
# observed number of matching bits would be by pure chance.
MAX_CHANCE = 1e-5


def chance_of_match(matches: int, n: int) -> float:
    """P(at least `matches` of `n` random bits agree with the payload)."""
    return sum(math.comb(n, k) for k in range(matches, n + 1)) / 2 ** n


def _decode_dwtdct(bgr: np.ndarray, n_bits: int) -> np.ndarray | None:
    from imwatermark import WatermarkDecoder
    dec = WatermarkDecoder("bits", n_bits)
    bits = dec.decode(bgr, "dwtDct")
    return np.array([1 if b else 0 for b in bits], dtype=np.uint8)


def check_stable_diffusion(rgb: np.ndarray) -> list[Evidence]:
    name = "Stable Diffusion watermark"
    if min(rgb.shape[:2]) < 256:
        return [Evidence(LAYER, name, "Image is too small (under 256 px) to check for the "
                         "Stable Diffusion watermark.")]
    try:
        import imwatermark  # noqa: F401
    except Exception:
        return [Evidence(LAYER, name, "Not checked: the invisible-watermark package is not "
                         "installed.")]
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    best = {}
    for label, expected in KNOWN_DWTDCT:
        try:
            got = _decode_dwtdct(bgr, len(expected))
        except Exception as exc:
            return [Evidence(LAYER, name, f"Watermark decode failed ({exc}).")]
        if got is None or len(got) != len(expected):
            continue
        n = len(expected)
        matches = int((got == expected).sum())
        chance = chance_of_match(matches, n)
        best[label] = f"{matches}/{n} bits"
        if chance <= MAX_CHANCE:
            return [Evidence(LAYER, name,
                             f"Found the invisible watermark that {label} embeds: {matches} of "
                             f"{n} bits match, which would happen by chance about once in "
                             f"{1 / chance:,.0f} images.",
                             AI, STRONG, {"scheme": "dwtDct", "generator": label,
                                          "bits_matched": matches, "bits_total": n,
                                          "chance_probability": chance})]
    return [Evidence(LAYER, name, "No Stable Diffusion watermark found. (It is fragile: resizing "
                     "or heavy compression removes it, and many tools never add it.)",
                     NEUTRAL, INFO, {"bits_matched": best})]


def check_own(rgb: np.ndarray) -> list[Evidence]:
    name = "This service's watermark"
    if min(rgb.shape[:2]) < 64:
        return []
    res = signing.detect(rgb)
    details = {"z_score": round(res.z_score, 2), "threshold": res.threshold}
    if res.present:
        details["payload"] = res.payload
        details["payload_hex"] = f"{res.payload:08X}"
        return [Evidence(LAYER, name,
                         f"Found a watermark embedded by this service (payload ID "
                         f"{res.payload:08X}, detection score {res.z_score:.1f} against a "
                         f"threshold of {res.threshold:.0f}).",
                         AI, STRONG, details)]
    return [Evidence(LAYER, name, "Not present.", NEUTRAL, INFO, details)]


def analyze(rgb: np.ndarray) -> list[Evidence]:
    return check_stable_diffusion(rgb) + check_own(rgb)
