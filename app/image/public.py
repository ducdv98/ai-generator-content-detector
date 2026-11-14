"""Stable Diffusion's public pixel watermark.

The reference Stable Diffusion scripts (1.x, 2.x) and the diffusers SDXL
pipeline stamp every output with the open-source `invisible-watermark` package,
method "dwtDct". The payload is a fixed public constant, there is no key, and
the algorithm is published, so anyone can read it. This file is a NumPy port of
that method, written out so each step is visible. (Upstream: ShieldMnt/
invisible-watermark, imwatermark/maxDct.py, MIT licence.)

Writing
  1. Convert to YUV and take the U (blue-difference) channel only. The eye is
     least sensitive to small colour shifts there.
  2. One level of Haar wavelet transform. Keep the low-pass band: each value is
     the sum of a 2x2 pixel group, divided by 2.
  3. Cut that band into 4x4 blocks (8x8 pixels each). Block number k carries
     payload bit k mod N, so the payload repeats hundreds of times.
  4. In each block find the largest value (skipping the first) and move it onto
     a grid of step 36: to x.25 * 36 for a 0 bit, x.75 * 36 for a 1 bit.
Despite the name there is no DCT step: upstream's DCT code is commented out.

Reading: same transform, same largest value, and the bit is whether its
remainder mod 36 is above 18. Each payload bit is a majority vote of its blocks.

It is fragile by construction. The grid is tied to pixel positions and exact
values, so resizing, cropping by a few pixels or strong JPEG compression moves
values off the grid and the votes turn into coin flips.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

SCALE = 36
BLOCK = 4
MIN_PIXELS = 256 * 256        # upstream refuses smaller images
MAX_CHANCE = 1e-5             # report a match only below this chance probability


def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


# Public constants: scripts/txt2img.py in the CompVis and Stability repos, and
# WATERMARK_MESSAGE in diffusers' StableDiffusionXLWatermarker.
PAYLOADS = [
    {"name": "Stable Diffusion 1.x", "text": "StableDiffusionV1", "bits": _bits(b"StableDiffusionV1")},
    {"name": "Stable Diffusion XL", "text": None,
     "bits": np.array([int(c) for c in bin(0b101100111110110010010000011110111011000110011110)[2:]],
                      dtype=np.uint8)},
    {"name": "Stable Diffusion 2.x", "text": "SDV2", "bits": _bits(b"SDV2")},
]


def _u_channel(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yuv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2YUV)
    rows, cols = rgb.shape[0] // 4 * 4, rgb.shape[1] // 4 * 4
    return yuv, yuv[:rows, :cols, 1].astype(np.float64)


def _haar_low(u: np.ndarray) -> np.ndarray:
    return (u[0::2, 0::2] + u[0::2, 1::2] + u[1::2, 0::2] + u[1::2, 1::2]) / 2.0


def _blocks(low: np.ndarray) -> np.ndarray:
    """4x4 blocks of the low-pass band, flattened: shape (rows, cols, 16)."""
    r, c = low.shape[0] // BLOCK, low.shape[1] // BLOCK
    return low[:r * BLOCK, :c * BLOCK].reshape(r, BLOCK, c, BLOCK).transpose(0, 2, 1, 3).reshape(r, c, 16)


def _carrier(blocks: np.ndarray) -> np.ndarray:
    """Index (1..15) of the value in each block that carries the bit."""
    return np.abs(blocks[..., 1:]).argmax(axis=-1) + 1


def embed(rgb: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Stamp `bits` into a uint8 RGB image the way Stable Diffusion's script does."""
    if rgb.shape[0] * rgb.shape[1] < MIN_PIXELS:
        raise ValueError("image too small for this watermark (needs at least 256 x 256 pixels)")
    yuv, u = _u_channel(rgb)
    low = _haar_low(u)
    blocks = _blocks(low)
    r, c = blocks.shape[:2]
    pos = _carrier(blocks)
    val = np.take_along_axis(blocks, pos[..., None], -1)[..., 0]
    bit = np.asarray(bits)[np.arange(r * c) % len(bits)].reshape(r, c)
    new = (val // SCALE + 0.25 + 0.5 * bit) * SCALE       # values here are never negative

    # Put the change back: one low-pass value spreads evenly over its 2x2 pixels.
    delta_blocks = np.zeros_like(blocks)
    np.put_along_axis(delta_blocks, pos[..., None], (new - val)[..., None], -1)
    delta = np.zeros_like(low)
    delta[:r * BLOCK, :c * BLOCK] = (delta_blocks.reshape(r, c, BLOCK, BLOCK)
                                     .transpose(0, 2, 1, 3).reshape(r * BLOCK, c * BLOCK))
    out = np.empty_like(u)
    # Upstream passes the two detail bands to the inverse transform in swapped
    # order, which mirrors each 2x2 pixel group across its diagonal. Kept, so
    # the output matches what real Stable Diffusion images went through.
    out[0::2, 0::2] = u[0::2, 0::2] + delta / 2
    out[0::2, 1::2] = u[1::2, 0::2] + delta / 2
    out[1::2, 0::2] = u[0::2, 1::2] + delta / 2
    out[1::2, 1::2] = u[1::2, 1::2] + delta / 2
    yuv[:u.shape[0], :u.shape[1], 1] = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def votes(rgb: np.ndarray) -> np.ndarray:
    """One 0/1 vote per 8x8-pixel block: shape (rows, cols)."""
    _, u = _u_channel(rgb)
    blocks = _blocks(_haar_low(u))
    val = np.abs(np.take_along_axis(blocks, _carrier(blocks)[..., None], -1)[..., 0])
    return (val % SCALE > 0.5 * SCALE).astype(np.uint8)


def decode(v: np.ndarray, n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Majority vote per payload bit. Returns (bits, share of blocks voting 1)."""
    flat = v.reshape(-1)
    idx = np.arange(flat.size) % n_bits
    ones = np.bincount(idx, weights=flat, minlength=n_bits)
    count = np.maximum(np.bincount(idx, minlength=n_bits), 1)
    share = ones / count
    return (share * 255 > 127).astype(np.uint8), share


def chance_of_match(matches: int, n: int) -> float:
    """P(at least `matches` of `n` coin-flip bits agree with the payload)."""
    return sum(math.comb(n, k) for k in range(matches, n + 1)) / 2 ** n


def _agreement_map(v: np.ndarray, bits: np.ndarray, max_cells: int = 40) -> dict:
    """Share of blocks agreeing with the payload, on a coarse grid over the image."""
    r, c = v.shape
    expected = bits[np.arange(r * c) % len(bits)].reshape(r, c)
    agree = (v == expected).astype(np.float64)
    step = max(1, math.ceil(max(r, c) / max_cells))
    rr, cc = math.ceil(r / step), math.ceil(c / step)
    cells = [[round(float(agree[i * step:(i + 1) * step, j * step:(j + 1) * step].mean()), 3)
              for j in range(cc)] for i in range(rr)]
    return {"rows": rr, "cols": cc, "cells": cells, "cell_px": 8 * step}


def detect(rgb: np.ndarray) -> dict:
    """Try every known public payload; report the best and whether it is a match."""
    if rgb.shape[0] * rgb.shape[1] < MIN_PIXELS:
        return {"verdict": "too_small", "candidates": [], "blocks": 0}
    v = votes(rgb)
    cands = []
    for p in PAYLOADS:
        bits, share = decode(v, len(p["bits"]))
        n = len(bits)
        matched = int((bits == p["bits"]).sum())
        cands.append({"name": p["name"], "text": p["text"], "n": n, "matched": matched,
                      "chance": chance_of_match(matched, n),
                      "decoded": bits.tolist(), "expected": p["bits"].tolist(),
                      "share": [round(float(x), 3) for x in share]})
    best = min(range(len(cands)), key=lambda i: cands[i]["chance"])
    found = cands[best]["chance"] <= MAX_CHANCE
    return {"verdict": "detected" if found else "none", "candidates": cands, "best": best,
            "blocks": int(v.size), "votes_per_bit": int(v.size // cands[best]["n"]),
            "map": _agreement_map(v, PAYLOADS[best]["bits"]),
            "grid_px": [int(v.shape[1] * 8), int(v.shape[0] * 8)], "max_chance": MAX_CHANCE}
