"""Keyed spread-spectrum watermark: the evidence behind a detection.

The scheme itself is in app/signing.py. In one line: a secret key generates a
+1/-1 noise pattern over mid-frequency DCT coefficients of the brightness
channel, and that pattern (times each payload bit) is added faintly to the
image. Detection multiplies the image's coefficients by the same pattern and
adds up: with the right key the faint pattern lines up and the sum grows with
the number of coefficients; with any other key it averages out to zero.

This is the principle (Cox et al., 1997) that learned watermarks such as
SynthID, TrustMark and Meta's Stable Signature still follow. They replace the
hand-made pattern and the correlation with a trained encoder and decoder, which
is what buys robustness to crops and rotations that this one lacks.
"""
from __future__ import annotations

import math

import numpy as np

from .. import signing as s

CELLS = 8   # the local-evidence map is CELLS x CELLS over the image


def embed(rgb: np.ndarray, payload: int, key: str) -> np.ndarray:
    return s.embed(rgb, payload, key=key).image


def coefficients(rgb: np.ndarray) -> np.ndarray:
    """The values the watermark lives in: 12 mid-frequency DCT values for each
    8x8 block of the brightness channel, on the fixed 512x512 grid."""
    return s._dct(s._to_grid(s._luma(rgb.astype(np.float64))))[:, :, s._PU, s._PV]


def detect(rgb: np.ndarray, key: str) -> dict:
    """Correlate with the key's pattern. Returns the overall score, one score
    per message bit, and a map of where in the image the pattern was found."""
    if min(rgb.shape[:2]) < 64:
        return {"verdict": "too_small", "z": 0.0, "threshold": s.DETECT_Z}
    return detect_coefficients(coefficients(rgb), key)


def detect_coefficients(coefs: np.ndarray, key: str) -> dict:
    """The detector itself. Works on one image's coefficients, or on the
    average of many frames' coefficients (see app/video/frames.py)."""
    assign, chips, preamble = s._pattern(key)
    sd = coefs.reshape(-1, len(s.POSITIONS)).std(axis=0) + 1e-9
    x = np.clip(coefs / sd, -1.5, 1.5).reshape(-1)
    x = x - x.mean()
    sigma = x.std() + 1e-12

    prod = x * chips                                   # positive: agrees with the key's pattern
    counts = np.bincount(assign, minlength=s.TOTAL_BITS)
    z_bits = np.bincount(assign, weights=prod, minlength=s.TOTAL_BITS) / (sigma * np.sqrt(counts))
    z = float((z_bits[:s.PREAMBLE_BITS] * preamble).sum() / math.sqrt(s.PREAMBLE_BITS))
    found = z >= s.DETECT_Z

    # Local evidence, from the preamble's coefficients only (their sign is
    # known in advance, so the map is not flattered by the decoded payload).
    is_pre = assign < s.PREAMBLE_BITS
    signed = np.where(is_pre, prod * preamble[np.minimum(assign, s.PREAMBLE_BITS - 1)], 0.0)
    per = s.GRID // s.BLOCK // CELLS
    cell_sum = signed.reshape(CELLS, per, CELLS, per, -1).sum(axis=(1, 3, 4))
    cell_n = is_pre.reshape(CELLS, per, CELLS, per, -1).sum(axis=(1, 3, 4))
    cell_z = cell_sum / (sigma * np.sqrt(np.maximum(cell_n, 1)))

    pz = z_bits[s.PREAMBLE_BITS:]
    payload = int(sum(1 << i for i in range(s.PAYLOAD_BITS) if pz[i] > 0)) if found else None
    return {"verdict": "detected" if found else "none", "z": z, "threshold": s.DETECT_Z,
            "p": 0.5 * math.erfc(z / math.sqrt(2)),
            "payload_hex": f"{payload:08X}" if payload is not None else None,
            "preamble_z": [round(float(v), 2) for v in z_bits[:s.PREAMBLE_BITS] * preamble],
            "payload_z": [round(float(v), 2) for v in pz],
            "coefficients": int(x.size), "per_bit": int(counts.mean()),
            "map": {"rows": CELLS, "cols": CELLS,
                    "cells": [[round(float(v), 2) for v in row] for row in cell_z]}}
