"""Keyed invisible watermark for the "Sign and verify" demo.

Spread-spectrum watermark in the block-DCT domain, written from scratch with
NumPy + OpenCV so it has no licence or model-weight dependency.

How it works
------------
1. The luminance channel is resized to a fixed 512x512 "canonical" grid. All
   embedding and detection happens on that grid, which is what makes the mark
   survive resizing: any resized copy maps back to the same grid.
2. The grid is split into 8x8 blocks and DCT-transformed. Twelve low/mid
   frequency coefficients per block carry the signal (low enough to survive
   JPEG, high enough to stay invisible).
3. A secret key seeds a PRNG that (a) assigns every coefficient to one of 48
   message bits and (b) gives it a random +1/-1 "chip". Each bit is spread over
   ~1000 coefficients: coefficient += strength * chip * (+1 or -1 for the bit).
4. The pixel-domain difference is scaled back to the original resolution and
   added to the image.

Detection correlates the coefficients with the keyed chips. Without the key
the pattern is indistinguishable from noise. The first 16 bits are a fixed
preamble: their combined correlation is a z-score that says whether *this
key's* watermark is present; the remaining 32 bits are the payload.

Known limits: cropping, rotation and heavy geometric edits break the block
alignment and defeat detection. Production schemes (learned watermarks such as
SynthID or Watermark Anything) add synchronisation to survive those.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import cv2
import numpy as np

GRID = 512
BLOCK = 8
# (row, col) DCT positions used per block: the diagonals u+v = 2..4.
POSITIONS = [(u, s - u) for s in (2, 3, 4) for u in range(s + 1)]
PREAMBLE_BITS = 16
PAYLOAD_BITS = 32
TOTAL_BITS = PREAMBLE_BITS + PAYLOAD_BITS
DEFAULT_STRENGTH = 5.0
# Presence threshold on the preamble z-score. Under "no watermark" the score
# is ~N(0,1), so 6.0 corresponds to roughly a one-in-a-billion false alarm.
DETECT_Z = 6.0

_DEV_KEY = "dev-key-change-me"


def get_key() -> str:
    return os.environ.get("AIGC_WM_KEY", _DEV_KEY)


def using_dev_key() -> bool:
    return "AIGC_WM_KEY" not in os.environ


def _dct_matrix(n: int = BLOCK) -> np.ndarray:
    k = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * x + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m


_D = _dct_matrix()
_PU = np.array([p[0] for p in POSITIONS])
_PV = np.array([p[1] for p in POSITIONS])


def _blocks(y: np.ndarray) -> np.ndarray:
    n = GRID // BLOCK
    return y.reshape(n, BLOCK, n, BLOCK).transpose(0, 2, 1, 3)


def _unblocks(b: np.ndarray) -> np.ndarray:
    return b.transpose(0, 2, 1, 3).reshape(GRID, GRID)


def _dct(y: np.ndarray) -> np.ndarray:
    return np.einsum("ij,abjk,lk->abil", _D, _blocks(y), _D, optimize=True)


def _idct(c: np.ndarray) -> np.ndarray:
    return _unblocks(np.einsum("ji,abjk,kl->abil", _D, c, _D, optimize=True))


def _pattern(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keyed (bit assignment, chips, preamble signs)."""
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    n = (GRID // BLOCK) ** 2 * len(POSITIONS)
    assign = rng.permutation(n) % TOTAL_BITS
    chips = rng.choice(np.array([-1.0, 1.0]), size=n)
    preamble = rng.choice(np.array([-1.0, 1.0]), size=PREAMBLE_BITS)
    return assign, chips, preamble


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _to_grid(y: np.ndarray) -> np.ndarray:
    h, w = y.shape
    interp = cv2.INTER_AREA if (h > GRID and w > GRID) else cv2.INTER_CUBIC
    return cv2.resize(y, (GRID, GRID), interpolation=interp)


@dataclass
class EmbedResult:
    image: np.ndarray  # uint8 RGB
    psnr_db: float


@dataclass
class DetectResult:
    present: bool
    z_score: float
    payload: int | None
    payload_confidence: float  # weakest payload bit, in sigmas
    threshold: float = DETECT_Z


def embed(rgb: np.ndarray, payload: int, key: str | None = None,
          strength: float = DEFAULT_STRENGTH) -> EmbedResult:
    """Embed a 32-bit payload into a uint8 RGB image."""
    if not 0 <= payload < 2 ** PAYLOAD_BITS:
        raise ValueError("payload must fit in 32 bits")
    if min(rgb.shape[:2]) < 64:
        raise ValueError("image too small to watermark (min 64 px per side)")
    key = key or get_key()
    assign, chips, preamble = _pattern(key)
    bits = np.concatenate([
        preamble,
        np.array([1.0 if (payload >> i) & 1 else -1.0 for i in range(PAYLOAD_BITS)]),
    ])

    src = rgb.astype(np.float64)
    h, w = src.shape[:2]
    grid = _to_grid(_luma(src))
    coefs = _dct(grid)

    # Perceptual masking: push harder in textured blocks, softer in flat ones.
    act = np.abs(coefs[:, :, _PU, _PV]).mean(axis=2)
    mask = np.clip(act / (act.mean() + 1e-9), 0.6, 1.6)
    mask /= np.sqrt((mask ** 2).mean())  # keep overall energy = strength

    delta = (strength * chips * bits[assign]).reshape(GRID // BLOCK, GRID // BLOCK, -1)
    delta_c = np.zeros_like(coefs)
    delta_c[:, :, _PU, _PV] = delta * mask[:, :, None]
    residual = _idct(delta_c)

    interp = cv2.INTER_AREA if (h < GRID and w < GRID) else cv2.INTER_CUBIC
    residual_full = cv2.resize(residual, (w, h), interpolation=interp)
    # Shrinking the image loses some signal; compensate so small images still verify.
    scale = np.sqrt((residual ** 2).mean() / max((residual_full ** 2).mean(), 1e-12))
    residual_full *= min(scale, 2.0)

    out = np.clip(np.rint(src + residual_full[..., None]), 0, 255).astype(np.uint8)
    mse = ((out.astype(np.float64) - src) ** 2).mean()
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    return EmbedResult(out, psnr)


def detect(rgb: np.ndarray, key: str | None = None) -> DetectResult:
    """Look for this key's watermark in a uint8 RGB image."""
    key = key or get_key()
    assign, chips, preamble = _pattern(key)
    grid = _to_grid(_luma(rgb.astype(np.float64)))
    c = _dct(grid)[:, :, _PU, _PV]

    # Whiten per frequency and clip: image coefficients are heavy-tailed, and a
    # few large edges would otherwise drown the watermark.
    sd = c.reshape(-1, len(POSITIONS)).std(axis=0) + 1e-9
    x = np.clip(c / sd, -1.5, 1.5).reshape(-1)
    x = x - x.mean()
    sigma = x.std() + 1e-12

    prod = x * chips
    sums = np.bincount(assign, weights=prod, minlength=TOTAL_BITS)
    counts = np.bincount(assign, minlength=TOTAL_BITS)
    z_bits = sums / (sigma * np.sqrt(counts))

    z = float((z_bits[:PREAMBLE_BITS] * preamble).sum() / np.sqrt(PREAMBLE_BITS))
    present = z >= DETECT_Z
    payload = None
    conf = 0.0
    if present:
        pz = z_bits[PREAMBLE_BITS:]
        payload = int(sum((1 << i) for i in range(PAYLOAD_BITS) if pz[i] > 0))
        conf = float(np.abs(pz).min())
    return DetectResult(present, z, payload, conf)
