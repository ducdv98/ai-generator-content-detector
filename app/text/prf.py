"""Keyed pseudo-random function shared by the watermark schemes.

Every sampling-time text watermark needs the same primitive: a function that
turns (secret key, the few tokens just written, a candidate next token) into a
number that looks random to anyone without the key and is exactly reproducible
by anyone with it. The generator uses it to nudge the choice of the next token;
the detector recomputes it for the tokens it sees and asks whether they score
better than chance.

This one is a SplitMix64 mixer. It is fast and well distributed, which is all a
demo needs. A production scheme would use a cryptographic PRF so the key cannot
be worked out from watermarked samples.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

MASK = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


def key_to_int(key: str) -> int:
    """Any string key -> 64-bit integer."""
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")


def mix(x: int) -> int:
    x &= MASK
    x = ((x ^ (x >> 30)) * _M1) & MASK
    x = ((x ^ (x >> 27)) * _M2) & MASK
    return x ^ (x >> 31)


def context_seed(key: int, context: tuple[int, ...] | list[int]) -> int:
    """Fold the key and the preceding tokens into one 64-bit seed."""
    s = mix(key)
    for tok in context:
        s = mix(s ^ ((int(tok) + 1) * _GOLDEN))
    return s


def mix_np(x: np.ndarray) -> np.ndarray:
    """Vectorised `mix` over a uint64 array (wrap-around is intended)."""
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(_M1)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(_M2)
        return x ^ (x >> np.uint64(31))


def token_hashes(seed: int, token_ids: np.ndarray) -> np.ndarray:
    """One pseudo-random uint64 per candidate token, for this context seed."""
    with np.errstate(over="ignore"):
        t = (token_ids.astype(np.uint64) + np.uint64(1)) * np.uint64(_GOLDEN)
    return mix_np(np.uint64(seed) ^ t)


def p_value(z: float) -> float:
    """One-sided chance of a z-score this high in text with no watermark."""
    return 0.5 * math.erfc(z / math.sqrt(2))
