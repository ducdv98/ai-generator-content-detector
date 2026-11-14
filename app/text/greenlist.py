"""Green-list watermark (Kirchenbauer et al., "A Watermark for Large Language
Models", ICML 2023). The simplest sampling-time scheme, and the one most later
work is measured against.

Writing. Before each token is sampled:
  1. Hash the key with the previous `context` token(s).
  2. Use that hash to split the whole vocabulary into a "green" part (a share
     `gamma` of it) and a "red" part. The split is different after every word.
  3. Add `delta` to the score (logit) of every green token, then sample as usual.
The model can still pick a red token when it is much the best fit, so quality
holds up, but over a paragraph it lands on green far more often than `gamma`.

Reading. Needs the key and the tokenizer, not the model. For each token, redo
the split from the token before it and note whether the token is green. Text
with no watermark hits green about `gamma` of the time; count the excess and
turn it into a z-score.

Differences from the paper: each token is green independently with probability
`gamma` (the paper shuffles the vocabulary and takes the first gamma * |V|),
which keeps the detector's maths exact and the code short.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import prf

_SALT = 0x67726E6C69737431  # keeps this scheme's randomness separate from the other's


@dataclass(frozen=True)
class Config:
    key: str = "demo-key"
    gamma: float = 0.25   # share of the vocabulary that is green at each step
    delta: float = 2.0    # bonus added to green logits (strength of the mark)
    context: int = 1      # how many preceding tokens seed the split


def green_mask(cfg: Config, context: list[int] | tuple[int, ...], token_ids: np.ndarray) -> np.ndarray:
    """True for each of `token_ids` that is green after `context`."""
    seed = prf.context_seed(prf.key_to_int(cfg.key) ^ _SALT, context)
    cutoff = np.uint64(min(int(cfg.gamma * 2 ** 64), prf.MASK))
    return prf.token_hashes(seed, token_ids) < cutoff


def bias_logits(cfg: Config, logits: np.ndarray, context: list[int]) -> np.ndarray:
    """The whole embedding step: raise every green token's logit by delta."""
    if len(context) < cfg.context:
        return logits
    ids = np.arange(logits.shape[0])
    return logits + cfg.delta * green_mask(cfg, context[-cfg.context:], ids)


def detect(cfg: Config, tokens: list[int]) -> dict:
    """Count green tokens and test the count against chance.

    Returns per-token marks (1 green, 0 red, None not scored) and the test.
    A (context, token) pair is scored once: repeating a phrase repeats its
    colours, which would count the same evidence twice.
    """
    marks: list[int | None] = [None] * len(tokens)
    seen: set[tuple[int, ...]] = set()
    for i in range(cfg.context, len(tokens)):
        gram = tuple(tokens[i - cfg.context:i + 1])
        if gram in seen:
            continue
        seen.add(gram)
        marks[i] = int(green_mask(cfg, gram[:-1], np.array([gram[-1]]))[0])

    scored = [m for m in marks if m is not None]
    n, hits = len(scored), sum(scored)
    expected = cfg.gamma * n
    sd = math.sqrt(cfg.gamma * (1 - cfg.gamma) * n) if n else 0.0
    z = (hits - expected) / sd if sd else 0.0
    return {"marks": marks, "scored": n, "hits": hits, "expected": expected,
            "rate": hits / n if n else 0.0, "chance_rate": cfg.gamma,
            "z": z, "p": prf.p_value(z)}
