"""Tournament-sampling watermark (SynthID-Text: Dathathri et al., "Scalable
watermarking for identifying large language model outputs", Nature 2024).
This is the family in production today: Google uses it in Gemini, and Anthropic
describes Claude's watermark as a version of it.

The green-list scheme changes the model's probabilities. This one leaves them
alone on average and changes only where the randomness comes from.

Writing. For the next token:
  1. Hash the key with the previous `context` tokens. From that, every
     candidate token gets `layers` pseudo-random bits, its g-values.
  2. Draw 2**layers candidates from the model's own distribution and run a
     knockout tournament: in round 1 pairs meet and the one with the higher
     first g-value goes through (coin flip on a tie), round 2 uses the second
     g-value, and so on. The winner is the token that gets written.
Each candidate was an honest draw from the model, and g-values are 0 or 1 with
equal odds, so averaged over keys every token is still chosen with the model's
probability ("non-distortionary"). But for the one key in use, winners carry
more 1-bits than chance.

Nobody draws 2**30 tokens. A round of two-candidate matches has a closed form:
it turns a distribution p into  p(x) * (1 + g(x) - sum_y p(y) g(y)),  so the
whole tournament is that update applied once per layer (`apply`), followed by
one ordinary draw. This is also how the open-source implementation does it.

Reading. Needs the key and the tokenizer, not the model. Recompute the g-values
of each token given the tokens before it and average them. No watermark: 0.5.
Watermark: above 0.5, most clearly in the early layers. Tokens whose context
already appeared are skipped on both sides, so a repeated phrase is neither
forced nor counted twice.

Simplifications: the mean score here is the paper's simplest detector (it also
gives a weighted mean and a trained Bayesian one), and one key drives all
layers through a hash instead of a list of per-layer keys.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import prf

_SALT = 0x746F75726E657931
_LAYER = 0xD6E8FEB86659FD93


@dataclass(frozen=True)
class Config:
    key: str = "demo-key"
    layers: int = 30     # tournament rounds = g-values per token (paper: 30)
    context: int = 4     # preceding tokens that seed the g-values (paper: 4)


def g_values(cfg: Config, context: list[int] | tuple[int, ...], token_ids: np.ndarray) -> np.ndarray:
    """g-values for each candidate: array of 0/1, shape (layers, len(token_ids))."""
    seed = prf.context_seed(prf.key_to_int(cfg.key) ^ _SALT, context)
    base = prf.token_hashes(seed, token_ids)
    with np.errstate(over="ignore"):
        layer = (np.arange(1, cfg.layers + 1, dtype=np.uint64) * np.uint64(_LAYER))[:, None]
    return (prf.mix_np(base[None, :] ^ layer) >> np.uint64(63)).astype(np.float64)


def apply(cfg: Config, probs: np.ndarray, token_ids: np.ndarray, context: list[int]) -> np.ndarray:
    """The whole embedding step: the distribution of the tournament's winner.

    `probs` are the model's probabilities for `token_ids` (already cut down by
    temperature / top-k). The result is sampled from in the normal way.
    """
    if len(context) < cfg.context:
        return probs
    g = g_values(cfg, context[-cfg.context:], token_ids)
    p = probs.astype(np.float64)
    for layer in g:
        p = np.clip(p * (1.0 + layer - p @ layer), 0.0, None)   # clip: rounding only
        p /= p.sum()
    return p


def detect(cfg: Config, tokens: list[int]) -> dict:
    """Average the g-values of the tokens and test the average against 0.5.

    Returns per-token scores (mean of that token's g-values, None if not
    scored), the per-layer means, and the test.
    """
    scores: list[float | None] = [None] * len(tokens)
    layer_sum = np.zeros(cfg.layers)
    seen: set[tuple[int, ...]] = set()
    n = 0
    for i in range(cfg.context, len(tokens)):
        ctx = tuple(tokens[i - cfg.context:i])
        if ctx in seen:
            continue
        seen.add(ctx)
        g = g_values(cfg, ctx, np.array([tokens[i]]))[:, 0]
        layer_sum += g
        scores[i] = float(g.mean())
        n += 1

    bits = n * cfg.layers
    mean = float(layer_sum.sum() / bits) if bits else 0.5
    z = (mean - 0.5) * 2 * math.sqrt(bits) if bits else 0.0
    return {"scores": scores, "scored": n, "mean": mean, "chance_rate": 0.5,
            "token_sd": 0.5 / math.sqrt(cfg.layers),
            "layer_means": (layer_sum / n).tolist() if n else [],
            "layer_sd": 0.5 / math.sqrt(n) if n else 0.0,
            "z": z, "p": prf.p_value(z)}
