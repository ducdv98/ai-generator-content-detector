"""A keyed audio watermark made of faint echoes.

Writing. The key picks a +1/-1 value for each of about 550 narrow pitch bands
(21.5 Hz wide, from 150 Hz to 12 kHz), and the sound is passed through a filter
that makes each band a fraction of a decibel louder or quieter accordingly.
Two ways to picture the same thing:

  * in pitch: a fine, key-shaped ripple laid over the sound's spectrum;
  * in time: a pattern of very faint echoes in the 50 ms after every sound
    (a ripple across pitch *is* a set of echoes; the filter is built
    "minimum-phase" so all of them come after the sound, where the ear's
    forward masking hides them, and none before it).

"Echo hiding" is one of the oldest audio watermarks (Gruhl, Lu and Bender,
1996). The mark is the sound itself, re-coloured, so it is loud where the music
is loud and silent where it is silent. That is the audio form of the perceptual
masking the image step used.

Reading. Needs the key, not the original. Measure how loud each band is in each
short stretch of time, subtract the smooth trend across neighbouring bands (the
instrument's own tone), average over time, and correlate what is left with the
key's +1/-1 pattern. The music's own fine structure changes from note to note
and averages away; the ripple is the same in every instant and stays.

Because the pattern does not depend on time, the detector needs no
synchronisation: cut the start off, keep any ten seconds, change the tempo, and
the bands are where they were. It also means every stretch of sound can be
scored separately, which gives a timeline of where the mark is. What does move
the bands is anything that changes pitch, and that is this scheme's weak spot.

Deployed systems (Meta's AudioSeal, Google's SynthID for audio) train an
encoder network to write the mark and a decoder to find it, with the mark
shaped by the sound moment to moment. The same two ideas survive: hide the
mark under the sound that masks it, and pool evidence over time.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .codec import RATE

N = 8192                    # analysis window: 186 ms, one bin = 5.4 Hz
HOP = 2048
CELL = 4                    # bins per band: 21.5 Hz
F_LO, F_HI = 150.0, 12000.0
K0 = int(math.ceil(F_LO * N / RATE))
CELLS = (int(F_HI * N / RATE) - K0) // CELL
PREAMBLE_BITS, PAYLOAD_BITS = 16, 16
TOTAL_BITS = PREAMBLE_BITS + PAYLOAD_BITS
DETECT_Z = 6.0              # as in the image and video steps
# name -> (largest change to any band, how far past its neighbours a band is pushed), in decibels
STRENGTHS = {"subtle": (1.0, 0.5), "normal": (2.0, 1.0), "strong": (3.5, 1.5)}
WINDOW_S, STEP_S = 1.5, 0.5                                    # the timeline's window and step
BANDS = 6                   # rows of the time-by-pitch evidence map


def _pattern(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keyed (which message bit each band carries, each band's +1/-1, preamble signs)."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(("audio:" + key).encode()).digest()[:8], "big"))
    assign = rng.permutation(CELLS) % TOTAL_BITS
    chips = rng.choice(np.array([-1.0, 1.0]), size=CELLS)
    preamble = rng.choice(np.array([-1.0, 1.0]), size=PREAMBLE_BITS)
    return assign, chips, preamble


def _filter(gain_db: np.ndarray) -> np.ndarray:
    """Impulse response of a filter that changes each band's loudness by
    `gain_db`: a spike followed by the echo pattern."""
    curve = np.zeros(N // 2 + 1)
    curve[K0:K0 + CELLS * CELL] = np.repeat(gain_db, CELL)
    # Minimum-phase filter with exactly this loudness curve (cepstrum folding):
    # keeps the ripple, moves every echo to after the sound.
    cep = np.fft.irfft(curve * (math.log(10) / 20), N)
    cep[1:N // 2] *= 2
    cep[N // 2 + 1:] = 0
    return np.fft.irfft(np.exp(np.fft.rfft(cep)), N)


def plan(x: np.ndarray, payload: int, key: str, strength: str = "normal") -> np.ndarray:
    """Decide how much to push each band, in dB.

    The writer listens first. It measures where each band already sits (the
    same measurement the detector will make) and compares that with where the
    key wants it. A band the music already has on the right side is left
    alone; a band on the wrong side is pushed across, but never by more than
    `limit`. Spending the changes only where they are needed is what makes a
    small mark detectable under music's own strong fine structure ("informed
    embedding"; a fixed ripple of the same size is drowned out).
    """
    if not 0 <= payload < 2 ** PAYLOAD_BITS:
        raise ValueError("ID must fit in 16 bits (4 hex digits)")
    limit, margin = STRENGTHS[strength]
    assign, chips, preamble = _pattern(key)
    bits = np.concatenate([preamble, [1.0 if (payload >> i) & 1 else -1.0 for i in range(PAYLOAD_BITS)]])
    target = chips * bits[assign]                       # which side of its neighbours each band should be on
    have = _average(*_band_levels(x)) * target          # how far on the right side it already is (dB)
    return target * np.clip(margin - have, 0.0, limit)


def embed(x: np.ndarray, payload: int, key: str, strength: str = "normal") -> np.ndarray:
    """Write the mark: pass the sound through the filter that `plan` designs."""
    h = _filter(plan(x, payload, key, strength))
    size = 1 << int(math.ceil(math.log2(len(x) + N)))
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)[:len(x)]
    peak = np.abs(y).max()
    return y / peak * 0.98 if peak > 0.98 else y


TONE_DB = 5.0               # a band this far above or below its neighbours holds a pure tone (or a gap), not the mark


def _band_levels(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How loud every band is in every frame, relative to its neighbours.

    Returns (frames x bands difference in dB, frames x bands "usable" flag).
    The neighbours' level is their median, so one strong harmonic does not
    drag down the bands beside it. An entry is not usable when the frame is
    silent, or when the band sticks out by more than TONE_DB: that is a
    musical note sitting in the band, far louder than any watermark, and it
    says nothing about the mark. The mark is read from everything else: noise,
    breath, reverb, the gaps between harmonics.
    """
    if len(x) < N:
        x = np.pad(x, (0, N - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, N)[::HOP]
    win = np.hanning(N)
    resid = np.empty((len(frames), CELLS))
    energy = np.empty(len(frames))
    for i in range(0, len(frames), 128):                       # in chunks, to keep memory small
        p = np.abs(np.fft.rfft(frames[i:i + 128] * win, axis=1)) ** 2
        cells = p[:, K0:K0 + CELLS * CELL].reshape(len(p), CELLS, CELL).sum(axis=2)
        level = 10 * np.log10(cells + 1e-14)
        near = np.lib.stride_tricks.sliding_window_view(np.pad(level, ((0, 0), (6, 6)), mode="edge"), 13, axis=1)
        resid[i:i + 128] = level - np.median(near, axis=2)
        energy[i:i + 128] = cells.sum(axis=1)
    usable = (np.abs(resid) < TONE_DB) & (energy > energy.max() * 1e-5)[:, None]
    return resid, usable


def _average(resid: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Each band's mean difference over the frames where it is usable (0 if never)."""
    n = usable.sum(axis=0)
    return np.where(n > 0, (resid * usable).sum(axis=0) / np.maximum(n, 1), 0.0)


def _score(m: np.ndarray, pattern: tuple) -> tuple[float, np.ndarray]:
    """Correlate time-averaged band residuals with the key. Each bit's score
    is exactly mean 0, variance 1 for any key that did not write the mark."""
    assign, chips, preamble = pattern
    num = np.bincount(assign, weights=m * chips, minlength=TOTAL_BITS)
    den = np.sqrt(np.bincount(assign, weights=m * m, minlength=TOTAL_BITS)) + 1e-12
    z_bits = num / den
    return float((z_bits[:PREAMBLE_BITS] * preamble).sum() / math.sqrt(PREAMBLE_BITS)), z_bits


def detect(x: np.ndarray, key: str) -> dict:
    """Score the whole clip, each 1.5-second stretch, and each pitch range."""
    pattern = _pattern(key)
    assign, chips, preamble = pattern
    resid, usable = _band_levels(x)
    active = usable.any(axis=1)
    if active.sum() < 4:
        return {"verdict": "too_short", "z": 0.0, "threshold": DETECT_Z, "timeline": [], "building": []}
    frame_s = HOP / RATE
    z, z_bits = _score(_average(resid, usable), pattern)

    # Timeline: the same test on a sliding window.
    win, step = max(4, round(WINDOW_S / frame_s)), max(1, round(STEP_S / frame_s))
    is_pre = assign < PREAMBLE_BITS
    sign = np.where(is_pre, chips * preamble[np.minimum(assign, PREAMBLE_BITS - 1)], 0.0)
    edges = np.linspace(0, CELLS, BANDS + 1).astype(int)
    timeline, grid = [], []
    for start in range(0, max(1, len(resid) - win + 1), step):
        sel = active[start:start + win]
        t = round((start + win / 2) * frame_s + N / 2 / RATE, 2)
        if sel.sum() < 4:
            timeline.append({"t": t, "z": None})
            grid.append([None] * BANDS)
            continue
        m = _average(resid[start:start + win], usable[start:start + win])
        timeline.append({"t": t, "z": round(_score(m, pattern)[0], 2)})
        grid.append([round(float((m[a:b] * sign[a:b]).sum() / (math.sqrt((m[a:b][is_pre[a:b]] ** 2).sum()) + 1e-12)), 2)
                     for a, b in zip(edges[:-1], edges[1:])])

    # Evidence building up: the whole-clip test on the first n seconds.
    building, total, count = [], np.zeros(CELLS), np.zeros(CELLS)
    for i in range(len(resid)):
        total += resid[i] * usable[i]
        count += usable[i]
        if (i + 1) % step == 0 or i == len(resid) - 1:
            m = np.where(count > 0, total / np.maximum(count, 1), 0.0)
            building.append({"t": round((i * HOP + N) / RATE, 2),
                             "z": round(_score(m, pattern)[0], 2) if active[:i + 1].sum() >= 4 else None})

    found = z >= DETECT_Z
    pz = z_bits[PREAMBLE_BITS:]
    payload = int(sum(1 << i for i in range(PAYLOAD_BITS) if pz[i] > 0)) if found else None
    return {"verdict": "detected" if found else "none", "z": z, "threshold": DETECT_Z,
            "p": 0.5 * math.erfc(z / math.sqrt(2)), "payload_hex": f"{payload:04X}" if found else None,
            "preamble_z": [round(float(v), 2) for v in z_bits[:PREAMBLE_BITS] * preamble],
            "payload_z": [round(float(v), 2) for v in pz],
            "bands_used": CELLS, "per_bit": CELLS // TOTAL_BITS, "ceiling": round(math.sqrt(is_pre.sum()), 1),
            "seconds_with_sound": round(float(active.sum() * frame_s), 1),
            "usable_share": round(float(usable[active].mean()), 2),
            "timeline": timeline, "building": building,
            "map": {"rows": BANDS, "cols": len(grid), "cells": [list(r) for r in zip(*grid)][::-1],
                    "hz": [round(float((K0 + e * CELL) * RATE / N)) for e in edges][::-1],
                    "t0": timeline[0]["t"] - WINDOW_S / 2 if timeline else 0.0, "step": STEP_S, "window": WINDOW_S}}
