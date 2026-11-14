"""A short synthesised tune to experiment on: plucked chords, a bass line and
light percussion over a faint room tone, eight seconds long. Made with NumPy so the project needs no
borrowed recording. It has what an audio watermark test needs: notes that
change, sound across the whole pitch range, and short silences."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .codec import RATE

SECONDS = 8.0
_BEAT = 0.3                                   # seconds per eighth note (100 beats per minute)
_CHORDS = [[261.63, 329.63, 392.00, 523.25],  # C
           [196.00, 246.94, 293.66, 392.00],  # G
           [220.00, 261.63, 329.63, 440.00],  # A minor
           [174.61, 220.00, 261.63, 349.23]]  # F


def _pluck(freq: float, dur: float, bright: float = 1.0) -> np.ndarray:
    """A plucked-string note: harmonics that die away, the high ones fastest."""
    t = np.arange(int(dur * RATE)) / RATE
    out = np.zeros_like(t)
    for h in range(1, int(min(40, 14000 / freq)) + 1):
        out += (bright / h ** 1.1) * np.sin(2 * np.pi * freq * h * t * (1 + 0.0004 * h * h)) * np.exp(-t * (1.2 + 0.55 * h))
    return out * np.minimum(1.0, t / 0.004)


def _add(mix: np.ndarray, at: float, sound: np.ndarray, gain: float) -> None:
    i = int(at * RATE)
    n = min(len(sound), len(mix) - i)
    if n > 0:
        mix[i:i + n] += gain * sound[:n]


@lru_cache(maxsize=1)
def samples() -> np.ndarray:
    rng = np.random.default_rng(5)
    mix = np.zeros(int(SECONDS * RATE))
    steps = int(SECONDS / _BEAT)
    for s in range(steps):
        at, chord = s * _BEAT, _CHORDS[(s // 8) % 4]
        if s % 8 == 7 and (s // 8) % 2 == 1:
            continue                                               # a short rest at the end of every second bar
        _add(mix, at, _pluck(chord[[0, 2, 1, 3, 2, 1, 3, 2][s % 8]] * 2, 1.4), 0.30)
        if s % 4 == 0:
            _add(mix, at, _pluck(chord[0] / 2, 1.1, bright=0.6), 0.45)        # bass
        if s % 8 == 0:
            for f in chord[:3]:
                _add(mix, at, _pluck(f, 2.2, bright=0.8), 0.16)               # the chord, held
        t = np.arange(int(0.08 * RATE)) / RATE
        hat = rng.normal(0, 1, len(t)) * np.exp(-t * 70)
        _add(mix, at, np.diff(hat, prepend=0.0), 0.10 if s % 2 else 0.05)     # hi-hat: bright noise
        if s % 4 == 2:
            t = np.arange(int(0.18 * RATE)) / RATE
            _add(mix, at, (rng.normal(0, 1, len(t)) * 0.7 + np.sin(2 * np.pi * 190 * t)) * np.exp(-t * 28), 0.16)   # snare
        if s % 4 == 0:
            t = np.arange(int(0.25 * RATE)) / RATE
            _add(mix, at, np.sin(2 * np.pi * (48 * t + 60 * (1 - np.exp(-t * 30)) / 30)) * np.exp(-t * 14), 0.5)    # kick
    mix = mix / np.abs(mix).max()
    # Room tone: every real recording has a noise floor (air, microphone, tape). Here about 45 dB down, tilted dark.
    hiss = np.cumsum(rng.normal(0, 1, len(mix))) * 0.02 + rng.normal(0, 1, len(mix))
    hiss -= np.convolve(hiss, np.ones(400) / 400, mode="same")
    mix += hiss / hiss.std() * mix.std() * 10 ** (-45 / 20)
    return mix / np.abs(mix).max() * 0.8
