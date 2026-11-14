"""A small procedural clip to experiment on: drifting clouds over hills and
rippling water, with fresh sensor grain on every frame. Made with NumPy and
OpenCV so the project needs no borrowed footage. It has what a video watermark
test needs: motion, flat areas, textured areas and frame-to-frame noise.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

W, H, FPS, SECONDS = 480, 320, 24, 3


def _fractal(rng: np.random.Generator, h: int, w: int, octaves: int = 6, base: int = 3) -> np.ndarray:
    out, amp, total = np.zeros((h, w)), 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        out += amp * cv2.resize(rng.random((n, max(2, n * w // h))), (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (out.max() - out.min())


@lru_cache(maxsize=1)
def frames() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(11)
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]
    horizon, hy = 0.62, int(0.62 * H)

    t = np.clip(y / horizon, 0, 1)
    sky = np.array([38, 74, 140.0]) + (np.array([250, 176, 120.0]) - np.array([38, 74, 140.0])) * (t ** 1.6)[..., None] * np.ones((1, W, 1))
    sun = np.exp(-(((x - 0.68) * W / H) ** 2 + (y - horizon + 0.05) ** 2) / 0.004)
    sky = sky + sun[..., None] * np.array([90, 70, 30.0])
    cloud_strip = np.clip((_fractal(rng, H, W * 2, 6, 2) - 0.5) * 3.2, 0, 1)     # twice as wide: it scrolls
    fade = np.clip(1 - y / horizon, 0, 1) ** 0.7

    hills = np.zeros((H, W, 3))
    hill_mask = np.zeros((H, W), dtype=bool)
    for i, (base, height, colour) in enumerate([(0.50, 0.16, (110, 118, 150)), (0.56, 0.13, (72, 84, 108)),
                                                (0.61, 0.10, (40, 54, 60))]):
        ridge = cv2.GaussianBlur(_fractal(rng, 2, W, 5, 2 + i), (0, 0), 6)[0]
        mask = np.arange(H)[:, None] >= ((base - height * ridge) * H).astype(int)[None, :]
        tex = (_fractal(rng, H, W, 7, 6) - 0.5) * (14 + 16 * i)
        hills = np.where(mask[..., None], np.array(colour, dtype=float) + tex[..., None], hills)
        hill_mask |= mask
    hill_mask &= np.arange(H)[:, None] < hy
    ripple_strip = (_fractal(rng, (H - hy) * 2, W, 5, 8) - 0.5) * 26               # twice as tall: it scrolls
    rows = np.clip((np.arange(H - hy) * (hy / (H - hy))).astype(int), 0, hy - 1)
    cols = np.arange(W)

    out = []
    for f in range(FPS * SECONDS):
        clouds = np.roll(cloud_strip, -2 * f, axis=1)[:, :W] * fade
        img = sky * (1 - 0.75 * clouds[..., None]) + np.array([252, 232, 222.0]) * 0.75 * clouds[..., None]
        img = np.where(hill_mask[..., None], hills, img)
        ripple = np.roll(ripple_strip, f, axis=0)[:H - hy]
        shift = (np.sin((np.arange(H - hy) + f * 0.7) / 3.0)[:, None] * 3 + ripple / 6).astype(int)
        water = img[:hy][::-1][rows[:, None], np.clip(cols[None, :] + shift, 0, W - 1)]
        img[hy:] = water * 0.72 + np.array([10, 24, 40.0]) + ripple[..., None]
        img = img + rng.normal(0, 3.0, img.shape)                                   # new grain every frame
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return tuple(out)
