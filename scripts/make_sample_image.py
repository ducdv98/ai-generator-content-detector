"""Draw the sample picture used by the image-watermark page.

    python scripts/make_sample_image.py

A procedural landscape (sky, clouds, hills, water) made with NumPy and OpenCV,
so the project ships a test image without downloading or borrowing one. It has
smooth areas and textured areas, and colour grain like a camera's, which is
what a watermark demo needs: marks hide in texture and struggle in flat sky.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

W, H = 960, 640
OUT = Path(__file__).resolve().parents[1] / "web" / "samples" / "landscape.png"


def fractal(rng: np.random.Generator, h: int, w: int, octaves: int = 6, base: int = 3) -> np.ndarray:
    """Smooth multi-scale noise in 0..1."""
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        layer = cv2.resize(rng.random((n, max(2, n * w // h))), (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * layer
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (out.max() - out.min())


def main() -> None:
    rng = np.random.default_rng(7)
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]
    horizon = 0.62

    # Sky: deep blue at the top, warm near the horizon, with a low sun and clouds.
    t = np.clip(y / horizon, 0, 1)
    top, low = np.array([38, 74, 140.0]), np.array([250, 176, 120.0])
    img = top + (low - top) * (t ** 1.6)[..., None] * np.ones((1, W, 1))
    sun = np.exp(-(((x - 0.68) * W / H) ** 2 + (y - horizon + 0.05) ** 2) / 0.004)
    img += sun[..., None] * np.array([90, 70, 30.0])
    clouds = np.clip((fractal(rng, H, W, 6, 2) - 0.52) * 3.2, 0, 1) * np.clip(1 - y / horizon, 0, 1) ** 0.7
    img = img * (1 - 0.75 * clouds[..., None]) + np.array([252, 232, 222.0]) * 0.75 * clouds[..., None]

    # Three ranges of hills, nearer ones darker and more textured.
    cols = np.arange(W)
    for i, (base, height, colour) in enumerate([(0.50, 0.16, (110, 118, 150)), (0.56, 0.13, (72, 84, 108)),
                                                (0.61, 0.10, (40, 54, 60))]):
        ridge = fractal(rng, 2, W, 6, 2 + i)[0]
        top_y = ((base - height * ridge) * H).astype(int)
        mask = np.arange(H)[:, None] >= top_y[None, :]
        tex = (fractal(rng, H, W, 7, 6) - 0.5) * (14 + 16 * i)
        layer = np.array(colour, dtype=float)[None, None, :] + tex[..., None]
        img = np.where(mask[..., None] & (np.arange(H)[:, None, None] < horizon * H), layer, img)

    # Water: the scene above mirrored, darkened and rippled.
    hy = int(horizon * H)
    mirror = img[:hy][::-1]
    rows = np.clip((np.arange(H - hy) * (hy / (H - hy))).astype(int), 0, hy - 1)
    ripple = (fractal(rng, H - hy, W, 5, 8) - 0.5) * 26
    shift = (np.sin(np.arange(H - hy) / 3.0)[:, None] * 3 + ripple / 6).astype(int)
    water = mirror[rows[:, None], np.clip(cols[None, :] + shift, 0, W - 1)]
    img[hy:] = water * 0.72 + np.array([10, 24, 40.0]) + ripple[..., None]

    img += rng.normal(0, 4.0, img.shape)          # sensor-like grain, in colour as well as brightness
    OUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(OUT)
    print(OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
