"""Common edits used to show what a watermark survives (and what it doesn't)."""
from __future__ import annotations

import io
from typing import Callable

import cv2
import numpy as np
from PIL import Image


def _jpeg(q: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(rgb: np.ndarray) -> np.ndarray:
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=q)
        return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return f


def _resize(factor: float) -> Callable[[np.ndarray], np.ndarray]:
    def f(rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        size = (max(1, round(w * factor)), max(1, round(h * factor)))
        interp = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
        return cv2.resize(rgb, size, interpolation=interp)
    return f


def _noise(rgb: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.clip(rgb + rng.normal(0, 5, rgb.shape), 0, 255).astype(np.uint8)


def _blur(rgb: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(rgb, (0, 0), 1.0)


def _brightness(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb.astype(np.float64) * 1.15 + 10, 0, 255).astype(np.uint8)


def _screenshot(rgb: np.ndarray) -> np.ndarray:
    """Displayed at 75% size, then saved as a medium-quality JPEG."""
    return _jpeg(70)(_resize(0.75)(rgb))


def _crop(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    dh, dw = int(h * 0.1), int(w * 0.1)
    return rgb[dh:h - dh, dw:w - dw]


ATTACKS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "No edit": lambda x: x,
    "JPEG quality 90": _jpeg(90),
    "JPEG quality 70": _jpeg(70),
    "JPEG quality 50": _jpeg(50),
    "JPEG quality 30": _jpeg(30),
    "Resize to 50%": _resize(0.5),
    "Resize to 200%": _resize(2.0),
    "Resize 75% + JPEG 70": _screenshot,
    "Gaussian blur": _blur,
    "Added noise": _noise,
    "Brightness / contrast": _brightness,
    "Crop 10% off each edge": _crop,
}
