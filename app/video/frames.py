"""A keyed watermark in every frame, detected by pooling frames.

Writing is the image watermark from step two, applied to each frame with the
same key and the same message. Using the *same* pattern in every frame is the
design choice that matters:

  * No timing to recover. Drop frames, trim the clip, change the speed: every
    frame that is left still carries the whole mark.
  * Frames can be pooled. Average many frames and whatever changes from frame
    to frame (motion, grain, the codec's rounding errors) blurs out, while the
    pattern, which does not change, stays. The gain is real but modest, because
    neighbouring frames are nearly the same picture: 72 frames are nowhere near
    72 independent pieces of evidence, unlike the tokens of the text step.

Deployed systems make the same choice. Google applies SynthID to each frame of
Veo output; Meta's Video Seal marks one frame in every few and copies that
pattern to the frames between ("temporal propagation") to save computing time.

Detection here averages the frames' DCT values first and runs the image
detector once on the average. Adding up per-frame scores instead would be
wrong: neighbouring frames are nearly identical, so their scores are not
independent, and a still video would turn one frame's luck into a "detection".

Known weakness of a fixed pattern: averaging also helps an attacker. Averaging
frames from many different marked clips cancels the pictures and leaves the
pattern, which can then be subtracted.
"""
from __future__ import annotations

import numpy as np

from .. import signing as s
from ..image import keyed

STRENGTH = 9.0   # stronger than for still images (5.0): the codec removes much of it again


def embed(frames: list[np.ndarray], payload: int, key: str, strength: float = STRENGTH) -> list[np.ndarray]:
    return [s.embed(f, payload, key=key, strength=strength).image for f in frames]


def detect(frames: list[np.ndarray], key: str) -> dict:
    """Score each frame alone, and the clip as a whole with frames pooled.

    Returns the pooled detection (same fields as the image detector) plus
    `per_frame` (each frame's own score) and `building` (the pooled score using
    only the first 1, 2, 3 ... frames).
    """
    if min(frames[0].shape[:2]) < 64:
        return {"verdict": "too_small", "z": 0.0, "threshold": s.DETECT_Z, "per_frame": [], "building": []}
    total = None
    per_frame, building = [], []
    for i, frame in enumerate(frames, 1):
        c = keyed.coefficients(frame)
        total = c.copy() if total is None else total + c
        per_frame.append(round(keyed.detect_coefficients(c, key)["z"], 2))
        building.append(round(keyed.detect_coefficients(total / i, key)["z"], 2))
    res = keyed.detect_coefficients(total / len(frames), key)
    res.update(per_frame=per_frame, building=building, frames=len(frames),
               frames_detected_alone=int(sum(z >= s.DETECT_Z for z in per_frame)),
               best_single=max(per_frame), median_single=float(np.median(per_frame)))
    return res
