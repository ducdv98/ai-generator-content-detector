"""The three things the video page does: make a marked clip, check a clip, and
test which edits each kind of mark survives."""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

from .. import signing
from ..detectors import provenance
from ..report import AI
from . import codec, container, frames as marks


def _jpeg(rgb: np.ndarray, quality: int = 88) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _psnr(a: list[np.ndarray], b: list[np.ndarray]) -> float | None:
    step = max(1, len(a) // 12)
    mse = float(np.mean([((x.astype(np.float64) - y) ** 2).mean() for x, y in zip(a[::step], b[::step])]))
    return None if mse == 0 else round(float(10 * np.log10(255.0 ** 2 / mse)), 1)


def make(src: list[np.ndarray], fps: float, label: bool, frame_mark: bool, key: str, payload: int,
         crf: int) -> dict:
    """Mark a clip and compress it. Returns the finished file plus a few
    stills: the frame before, the frame as it is in the finished file, and what
    the watermark added (amplified until it can be seen)."""
    h, w = src[0].shape[0] // 2 * 2, src[0].shape[1] // 2 * 2
    src = [f[:h, :w] for f in src]
    marked = marks.embed(src, payload, key) if frame_mark else src
    data = codec.encode(marked, fps, crf)
    if label:
        data = container.add_label(data)
    final = codec.decode(data)["frames"]

    diff = [m.astype(np.float64) - s for m, s in zip(marked[::max(1, len(src) // 6)], src[::max(1, len(src) // 6)])]
    typical = float(np.percentile(np.abs(np.stack(diff)), 98)) if frame_mark else 0.0
    gain = float(np.clip(110.0 / max(typical, 1.0), 2, 60)) if typical else 1.0
    stills = []
    for i in sorted({int(round(x)) for x in np.linspace(0, min(len(src), len(final)) - 1, 6)}):
        d = np.clip((marked[i].astype(np.float64) - src[i]) * gain + 128, 0, 255).astype(np.uint8)
        stills.append({"frame": i, "time": round(i / fps, 2), "before": _jpeg(src[i]), "after": _jpeg(final[i]),
                       "diff": _jpeg(d, 92)})
    return {"video": base64.b64encode(data).decode(), "bytes": len(data), "stills": stills,
            "width": w, "height": h, "fps": round(fps, 2), "frames": len(src), "seconds": round(len(src) / fps, 2),
            "gain": round(gain, 1), "crf": crf,
            "mark_psnr_db": _psnr(marked, src) if frame_mark else None,
            "file_psnr_db": _psnr(final, src[:len(final)]),
            "marks": {"label": label, "frames": frame_mark},
            "payload_hex": f"{payload:08X}" if frame_mark else None}


def check(data: bytes, key: str) -> dict:
    """Everything this project can read from one video file."""
    v = codec.decode(data)
    fr = v["frames"]
    res = marks.detect(fr, key)
    types = codec.frame_types(data)
    if types and len(types) >= len(fr) and res.get("per_frame"):
        res["types"] = types[:len(fr)]
        res["by_type"] = {t: {"frames": types[:len(fr)].count(t),
                              "mean": round(float(np.mean([z for z, k in zip(res["per_frame"], types) if k == t])), 1)}
                          for t in ("I", "P", "B") if t in types[:len(fr)]}
    return {"file": {"width": v["width"], "height": v["height"], "fps": round(v["fps"], 2), "codec": v["codec"],
                     "frames": v["total_frames"], "frames_used": len(fr), "cut_short": v["cut_short"],
                     "scaled_down": v["scaled_down"], "seconds": round(v["total_frames"] / v["fps"], 2),
                     "bytes": len(data)},
            "labels": container.labels(data), "frames": res, "poster": _jpeg(fr[len(fr) // 2]), "key": key}


def _resize(f: np.ndarray, factor: float) -> np.ndarray:
    return cv2.resize(f, (round(f.shape[1] * factor), round(f.shape[0] * factor)), interpolation=cv2.INTER_AREA)


def _crop(f: np.ndarray) -> np.ndarray:
    dh, dw = int(f.shape[0] * 0.1), int(f.shape[1] * 0.1)
    return f[dh:f.shape[0] - dh, dw:f.shape[1] - dw]


# name -> (what to do to the frames, frame-rate factor, quality to re-compress at)
EDITS = {
    "Re-encode, light compression": (lambda fr: fr, 1.0, 23),
    "Re-encode, typical upload": (lambda fr: fr, 1.0, 30),
    "Re-encode, heavy compression": (lambda fr: fr, 1.0, 36),
    "Scale to half size": (lambda fr: [_resize(f, 0.5) for f in fr], 1.0, 23),
    "Half the frame rate": (lambda fr: fr[::2], 0.5, 23),
    "Trim to the last third": (lambda fr: fr[-max(2, len(fr) // 3):], 1.0, 23),
    "Crop 10% off each edge": (lambda fr: [_crop(f) for f in fr], 1.0, 23),
    "Mirror left to right": (lambda fr: [np.ascontiguousarray(f[:, ::-1]) for f in fr], 1.0, 23),
    "Screen recording": (lambda fr: [cv2.GaussianBlur(_resize(f, 0.75), (0, 0), 0.6) for f in fr], 1.0, 28),
}


def survive(data: bytes, key: str) -> dict:
    """Apply each edit to the clip, compress it again, and re-run both checks
    on the new file. The file being edited has already been compressed once,
    so every row is at least a second generation, as shared video always is."""
    v = codec.decode(data)
    fr, fps = v["frames"], v["fps"]

    def row(name: str, file: bytes, frames_: list[np.ndarray], label: bool | None = None) -> dict:
        d = marks.detect(frames_, key)
        return {"edit": name, "label": container.labels(file)["verdict"] == "detected" if label is None else label,
                "z": round(d["z"], 1), "found": d["verdict"] == "detected",
                "alone": d.get("frames_detected_alone", 0), "frames": len(frames_),
                "median_single": round(d.get("median_single", 0.0), 1)}

    rows = [row("No edit", data, fr)]
    for name, (edit, rate, crf) in EDITS.items():
        out = codec.encode(edit(fr), fps * rate, crf)
        rows.append(row(name, out, codec.decode(out)["frames"]))

    still = fr[len(fr) // 2]                      # a frame grab is a picture file, so use the image label check
    buf = io.BytesIO()
    Image.fromarray(still).save(buf, "PNG")
    png = buf.getvalue()
    has = any(e.points_to == AI for e in provenance.analyze(png, Image.open(io.BytesIO(png))))
    rows.append(row("One frame saved as a picture", png, [still], label=has))
    return {"rows": rows, "threshold": signing.DETECT_Z}
