"""The three things the image page does: make a marked sample, check an image,
and test which edits each kind of mark survives."""
from __future__ import annotations

import base64
import io
from dataclasses import asdict

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .. import attacks
from ..detectors import provenance
from ..report import AI, NEUTRAL
from . import anatomy, keyed, public

MAX_SIDE = 1600   # samples are scaled down to this before marking, to keep the page quick

# The unsigned "made by AI" tag: IPTC DigitalSourceType inside an XMP packet.
XMP_AI = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
          '<rdf:Description rdf:about="" xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
          'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
          '</rdf:RDF></x:xmpmeta>')


def _png(rgb: np.ndarray, label: bool = False) -> bytes:
    info = PngInfo()
    if label:
        info.add_itxt("XML:com.adobe.xmp", XMP_AI)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _psnr(a: np.ndarray, b: np.ndarray) -> float | None:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return None if mse == 0 else round(float(10 * np.log10(255.0 ** 2 / mse)), 1)


def make(rgb: np.ndarray, label: bool, public_mark: bool, keyed_mark: bool,
         key: str, payload: int) -> dict:
    """Add the chosen marks to an image. Returns the before/after pictures and
    the difference between them, amplified until it can be seen."""
    h, w = rgb.shape[:2]
    if max(h, w) > MAX_SIDE:
        f = MAX_SIDE / max(h, w)
        rgb = cv2.resize(rgb, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    out = rgb
    if keyed_mark:
        out = keyed.embed(out, payload, key)
    if public_mark:            # last: it is the fragile one, nothing may touch the pixels after it
        out = public.embed(out, public.PAYLOADS[0]["bits"])

    diff = out.astype(np.float64) - rgb.astype(np.float64)
    # Scale so that typical changes are clearly visible; the few largest ones clip.
    typical = float(np.percentile(np.abs(diff), 98)) if diff.any() else 0.0
    gain = float(np.clip(110.0 / max(typical, 1.0), 2, 60)) if typical else 1.0
    shown = np.clip(diff * gain + 128, 0, 255).astype(np.uint8)
    return {
        "width": int(out.shape[1]), "height": int(out.shape[0]),
        "before_png": _b64(_png(rgb)), "after_png": _b64(_png(out, label)), "diff_png": _b64(_png(shown)),
        "gain": round(gain, 1), "psnr_db": _psnr(rgb, out),
        "changed_pixels": round(float((np.abs(diff).max(axis=2) > 0).mean()), 4),
        "largest_change": int(np.abs(diff).max()),
        "marks": {"label": label, "public": public_mark, "keyed": keyed_mark},
        "payload_hex": f"{payload:08X}" if keyed_mark else None,
    }


def _labels(data: bytes, img: Image.Image) -> dict:
    ev = provenance.analyze(data, img)
    ai = [e for e in ev if e.points_to == AI]
    said = [e for e in ev if e.points_to != NEUTRAL]
    blocks = anatomy.parts(data)
    return {"verdict": "detected" if ai else "other" if said else "none",
            "headline": (ai or said or ev)[0].finding,
            "findings": [asdict(e) for e in ev], "parts": blocks,
            "label_bytes": sum(p["bytes"] for p in blocks if p["kind"] == anatomy.LABEL),
            "total_bytes": len(data)}


def check(data: bytes, img: Image.Image, rgb: np.ndarray, key: str) -> dict:
    """Everything this project can read from one image file."""
    return {"file": {"format": img.format, "width": img.width, "height": img.height, "bytes": len(data)},
            "labels": _labels(data, img), "public": public.detect(rgb), "keyed": keyed.detect(rgb, key),
            "key": key}


def survive(data: bytes, img: Image.Image, rgb: np.ndarray, key: str) -> dict:
    """Apply each common edit and re-run all three checks on the result.

    Every edit here re-encodes the picture, as uploading to a social platform,
    pasting into a chat or taking a screenshot does. The label check is really
    run on the edited file, not assumed.
    """
    rows = []
    which = None          # the public message found in the unedited image; edits are scored against it
    for name, edit in attacks.ATTACKS.items():
        if name == "No edit":
            d, im, px = data, img, rgb
        else:
            px = np.ascontiguousarray(edit(rgb))
            d = _png(px)
            im = Image.open(io.BytesIO(d))
        lab = any(e.points_to == AI for e in provenance.analyze(d, im))
        pub = public.detect(px)
        if which is None:
            which = pub.get("best", 0)
        best = pub["candidates"][which] if pub["candidates"] else None
        ky = keyed.detect(px, key)
        rows.append({"edit": name, "label": lab,
                     "public": {"found": bool(best) and best["chance"] <= public.MAX_CHANCE,
                                "matched": best["matched"] if best else 0, "n": best["n"] if best else 0},
                     "keyed": {"found": ky["verdict"] == "detected", "z": round(ky["z"], 1)}})
    return {"rows": rows, "threshold": keyed.s.DETECT_Z}
