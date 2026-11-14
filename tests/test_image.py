"""Image watermark lab tests."""
import base64
import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.image import anatomy, keyed, lab, public
from app.main import app

FIX = Path(__file__).parent / "fixtures"
SD1 = public.PAYLOADS[0]["bits"]


def _textured(seed: int = 0, size: int = 512) -> np.ndarray:
    return np.clip(np.random.default_rng(seed).normal(128, 40, (size, size, 3)), 0, 255).astype(np.uint8)


def test_public_watermark_round_trip_and_no_false_alarm():
    rgb = _textured()
    res = public.detect(public.embed(rgb, SD1))
    assert res["verdict"] == "detected" and res["candidates"][res["best"]]["name"] == "Stable Diffusion 1.x"
    assert res["candidates"][res["best"]]["matched"] >= 130
    assert all(public.detect(_textured(s))["verdict"] == "none" for s in range(1, 6))


def test_public_watermark_matches_reference_library():
    """Cross-check the port against invisible-watermark itself, when it is installed."""
    for name in ("torch", "torch.nn", "torch.nn.functional"):     # only its RivaGAN method needs torch
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    if not hasattr(sys.modules["torch.nn"], "Module"):
        sys.modules["torch.nn"].Module = object
    imw = pytest.importorskip("imwatermark")
    rgb = _textured()
    enc = imw.WatermarkEncoder()
    enc.set_watermark("bytes", b"StableDiffusionV1")
    ref = np.ascontiguousarray(enc.encode(np.ascontiguousarray(rgb[..., ::-1]).copy(), "dwtDct")[..., ::-1])
    assert (public.decode(public.votes(ref), 136)[0] == SD1).sum() >= 130          # we read theirs
    mine = np.ascontiguousarray(public.embed(rgb, SD1)[..., ::-1])
    assert imw.WatermarkDecoder("bytes", 136).decode(mine, "dwtDct") == b"StableDiffusionV1"   # they read ours


def test_keyed_watermark_needs_the_right_key():
    marked = keyed.embed(_textured(), 0xC0FFEE42, "secret")
    good = keyed.detect(marked, "secret")
    assert good["verdict"] == "detected" and good["payload_hex"] == "C0FFEE42"
    assert min(min(row) for row in good["map"]["cells"]) > 0
    assert keyed.detect(marked, "other")["verdict"] == "none"
    assert keyed.detect(_textured(), "secret")["verdict"] == "none"


def test_file_anatomy_finds_the_label_blocks():
    jpg = anatomy.parts((FIX / "ai_signed.jpg").read_bytes())
    assert [p["name"] for p in jpg if p["kind"] == "label"] == ["C2PA Content Credentials"]
    assert sum(p["bytes"] for p in jpg) == (FIX / "ai_signed.jpg").stat().st_size
    png = anatomy.parts(lab._png(_textured(size=64), label=True))
    assert any(p["kind"] == "label" and "XMP" in p["name"] for p in png)
    assert not any(p["kind"] == "label" for p in anatomy.parts(lab._png(_textured(size=64))))


def test_make_check_and_survive_through_the_api():
    client = TestClient(app)
    buf = io.BytesIO()
    Image.fromarray(_textured()).save(buf, "PNG")
    made = client.post("/api/image/make", files={"file": ("a.png", buf.getvalue(), "image/png")},
                       data={"key": "k1", "payload": "0000BEEF"}).json()
    marked = base64.b64decode(made["after_png"])
    files = {"file": ("m.png", marked, "image/png")}

    res = client.post("/api/image/check", files=files, data={"key": "k1"}).json()
    assert (res["labels"]["verdict"], res["public"]["verdict"], res["keyed"]["verdict"]) == ("detected",) * 3
    assert res["keyed"]["payload_hex"] == "0000BEEF"
    assert client.post("/api/image/check", files=files, data={"key": "k2"}).json()["keyed"]["verdict"] == "none"

    rows = client.post("/api/image/survive", files=files, data={"key": "k1"}).json()["rows"]
    assert rows[0]["label"] and rows[0]["public"]["found"] and rows[0]["keyed"]["found"]
    assert not any(r["label"] for r in rows[1:])                      # a re-encode drops the label
    jpeg = next(r for r in rows if r["edit"] == "JPEG quality 70")
    assert jpeg["keyed"]["found"] and not jpeg["public"]["found"]
