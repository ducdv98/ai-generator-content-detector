"""End-to-end tests. Run with:  pytest -q"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app import signing
from app.detectors import classifier
from app.main import app

FIX = Path(__file__).parent / "fixtures"
client = TestClient(app)


def photo(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """Synthetic photo-like image: smooth gradients plus texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([120 + 80 * np.sin(xx / 60 + seed), 110 + 70 * np.cos(yy / 45),
                     100 + 60 * np.sin((xx + yy) / 80)], axis=-1)
    tex = rng.normal(0, 12, (h // 4, w // 4, 3)).repeat(4, 0).repeat(4, 1)
    return np.clip(base + tex + rng.normal(0, 3, (h, w, 3)), 0, 255).astype(np.uint8)


def encode(rgb: np.ndarray, fmt: str = "PNG", **kw) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, fmt, **kw)
    return buf.getvalue()


def analyze(data: bytes, name: str = "x.png") -> dict:
    r = client.post("/api/analyze", files={"file": (name, data)})
    assert r.status_code == 200, r.text
    return r.json()


def checks(report: dict) -> dict[str, dict]:
    return {e["check"]: e for e in report["evidence"]}


# ------------------------------------------------------------------ layer 1

def test_clean_image_has_no_evidence():
    rep = analyze(encode(photo()))
    assert rep["verdict"] == "No evidence of AI generation"
    assert rep["confidence"] == "none"


def test_c2pa_ai_declaration():
    rep = analyze((FIX / "ai_signed.jpg").read_bytes(), "ai.jpg")
    assert rep["verdict"] == "AI-generated" and rep["confidence"] == "declared"
    ev = checks(rep)["C2PA Content Credentials"]
    assert ev["details"]["validation_state"] == "Valid"
    assert "C2PA Test Signing Cert" in ev["finding"]


def test_c2pa_tampered_is_flagged_but_not_declared():
    rep = analyze((FIX / "tampered.jpg").read_bytes(), "t.jpg")
    ev = checks(rep)["C2PA Content Credentials"]
    assert ev["details"]["validation_state"] == "Invalid"
    assert "altered after signing" in ev["finding"]
    assert rep["confidence"] == "high"


def test_c2pa_without_ai_source_is_neutral():
    rep = analyze((FIX / "c2pa_non_ai.jpg").read_bytes(), "c.jpg")
    ev = checks(rep)["C2PA Content Credentials"]
    assert ev["points_to"] == "neutral"
    assert rep["verdict"] == "No evidence of AI generation"


def test_iptc_digital_source_type():
    xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/'
           b'22-rdf-syntax-ns#"><rdf:Description xmlns:Iptc4xmpExt="http://iptc.org/std/'
           b'Iptc4xmpExt/2008-02-29/" Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/'
           b'digitalsourcetype/trainedAlgorithmicMedia"/></rdf:RDF></x:xmpmeta>')
    rep = analyze(encode(photo(), "JPEG", xmp=xmp), "x.jpg")
    assert rep["verdict"] == "AI-generated" and rep["confidence"] == "high"
    assert "IPTC DigitalSourceType" in checks(rep)


def test_stable_diffusion_webui_png_chunk():
    info = PngInfo()
    info.add_text("parameters", "a cat\nSteps: 20, Sampler: Euler a, CFG scale: 7")
    rep = analyze(encode(photo(), pnginfo=info))
    assert rep["verdict"] == "AI-generated"
    assert "AUTOMATIC1111" in checks(rep)["Generator metadata"]["finding"]


def test_comfyui_prompt_must_be_json():
    info = PngInfo()
    info.add_text("prompt", "just a caption someone typed")
    assert analyze(encode(photo(), pnginfo=info))["verdict"] == "No evidence of AI generation"
    info = PngInfo()
    info.add_text("prompt", '{"3": {"class_type": "KSampler"}}')
    assert analyze(encode(photo(), pnginfo=info))["verdict"] == "AI-generated"


def test_exif_software_and_camera():
    exif = Image.Exif()
    exif[0x0131] = "Midjourney v6"
    rep = analyze(encode(photo(), "JPEG", exif=exif), "x.jpg")
    assert rep["verdict"] == "AI-generated"

    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = "Canon", "EOS R6"
    exif.get_ifd(0x8769)[0x829A] = (1, 250)
    rep = analyze(encode(photo(), "JPEG", exif=exif), "x.jpg")
    cam = checks(rep)["Camera EXIF"]
    assert cam["points_to"] == "not_ai" and cam["strength"] == "weak"
    assert rep["verdict"] == "No evidence of AI generation"


# ------------------------------------------------------------------ layer 2

@pytest.mark.parametrize("label,kind,payload", [
    ("Stable Diffusion 1.x", "bytes", b"StableDiffusionV1"),
    ("Stable Diffusion 2.x", "bytes", b"SDV2"),
    ("Stable Diffusion XL", "bits",
     [int(c) for c in bin(0b101100111110110010010000011110111011000110011110)[2:]]),
])
def test_stable_diffusion_watermark(label, kind, payload):
    WatermarkEncoder = pytest.importorskip("imwatermark").WatermarkEncoder
    enc = WatermarkEncoder()
    enc.set_watermark(kind, payload)
    bgr = enc.encode(np.ascontiguousarray(photo(512, 512)[..., ::-1]), "dwtDct")
    rep = analyze(encode(np.ascontiguousarray(bgr[..., ::-1])))
    ev = checks(rep)["Stable Diffusion watermark"]
    assert ev["points_to"] == "ai", ev
    assert ev["details"]["generator"] == label
    assert rep["verdict"] == "AI-generated" and rep["confidence"] == "high"


def test_own_watermark_roundtrip_and_key_secrecy():
    rgb = photo()
    res = signing.embed(rgb, 0xDEADBEEF)
    assert res.psnr_db > 38
    assert not signing.detect(rgb).present
    d = signing.detect(res.image)
    assert d.present and d.payload == 0xDEADBEEF
    assert not signing.detect(res.image, key="some-other-key").present
    jpeg = np.asarray(Image.open(io.BytesIO(encode(res.image, "JPEG", quality=50))))
    d = signing.detect(jpeg)
    assert d.present and d.payload == 0xDEADBEEF


def test_sign_then_analyze_via_api():
    r = client.post("/api/sign", files={"file": ("p.png", encode(photo(seed=3)))},
                    data={"payload": "0BADF00D"})
    assert r.status_code == 200, r.text
    body = r.json()
    rows = {x["edit"]: x for x in body["robustness"]}
    assert rows["JPEG quality 50"]["payload_recovered"]
    assert rows["Resize to 50%"]["payload_recovered"]
    assert not rows["Crop 10% off each edge"]["detected"]  # documented limit

    signed = base64.b64decode(body["image_png_base64"])
    rep = analyze(signed)
    assert checks(rep)["This service's watermark"]["details"]["payload_hex"] == "0BADF00D"
    v = client.post("/api/verify", files={"file": ("s.png", signed)}).json()
    assert v["present"] and v["payload_hex"] == "0BADF00D"


# ------------------------------------------------------------------ layer 3

def test_label_mapping():
    assert classifier.ai_probability({"artificial": 0.9, "human": 0.1}) == pytest.approx(0.9)
    assert classifier.ai_probability({"FAKE": 0.2, "REAL": 0.8}) == pytest.approx(0.2)
    assert classifier.ai_probability({"real": 0.3, "dalle": 0.5, "midj": 0.2}) == pytest.approx(0.7)
    assert classifier.ai_probability({"LABEL_0": 0.5, "LABEL_1": 0.5}) is None


def test_classifier_hf_and_onnx_paths(tmp_path, monkeypatch):
    """Plumbing test with a tiny randomly initialised model (no download)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import subprocess
    import sys

    cfg = transformers.ViTConfig(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=64,
                                 id2label={0: "artificial", 1: "human"},
                                 label2id={"artificial": 0, "human": 1})
    torch.manual_seed(0)
    model_dir = tmp_path / "tiny"
    transformers.ViTForImageClassification(cfg).save_pretrained(model_dir)
    transformers.ViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(model_dir)

    data = encode(photo())
    monkeypatch.setenv("AIGC_CLASSIFIER_MODEL", str(model_dir))
    classifier.reset()
    ev = checks(analyze(data))["Forensic classifier"]
    assert ev["strength"] == "statistical", ev
    p_hf = ev["details"]["ai_probability"]

    onnx_dir = tmp_path / "onnx"
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "export_onnx.py"),
                    str(model_dir), str(onnx_dir)], check=True, cwd=root)
    monkeypatch.setenv("AIGC_CLASSIFIER_ONNX", str(onnx_dir))
    classifier.reset()
    ev = checks(analyze(data))["Forensic classifier"]
    assert ev["details"]["model"].startswith("ONNX")
    assert ev["details"]["ai_probability"] == pytest.approx(p_hf, abs=0.02)
    classifier.reset()


# ------------------------------------------------------------------ API edges

def test_rejects_non_image_and_bad_payload():
    assert client.post("/api/analyze", files={"file": ("a.txt", b"hello")}).status_code == 415
    r = client.post("/api/sign", files={"file": ("p.png", encode(photo()))}, data={"payload": "xyz"})
    assert r.status_code == 400


def test_health_and_index():
    assert client.get("/api/health").json()["status"] == "ok"
    assert "<html" in client.get("/").text.lower()
