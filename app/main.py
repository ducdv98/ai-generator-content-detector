"""FastAPI service: layered AI-content detection + watermark sign/verify demo."""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from . import attacks, signing
from .detectors import classifier, provenance, watermark
from .limits import Limits
from .report import build_report
from .audio import codec as audio_codec
from .audio import lab as audio_lab
from .audio import mark as audio_mark
from .audio import sample as audio_sample
from .image import lab as image_lab
from .video import codec as video_codec
from .video import lab as video_lab
from .video import sample as video_sample
from .text import analyze as text_analyze
from .text import lm as text_lm

MAX_BYTES = int(os.environ.get("AIGC_MAX_UPLOAD_MB", "20")) * 1024 * 1024
MAX_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
WEB = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title="AI-generated content detector", version="0.1.0")
app.add_middleware(Limits)


async def _read(file: UploadFile) -> bytes:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    if not data:
        raise HTTPException(400, "Empty file.")
    return data


def _open(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(415, "Not a readable image. Supported: JPEG, PNG, WebP and other "
                                 "formats Pillow can open.")
    if img.width * img.height > MAX_PIXELS:
        raise HTTPException(413, "Image has too many pixels.")
    return img


def _rgb(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    return np.asarray(img.convert("RGB"))


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "classifier_configured": bool(classifier.configured()),
        "using_dev_watermark_key": signing.using_dev_key(),
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    data = await _read(file)
    img = _open(data)
    rgb = _rgb(img)

    evidence = provenance.analyze(data, img)
    evidence += watermark.analyze(rgb)
    evidence += classifier.analyze(Image.fromarray(rgb))

    report = build_report(evidence).to_dict()
    report["file"] = {"name": file.filename, "format": img.format,
                      "width": img.width, "height": img.height, "bytes": len(data)}
    return report


def _parse_payload(text: str) -> int:
    t = text.strip().lower().removeprefix("0x")
    try:
        value = int(t, 16)
    except ValueError:
        raise HTTPException(400, "Payload ID must be up to 8 hex digits, e.g. C0FFEE42.")
    if not 0 <= value < 2 ** signing.PAYLOAD_BITS:
        raise HTTPException(400, "Payload ID must be up to 8 hex digits.")
    return value


@app.post("/api/sign")
async def sign(file: UploadFile = File(...), payload: str = Form("C0FFEE42")) -> dict:
    """Embed the invisible watermark, then report which edits it survives."""
    value = _parse_payload(payload)
    rgb = _rgb(_open(await _read(file)))
    try:
        res = signing.embed(rgb, value)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    rows = []
    for name, fn in attacks.ATTACKS.items():
        d = signing.detect(fn(res.image))
        rows.append({"edit": name, "detected": d.present, "score": round(d.z_score, 1),
                     "payload_recovered": bool(d.present and d.payload == value)})

    buf = io.BytesIO()
    Image.fromarray(res.image).save(buf, "PNG")
    return {
        "payload_hex": f"{value:08X}",
        "psnr_db": round(res.psnr_db, 1),
        "threshold": signing.DETECT_Z,
        "image_png_base64": base64.b64encode(buf.getvalue()).decode(),
        "robustness": rows,
        "using_dev_key": signing.using_dev_key(),
    }


@app.post("/api/verify")
async def verify(file: UploadFile = File(...)) -> dict:
    rgb = _rgb(_open(await _read(file)))
    d = signing.detect(rgb)
    return {"present": d.present, "score": round(d.z_score, 1), "threshold": d.threshold,
            "payload_hex": f"{d.payload:08X}" if d.payload is not None else None}


# ---------------------------------------------------------------- text ------

class TextIn(BaseModel):
    text: str = Field(max_length=200_000)
    key: str = Field("demo-key", min_length=1, max_length=200)


class GenerateIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    scheme: Literal["none", "greenlist", "tournament"] = "greenlist"
    key: str = Field("demo-key", min_length=1, max_length=200)
    delta: float = Field(2.0, ge=0, le=10)
    max_new_tokens: int = Field(200, ge=20, le=400)
    seed: int | None = None


@app.get("/api/text/status")
def text_status() -> dict:
    return {"tokenizer_ready": text_lm.tokenizer_ready(), "model_ready": text_lm.model_ready(),
            "model": text_lm.MODEL_NAME, "max_chars": text_analyze.MAX_CHARS}


@app.post("/api/text/analyze")
def text_analyze_route(body: TextIn) -> dict:
    if not text_lm.tokenizer_ready():
        raise HTTPException(503, "Tokenizer not found. Run: python scripts/download_text_model.py")
    return text_analyze.analyze(body.text, body.key)


@app.post("/api/text/generate")
def text_generate(body: GenerateIn) -> StreamingResponse:
    """Stream a reply as newline-delimited JSON: {"t": "piece"} ... {"done": true}."""
    if not text_lm.model_ready():
        raise HTTPException(503, "Model not found. Run: python scripts/download_text_model.py")

    def stream():
        try:
            for piece in text_lm.generate(body.prompt, body.scheme, body.key, body.delta,
                                          body.max_new_tokens, seed=body.seed):
                yield json.dumps({"t": piece}) + "\n"
            yield json.dumps({"done": True}) + "\n"
        except Exception as exc:  # surface model errors to the page instead of a cut stream
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/text")
def text_page() -> FileResponse:
    return FileResponse(WEB / "text.html")


# --------------------------------------------------------------- image lab ---

def _lab_key(key: str) -> str:
    key = key.strip()
    if not 1 <= len(key) <= 200:
        raise HTTPException(400, "Key must be 1 to 200 characters.")
    return key


@app.post("/api/image/make")
async def image_make(file: UploadFile = File(...), label: bool = Form(True), public: bool = Form(True),
                     keyed: bool = Form(True), key: str = Form("demo-key"),
                     payload: str = Form("C0FFEE42")) -> dict:
    """Add the chosen marks to an image; return before, after and the amplified difference."""
    rgb = _rgb(_open(await _read(file)))
    try:
        return image_lab.make(rgb, label, public, keyed, _lab_key(key), _parse_payload(payload))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/image/check")
async def image_check(file: UploadFile = File(...), key: str = Form("demo-key")) -> dict:
    data = await _read(file)
    img = _open(data)
    return image_lab.check(data, img, _rgb(img), _lab_key(key))


@app.post("/api/image/survive")
async def image_survive(file: UploadFile = File(...), key: str = Form("demo-key")) -> dict:
    data = await _read(file)
    img = _open(data)
    return image_lab.survive(data, img, _rgb(img), _lab_key(key))


@app.get("/image")
def image_page() -> FileResponse:
    return FileResponse(WEB / "image.html")


# --------------------------------------------------------------- video lab ---

VIDEO_MAX_BYTES = int(os.environ.get("AIGC_MAX_VIDEO_MB", "80")) * 1024 * 1024
VIDEO_QUALITY = {"light": 20, "typical": 28, "heavy": 34}       # ffmpeg crf for the three choices on the page


async def _read_video(file: UploadFile) -> bytes:
    data = await file.read(VIDEO_MAX_BYTES + 1)
    if len(data) > VIDEO_MAX_BYTES:
        raise HTTPException(413, f"File is larger than {VIDEO_MAX_BYTES // (1024 * 1024)} MB.")
    if not data:
        raise HTTPException(400, "Empty file.")
    return data


def _need_ffmpeg() -> None:
    if video_codec.ffmpeg_path() is None:
        raise HTTPException(503, "ffmpeg is not installed or not on PATH. It is needed to write video; "
                                 "checking a clip works without it.")


@app.get("/api/video/status")
def video_status() -> dict:
    return {"ffmpeg": video_codec.ffmpeg_path() is not None, "max_frames": video_codec.MAX_FRAMES,
            "max_side": video_codec.MAX_SIDE}


@app.get("/api/video/sample")
def video_sample_clip() -> Response:
    """The built-in clip with no marks, lightly compressed."""
    _need_ffmpeg()
    return Response(video_codec.encode(list(video_sample.frames()), video_sample.FPS, 18), media_type="video/mp4")


@app.post("/api/video/make")
async def video_make(file: UploadFile | None = File(None), label: bool = Form(True), frames: bool = Form(True),
                     key: str = Form("demo-key"), payload: str = Form("C0FFEE42"),
                     quality: Literal["light", "typical", "heavy"] = Form("typical")) -> dict:
    """Mark the uploaded clip (or the built-in sample) and compress it."""
    _need_ffmpeg()
    key, value = _lab_key(key), _parse_payload(payload)
    if file is None:
        src, fps = list(video_sample.frames()), float(video_sample.FPS)
    else:
        try:
            v = video_codec.decode(await _read_video(file))
        except ValueError as exc:
            raise HTTPException(415, str(exc))
        src, fps = v["frames"], v["fps"]
    try:
        return video_lab.make(src, fps, label, frames, key, value, VIDEO_QUALITY[quality])
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/video/check")
async def video_check(file: UploadFile = File(...), key: str = Form("demo-key")) -> dict:
    try:
        return video_lab.check(await _read_video(file), _lab_key(key))
    except ValueError as exc:
        raise HTTPException(415, str(exc))


@app.post("/api/video/survive")
async def video_survive(file: UploadFile = File(...), key: str = Form("demo-key")) -> dict:
    _need_ffmpeg()
    try:
        return video_lab.survive(await _read_video(file), _lab_key(key))
    except ValueError as exc:
        raise HTTPException(415, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))


@app.get("/video")
def video_page() -> FileResponse:
    return FileResponse(WEB / "video.html")


# --------------------------------------------------------------- audio lab ---

def _audio_id(text: str) -> int:
    try:
        value = int(text.strip().lower().removeprefix("0x"), 16)
    except ValueError:
        raise HTTPException(400, "ID must be up to 4 hex digits, e.g. BEEF.")
    if not 0 <= value < 2 ** audio_mark.PAYLOAD_BITS:
        raise HTTPException(400, "ID must be up to 4 hex digits.")
    return value


def _audio_errors(run):
    try:
        return run()
    except ValueError as exc:
        raise HTTPException(415, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.get("/api/audio/status")
def audio_status() -> dict:
    return {"ffmpeg": audio_codec.ffmpeg_path() is not None, "make_seconds": audio_lab.MAKE_SECONDS,
            "max_seconds": audio_codec.MAX_SECONDS}


@app.get("/api/audio/sample")
def audio_sample_clip() -> Response:
    """The built-in tune with no marks."""
    return Response(audio_codec.wav_bytes(audio_sample.samples()), media_type="audio/wav")


@app.post("/api/audio/make")
async def audio_make(file: UploadFile | None = File(None), label: bool = Form(True), mark: bool = Form(True),
                     key: str = Form("demo-key"), payload: str = Form("BEEF"),
                     strength: Literal["subtle", "normal", "strong"] = Form("normal")) -> dict:
    """Mark the uploaded clip (or the built-in tune)."""
    key, value = _lab_key(key), _audio_id(payload)
    data = await _read_video(file) if file is not None else None
    return _audio_errors(lambda: audio_lab.make(
        audio_codec.decode(data)["samples"] if data is not None else audio_sample.samples(),
        label, mark, key, value, strength))


@app.post("/api/audio/check")
async def audio_check(file: UploadFile = File(...), key: str = Form("demo-key")) -> dict:
    data, key = await _read_video(file), _lab_key(key)
    return _audio_errors(lambda: audio_lab.check(data, key))


@app.post("/api/audio/survive")
async def audio_survive(file: UploadFile = File(...), key: str = Form("demo-key")) -> dict:
    data, key = await _read_video(file), _lab_key(key)
    return _audio_errors(lambda: audio_lab.survive(data, key))


@app.get("/audio")
def audio_page() -> FileResponse:
    return FileResponse(WEB / "audio.html")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
