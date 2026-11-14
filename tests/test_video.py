"""Video watermark lab tests. The codec tests need ffmpeg and are skipped without it."""
import base64
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.video import codec, container, frames, sample

needs_ffmpeg = pytest.mark.skipif(codec.ffmpeg_path() is None, reason="ffmpeg not installed")


def _clip(n: int = 12, seed: int = 0) -> list[np.ndarray]:
    """Textured frames that drift a little, with fresh noise on each."""
    rng = np.random.default_rng(seed)
    base = np.clip(rng.normal(128, 40, (256, 320, 3)), 0, 255)
    return [np.clip(np.roll(base, i, axis=1) + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8) for i in range(n)]


def test_frame_watermark_needs_the_right_key():
    marked = frames.embed(_clip(), 0x0000BEEF, "secret")
    good = frames.detect(marked, "secret")
    assert good["verdict"] == "detected" and good["payload_hex"] == "0000BEEF"
    assert good["frames_detected_alone"] == len(marked) and len(good["building"]) == len(marked)
    assert frames.detect(marked, "other")["verdict"] == "none"
    assert frames.detect(_clip(), "secret")["verdict"] == "none"


def test_pooling_a_still_clip_does_not_inflate_the_score():
    """Thirty copies of one unmarked frame are one piece of evidence, not thirty."""
    still = _clip(1, seed=3) * 30
    res = frames.detect(still, "secret")
    assert abs(res["z"] - res["per_frame"][0]) < 0.01 and res["verdict"] == "none"


def test_pooling_helps_when_frames_differ():
    marked = frames.embed(_clip(24), 1, "secret", strength=1.5)      # a weak mark under heavy per-frame noise
    res = frames.detect(marked, "secret")
    assert res["z"] > 1.5 * res["median_single"]


def test_label_box_is_added_and_read():
    mp4 = struct.pack(">I", 16) + b"ftypisom" + b"\x00" * 4 + struct.pack(">I", 12) + b"mdat" + b"\x00" * 4
    assert container.labels(mp4)["verdict"] == "none"
    labelled = container.add_label(mp4)
    res = container.labels(labelled)
    assert res["verdict"] == "detected" and res["parts"][-1]["kind"] == "label"
    assert sum(p["bytes"] for p in res["parts"]) == len(labelled)


@needs_ffmpeg
def test_codec_round_trip():
    src = list(sample.frames())[:12]
    v = codec.decode(codec.encode(src, sample.FPS, 18))
    assert len(v["frames"]) == 12 and v["frames"][0].shape == src[0].shape and v["codec"] == "h264"
    assert abs(v["fps"] - sample.FPS) < 0.01


@needs_ffmpeg
def test_make_check_and_survive_through_the_api():
    client = TestClient(app)
    made = client.post("/api/video/make", data={"key": "k1", "payload": "0000BEEF", "quality": "typical"}).json()
    files = {"file": ("m.mp4", base64.b64decode(made["video"]), "video/mp4")}

    res = client.post("/api/video/check", files=files, data={"key": "k1"}).json()
    assert res["labels"]["verdict"] == "detected" and res["frames"]["verdict"] == "detected"
    assert res["frames"]["payload_hex"] == "0000BEEF"
    assert client.post("/api/video/check", files=files, data={"key": "k2"}).json()["frames"]["verdict"] == "none"

    rows = {r["edit"]: r for r in client.post("/api/video/survive", files=files, data={"key": "k1"}).json()["rows"]}
    assert rows["No edit"]["label"] and not any(r["label"] for name, r in rows.items() if name != "No edit")
    assert rows["Half the frame rate"]["found"] and rows["Trim to the last third"]["found"]
    assert not rows["Mirror left to right"]["found"] and not rows["Crop 10% off each edge"]["found"]
