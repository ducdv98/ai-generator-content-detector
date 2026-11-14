"""Audio watermark lab tests. Codec and effect tests need ffmpeg and are skipped without it."""
import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio import codec, container, mark, sample
from app.main import app

needs_ffmpeg = pytest.mark.skipif(codec.ffmpeg_path() is None, reason="ffmpeg not installed")


def _noise(seconds: float = 6.0, seed: int = 0) -> np.ndarray:
    """Noise with a slowly changing colour: every band has sound, no pure tones."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.1, int(seconds * codec.RATE))
    return x + 0.5 * np.convolve(x, np.ones(8) / 8, mode="same")


def test_mark_needs_the_right_key_and_carries_the_id():
    y = mark.embed(_noise(), 0xBEEF, "secret")
    good = mark.detect(y, "secret")
    assert good["verdict"] == "detected" and good["payload_hex"] == "BEEF"
    assert mark.detect(y, "other")["verdict"] == "none"
    assert mark.detect(_noise(), "secret")["verdict"] == "none"


def test_unmarked_sound_scores_like_a_standard_normal():
    x = sample.samples()
    z = np.array([mark.detect(x, f"key-{i}")["z"] for i in range(120)])
    assert abs(z.mean()) < 0.3 and 0.7 < z.std() < 1.3 and z.max() < 4


def test_mark_needs_no_alignment_and_builds_with_time():
    y = mark.embed(_noise(8.0), 0x1234, "secret")
    whole = mark.detect(y, "secret")
    assert mark.detect(y[int(0.37 * codec.RATE):], "secret")["payload_hex"] == "1234"     # start cut off
    assert mark.detect(y[-len(y) // 3:], "secret")["verdict"] == "detected"               # last third only
    assert whole["building"][-1]["z"] > whole["building"][0]["z"]


def test_informed_writer_only_pushes_bands_on_the_wrong_side():
    x = sample.samples()
    push = mark.plan(x, 0xBEEF, "secret", "normal")
    limit, _ = mark.STRENGTHS["normal"]
    assert np.abs(push).max() <= limit + 1e-9 and 0.3 < (push != 0).mean() < 1.0
    assert mark.detect(mark.embed(x, 0xBEEF, "secret"), "secret")["payload_hex"] == "BEEF"


def test_label_chunk_is_added_and_read():
    wav = codec.wav_bytes(_noise(0.5))
    assert container.labels(wav)["verdict"] == "none"
    labelled = container.add_label(wav)
    res = container.labels(labelled)
    assert res["verdict"] == "detected" and res["parts"][-1]["kind"] == "label"
    assert sum(p["bytes"] for p in res["parts"]) == len(labelled)
    assert int.from_bytes(labelled[4:8], "little") == len(labelled) - 8          # RIFF size was updated


@needs_ffmpeg
def test_pitch_change_removes_the_mark_but_tempo_change_does_not():
    y = mark.embed(sample.samples(), 0xBEEF, "secret")
    assert mark.detect(codec.effect(y, "atempo=1.05"), "secret")["verdict"] == "detected"
    assert mark.detect(codec.effect(y, f"asetrate={codec.RATE}*1.05,aresample={codec.RATE}"), "secret")["verdict"] == "none"


@needs_ffmpeg
def test_make_check_and_survive_through_the_api():
    client = TestClient(app)
    made = client.post("/api/audio/make", data={"key": "k1", "payload": "BEEF", "strength": "normal"}).json()
    files = {"file": ("m.wav", base64.b64decode(made["after"]), "audio/wav")}

    res = client.post("/api/audio/check", files=files, data={"key": "k1"}).json()
    assert res["labels"]["verdict"] == "detected" and res["mark"]["payload_hex"] == "BEEF"
    assert client.post("/api/audio/check", files=files, data={"key": "k2"}).json()["mark"]["verdict"] == "none"

    rows = {r["edit"]: r for r in client.post("/api/audio/survive", files=files, data={"key": "k1"}).json()["rows"]}
    assert rows["No edit"]["label"] and not any(r["label"] for name, r in rows.items() if name != "No edit")
    assert rows["MP3, 128 kbps"]["id_ok"] and rows["Trim to the last third"]["found"]
    assert not rows["Pitch up 1%, same tempo"]["found"] and not rows["MP3, 64 kbps"]["found"]
