"""Text watermark tests. The statistics are tested with a stand-in "model"
(random next-token scores), so they run without the downloaded weights."""
import numpy as np
import pytest

from app.text import greenlist, hidden, lm, tournament

VOCAB = 5000


def _write(scheme: str, key: str, n: int = 300, seed: int = 0) -> list[int]:
    """Sample n tokens from random distributions, watermarking as lm.generate does."""
    rng = np.random.default_rng(seed)
    gl, tn = greenlist.Config(key=key), tournament.Config(key=key)
    out = [1, 2, 3, 4]
    seen = set()
    for _ in range(n):
        logits = rng.normal(0, 2.5, VOCAB)
        if scheme == "greenlist":
            logits = greenlist.bias_logits(gl, logits, out)
        ids = np.argpartition(logits, -50)[-50:]
        probs = np.exp(logits[ids] - logits[ids].max())
        probs /= probs.sum()
        if scheme == "tournament":
            ctx = tuple(out[-tn.context:])
            if ctx not in seen:
                seen.add(ctx)
                probs = tournament.apply(tn, probs, ids, out)
        out.append(int(rng.choice(ids, p=probs)))
    return out[4:]


@pytest.mark.parametrize("scheme,mod,cfg", [("greenlist", greenlist, greenlist.Config),
                                            ("tournament", tournament, tournament.Config)])
def test_watermark_is_found_only_with_the_right_key(scheme, mod, cfg):
    tokens = _write(scheme, "secret")
    assert mod.detect(cfg(key="secret"), tokens)["z"] > 6
    assert abs(mod.detect(cfg(key="other"), tokens)["z"]) < 4
    assert abs(mod.detect(cfg(key="secret"), _write("none", "secret"))["z"]) < 4


@pytest.mark.parametrize("mod,cfg", [(greenlist, greenlist.Config), (tournament, tournament.Config)])
def test_unmarked_text_scores_like_a_standard_normal(mod, cfg):
    rng = np.random.default_rng(1)
    z = np.array([mod.detect(cfg(), rng.integers(0, VOCAB, 200).tolist())["z"] for _ in range(300)])
    assert abs(z.mean()) < 0.2 and 0.85 < z.std() < 1.15 and z.max() < 4


def test_tournament_keeps_the_models_probabilities_on_average():
    rng = np.random.default_rng(2)
    p, ids = rng.dirichlet(np.ones(40) * 0.3), np.arange(40)
    mean = sum(tournament.apply(tournament.Config(), p, ids, rng.integers(0, VOCAB, 4).tolist())
               for _ in range(3000)) / 3000
    assert np.abs(mean - p).max() < 0.02


def test_repeated_phrases_are_scored_once():
    tokens = [10, 11, 12, 13, 14, 15] * 20
    assert greenlist.detect(greenlist.Config(), tokens)["scored"] == 6
    assert tournament.detect(tournament.Config(), tokens)["scored"] == 6


def test_hidden_character_scan():
    res = hidden.scan("plain\u200b text\u202fwith \U000E0048\U000E0049 marks")
    assert res["invisible"] == 3 and res["spaces"] == 1 and res["tag_message"] == "HI"
    assert hidden.scan("nothing here")["items"] == []


@pytest.mark.skipif(not lm.tokenizer_ready(), reason="tokenizer not downloaded")
def test_analyze_endpoint_covers_the_whole_text():
    from fastapi.testclient import TestClient
    from app.main import app
    text = "Caf\u00e9 \u4f60\u597d.\n\nSecond line\u200b here."
    res = TestClient(app).post("/api/text/analyze", json={"text": text, "key": "k"}).json()
    assert "".join(t["t"] for t in res["tokens"]) + res["tail"] == text
    assert res["hidden"]["invisible"] == 1 and res["greenlist"]["verdict"] == "too_short"
