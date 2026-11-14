"""Layer 3: forensic classifier (optional).

Runs a pretrained "real vs AI" image classifier when nothing else is found.
The model is not bundled; point the service at one with environment variables:

  AIGC_CLASSIFIER_MODEL  Hugging Face model id or local directory of an
                         image-classification model (needs torch + transformers)
  AIGC_CLASSIFIER_ONNX   Directory containing model.onnx, config.json and
                         preprocessor_config.json (needs only onnxruntime;
                         create it with scripts/export_onnx.py). Preferred on
                         a CPU-only server.

The output is a probability, never proof. Thresholds below decide when the
score is reported as "likely AI", "likely not AI" or "undecided".
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from ..report import AI, INFO, NEUTRAL, NOT_AI, STATISTICAL, Evidence

LAYER = "classifier"
NAME = "Forensic classifier"
AI_THRESHOLD = float(os.environ.get("AIGC_AI_THRESHOLD", "0.75"))
REAL_THRESHOLD = float(os.environ.get("AIGC_REAL_THRESHOLD", "0.25"))

# Label names used by common detectors for the "AI" class.
AI_LABEL_HINTS = ("artificial", "ai", "fake", "synthetic", "generated", "sd", "diffusion",
                  "deepfake", "aigc", "machine")
REAL_LABEL_HINTS = ("human", "real", "natural", "photo", "authentic", "camera")

_lock = threading.Lock()
_model = None
_load_error: str | None = None


def is_ai_label(label: str) -> bool | None:
    low = label.lower().replace("-", " ").replace("_", " ")
    words = low.split()
    if any(h in words for h in REAL_LABEL_HINTS) or any(low.startswith(h) for h in REAL_LABEL_HINTS):
        return False
    if any(h in words for h in AI_LABEL_HINTS) or any(low.startswith(h) for h in AI_LABEL_HINTS):
        return True
    return None


def ai_probability(scores: dict[str, float]) -> float | None:
    """Collapse per-label scores into P(AI). None if labels are unrecognised."""
    ai = sum(s for l, s in scores.items() if is_ai_label(l) is True)
    real = sum(s for l, s in scores.items() if is_ai_label(l) is False)
    if ai == 0 and real == 0:
        return None
    if ai == 0:  # only a "real" label is recognisable: AI is everything else
        return max(0.0, 1.0 - real)
    return ai / (ai + real) if (ai + real) > 0 else None


class _OnnxModel:
    def __init__(self, directory: str):
        import onnxruntime as ort
        d = Path(directory)
        self.session = ort.InferenceSession(str(d / "model.onnx"),
                                            providers=["CPUExecutionProvider"])
        cfg = json.loads((d / "config.json").read_text())
        pre = json.loads((d / "preprocessor_config.json").read_text())
        self.labels = {int(k): v for k, v in cfg["id2label"].items()}
        size = pre.get("size", {"height": 224, "width": 224})
        if isinstance(size, int):
            size = {"height": size, "width": size}
        self.hw = (size.get("height") or size.get("shortest_edge", 224),
                   size.get("width") or size.get("shortest_edge", 224))
        self.mean = np.array(pre.get("image_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.std = np.array(pre.get("image_std", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.rescale = float(pre.get("rescale_factor", 1 / 255))
        self.name = f"ONNX: {d.name}"

    def __call__(self, img: Image.Image) -> dict[str, float]:
        x = np.asarray(img.convert("RGB").resize((self.hw[1], self.hw[0]), Image.BICUBIC),
                       dtype=np.float32) * self.rescale
        x = ((x - self.mean) / self.std).transpose(2, 0, 1)[None]
        logits = self.session.run(None, {self.session.get_inputs()[0].name: x})[0][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return {self.labels[i]: float(p) for i, p in enumerate(probs)}


class _HfModel:
    def __init__(self, model_id: str):
        from transformers import pipeline
        self.pipe = pipeline("image-classification", model=model_id)
        self.name = model_id

    def __call__(self, img: Image.Image) -> dict[str, float]:
        return {r["label"]: float(r["score"]) for r in self.pipe(img.convert("RGB"), top_k=None)}


def configured() -> str | None:
    return os.environ.get("AIGC_CLASSIFIER_ONNX") or os.environ.get("AIGC_CLASSIFIER_MODEL")


def _load():
    global _model, _load_error
    with _lock:
        if _model is not None or _load_error is not None:
            return
        try:
            onnx_dir = os.environ.get("AIGC_CLASSIFIER_ONNX")
            _model = _OnnxModel(onnx_dir) if onnx_dir else _HfModel(os.environ["AIGC_CLASSIFIER_MODEL"])
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"


def reset() -> None:
    """Forget the loaded model (used by tests when env vars change)."""
    global _model, _load_error
    with _lock:
        _model, _load_error = None, None


def analyze(img: Image.Image) -> list[Evidence]:
    if not configured():
        return [Evidence(LAYER, NAME, "Not run: no classifier model is configured. See the "
                         "README to enable this layer.")]
    _load()
    if _model is None:
        return [Evidence(LAYER, NAME, f"Not run: the classifier failed to load ({_load_error}).")]
    try:
        scores = _model(img)
    except Exception as exc:
        return [Evidence(LAYER, NAME, f"Classifier error ({type(exc).__name__}: {exc}).")]

    p = ai_probability(scores)
    details = {"model": _model.name, "scores": {k: round(v, 4) for k, v in scores.items()}}
    if p is None:
        return [Evidence(LAYER, NAME, "The model's labels were not recognised as real/AI classes: "
                         + ", ".join(scores), NEUTRAL, INFO, details)]
    details["ai_probability"] = round(p, 4)
    if p >= AI_THRESHOLD:
        return [Evidence(LAYER, NAME, f"The classifier estimates a {p:.0%} probability that "
                         "this image is AI-generated.", AI, STATISTICAL, details)]
    if p <= REAL_THRESHOLD:
        return [Evidence(LAYER, NAME, f"The classifier estimates only a {p:.0%} probability "
                         "that this image is AI-generated.", NOT_AI, STATISTICAL, details)]
    return [Evidence(LAYER, NAME, f"The classifier is undecided ({p:.0%} probability of AI).",
                     NEUTRAL, STATISTICAL, details)]
