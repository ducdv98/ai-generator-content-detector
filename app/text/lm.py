"""The language model the watermarks are applied to, and the sampling loop.

A small open model (SmolLM2-360M-Instruct) run with onnxruntime. Any model that
gives next-token scores would do: both watermarks act between "the model scored
every token" and "one token is picked", and `generate` is written out in full
so that the place they act is visible.

Detection never loads the model. It uses the tokenizer only.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np

from . import greenlist, tournament

MODEL_DIR = Path(os.environ.get(
    "AIGC_TEXT_MODEL", Path(__file__).resolve().parents[2] / "models" / "SmolLM2-360M-Instruct"))
MODEL_NAME = "SmolLM2-360M-Instruct"
_SYSTEM = "You are a helpful assistant. Answer in clear, flowing paragraphs of plain prose."
_lock = threading.Lock()


def tokenizer_ready() -> bool:
    return (MODEL_DIR / "tokenizer.json").exists()


def model_ready() -> bool:
    return tokenizer_ready() and (MODEL_DIR / "onnx" / "model_q4.onnx").exists()


@lru_cache(maxsize=1)
def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort
    return ort.InferenceSession(str(MODEL_DIR / "onnx" / "model_q4.onnx"),
                                providers=["CPUExecutionProvider"])


class _Decoder:
    """Feeds tokens to the model one step at a time, keeping its attention cache."""

    def __init__(self) -> None:
        self.sess = _session()
        self.past = {}
        for inp in self.sess.get_inputs():
            if inp.name.startswith("past_key_values"):
                self.past[inp.name] = np.zeros((1, inp.shape[1], 0, inp.shape[3]), dtype=np.float32)
        self.length = 0

    def step(self, ids: list[int]) -> np.ndarray:
        """Append `ids`; return the scores (logits) for the token after them."""
        n = len(ids)
        feeds = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, self.length + n), dtype=np.int64),
            "position_ids": np.arange(self.length, self.length + n, dtype=np.int64)[None, :],
            **self.past,
        }
        out = self.sess.run(None, feeds)
        names = [o.name for o in self.sess.get_outputs()]
        for name, value in zip(names[1:], out[1:]):
            self.past[name.replace("present", "past_key_values")] = value
        self.length += n
        return out[0][0, -1].astype(np.float64)


def _chat_prompt(prompt: str) -> list[int]:
    text = (f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    return tokenizer().encode(text, add_special_tokens=False).ids


def generate(prompt: str, scheme: str = "none", key: str = "demo-key", delta: float = 2.0,
             max_new_tokens: int = 200, temperature: float = 0.9, top_k: int = 50,
             seed: int | None = None) -> Iterator[str]:
    """Write a reply to `prompt`, yielding text as it is produced.

    scheme: "none", "greenlist" or "tournament".
    """
    tok = tokenizer()
    stop = {tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")}
    rng = np.random.default_rng(seed)
    gl = greenlist.Config(key=key, delta=delta)
    tn = tournament.Config(key=key)

    with _lock:
        model = _Decoder()
        history = _chat_prompt(prompt)        # every token so far, prompt included
        written: list[int] = []               # the reply only
        seen_contexts: set[tuple[int, ...]] = set()
        shown = ""
        logits = model.step(history)

        for _ in range(max_new_tokens):
            # ---- watermark hook 1: green-list changes the scores -------------
            if scheme == "greenlist":
                logits = greenlist.bias_logits(gl, logits, history)

            # Ordinary sampling set-up: temperature, keep the top-k, softmax.
            logits = logits / temperature
            ids = np.argpartition(logits, -top_k)[-top_k:]
            probs = np.exp(logits[ids] - logits[ids].max())
            probs /= probs.sum()

            # ---- watermark hook 2: tournament changes which draw wins --------
            if scheme == "tournament":
                ctx = tuple(history[-tn.context:])
                if ctx not in seen_contexts:      # a repeated context is left unmarked
                    seen_contexts.add(ctx)
                    probs = tournament.apply(tn, probs, ids, history)

            token = int(rng.choice(ids, p=probs))
            if token in stop:
                break
            history.append(token)
            written.append(token)

            text = tok.decode(written)
            if not text.endswith("�"):       # wait for multi-byte characters to complete
                yield text[len(shown):]
                shown = text
            logits = model.step([token])
