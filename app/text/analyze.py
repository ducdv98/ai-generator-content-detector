"""Run every text check over a pasted passage and return the evidence."""
from __future__ import annotations

from . import greenlist, hidden, lm, tournament

MAX_CHARS = 20_000
DETECT_Z = 4.0     # about 1 unmarked text in 30,000 scores this high by chance
HINT_Z = 2.5
MIN_TOKENS = 50    # below this even a real watermark rarely reaches DETECT_Z


def _verdict(z: float, scored: int) -> str:
    if z >= DETECT_Z:
        return "detected"
    if scored < MIN_TOKENS:
        return "too_short"
    return "weak" if z >= HINT_Z else "none"


def analyze(text: str, key: str = "demo-key") -> dict:
    text = text[:MAX_CHARS]
    enc = lm.tokenizer().encode(text, add_special_tokens=False)
    ids = list(enc.ids)

    gl_cfg = greenlist.Config(key=key)
    tn_cfg = tournament.Config(key=key)
    gl = greenlist.detect(gl_cfg, ids)
    tn = tournament.detect(tn_cfg, ids)

    # Cut the original text into one piece per token, so the page can colour it
    # in place. A token's piece runs from the end of the previous one to its own
    # end, which keeps spaces and line breaks.
    tokens, pos = [], 0
    for i, (tid, (_, end)) in enumerate(zip(ids, enc.offsets)):
        end = max(end, pos)
        tokens.append({"t": text[pos:end], "id": tid, "gl": gl["marks"][i],
                       "tn": None if tn["scores"][i] is None else round(tn["scores"][i], 4)})
        pos = end
    tail = text[pos:]

    for res in (gl, tn):
        res["verdict"] = _verdict(res["z"], res["scored"])
    del gl["marks"], tn["scores"]
    gl["params"] = {"gamma": gl_cfg.gamma, "context": gl_cfg.context}
    tn["params"] = {"layers": tn_cfg.layers, "context": tn_cfg.context}

    return {"tokens": tokens, "tail": tail, "n_tokens": len(ids), "key": key,
            "tokenizer": lm.MODEL_NAME, "detect_z": DETECT_Z, "hint_z": HINT_Z,
            "min_tokens": MIN_TOKENS,
            "greenlist": gl, "tournament": tn, "hidden": hidden.scan(text)}
