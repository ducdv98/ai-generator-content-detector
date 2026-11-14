"""Download the small language model used by the text-watermark showcase.

    python scripts/download_text_model.py

Fetches HuggingFaceTB/SmolLM2-360M-Instruct (Apache-2.0) from Hugging Face into
models/SmolLM2-360M-Instruct: the 4-bit ONNX weights (about 390 MB) plus the
tokenizer and config. Detection needs only the tokenizer; generation needs the
weights.
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "HuggingFaceTB/SmolLM2-360M-Instruct"
FILES = ["tokenizer.json", "config.json", "onnx/model_q4.onnx"]
DEST = Path(__file__).resolve().parents[1] / "models" / "SmolLM2-360M-Instruct"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = hf_hub_download(REPO, name, local_dir=DEST)
        print(f"{Path(path).stat().st_size / 1e6:8.1f} MB  {path}")


if __name__ == "__main__":
    main()
