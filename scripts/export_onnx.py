"""Export a Hugging Face image-classification model to ONNX for CPU serving.

Run this on a machine with torch + transformers (e.g. the GPU laptop), then
copy the output directory to the server and set AIGC_CLASSIFIER_ONNX to it.
The server then needs only onnxruntime, not torch.

Usage:  python scripts/export_onnx.py <model-id-or-local-dir> <output-dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


class _LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model_id, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForImageClassification.from_pretrained(model_id).eval()
    processor = AutoImageProcessor.from_pretrained(model_id)

    size = processor.size
    h = size.get("height") or size.get("shortest_edge", 224)
    w = size.get("width") or size.get("shortest_edge", 224)
    dummy = torch.zeros(1, 3, h, w)
    torch.onnx.export(
        _LogitsOnly(model), (dummy,), str(out / "model.onnx"),
        input_names=["pixel_values"], output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    (out / "config.json").write_text(json.dumps(
        {"id2label": {str(k): v for k, v in model.config.id2label.items()}}, indent=2))
    (out / "preprocessor_config.json").write_text(json.dumps({
        "size": {"height": h, "width": w},
        "image_mean": list(processor.image_mean),
        "image_std": list(processor.image_std),
        "rescale_factor": float(getattr(processor, "rescale_factor", 1 / 255)),
    }, indent=2))
    print(f"Exported to {out} (labels: {model.config.id2label})")


if __name__ == "__main__":
    main()
