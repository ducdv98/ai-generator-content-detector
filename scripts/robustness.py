"""Measure how the sign-and-verify watermark holds up under common edits.

Usage:  python scripts/robustness.py [image ...]
With no arguments it uses sample photos bundled with scikit-learn / scikit-image
(if installed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import attacks, signing  # noqa: E402

PAYLOAD = 0xC0FFEE42


def sample_images() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    try:
        from sklearn.datasets import load_sample_images
        ds = load_sample_images()
        for name, img in zip(ds.filenames, ds.images):
            out[Path(name).stem] = np.asarray(img)
    except Exception:
        pass
    try:
        from skimage import data
        for name in ("astronaut", "coffee", "chelsea", "rocket"):
            out[name] = np.asarray(getattr(data, name)())
    except Exception:
        pass
    return out


def main() -> None:
    if len(sys.argv) > 1:
        images = {Path(p).stem: np.asarray(Image.open(p).convert("RGB")) for p in sys.argv[1:]}
    else:
        images = sample_images()
    if not images:
        sys.exit("No images given and no sample images available.")

    names = list(attacks.ATTACKS)
    detected = {n: 0 for n in names}
    payload_ok = {n: 0 for n in names}
    zs = {n: [] for n in names}
    psnrs, null_z = [], []

    for name, rgb in images.items():
        null_z.append(signing.detect(rgb).z_score)
        res = signing.embed(rgb, PAYLOAD)
        psnrs.append(res.psnr_db)
        for a in names:
            d = signing.detect(attacks.ATTACKS[a](res.image))
            zs[a].append(d.z_score)
            detected[a] += d.present
            payload_ok[a] += d.present and d.payload == PAYLOAD

    n = len(images)
    print(f"{n} images, mean PSNR {np.mean(psnrs):.1f} dB (min {np.min(psnrs):.1f})")
    print(f"unsigned originals: max z = {max(null_z):.2f} (threshold {signing.DETECT_Z})")
    print(f"{'edit':<28}{'detected':>10}{'payload ok':>12}{'min z':>8}{'mean z':>8}")
    for a in names:
        print(f"{a:<28}{detected[a]:>7}/{n}{payload_ok[a]:>9}/{n}{min(zs[a]):>8.1f}{np.mean(zs[a]):>8.1f}")

    # False-alarm check: unsigned images against many wrong keys.
    worst = max(
        signing.detect(rgb, key=f"wrong-key-{i}").z_score
        for rgb in images.values() for i in range(200)
    )
    print(f"false-alarm check: {n * 200} unsigned image/key pairs, max z = {worst:.2f}")


if __name__ == "__main__":
    main()
