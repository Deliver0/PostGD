"""Fit a public Lab calibration profile from a training-only image directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def estimate_profile(reference_dir: Path, resize: int) -> dict:
    values: list[np.ndarray] = []
    files = image_files(reference_dir)
    if not files:
        raise FileNotFoundError(f"no images found in {reference_dir}")

    for path in files:
        with Image.open(path) as source:
            image = source.convert("RGB")
            scale = min(1.0, float(resize) / max(image.size))
            if scale < 1.0:
                size = (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                )
                image = image.resize(size, Image.Resampling.BILINEAR)
            rgb = np.asarray(image, dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        chroma = lab[..., 1:3].ptp(axis=2)
        mask = (rgb.max(axis=2) > 20) & (chroma > 8)
        if mask.any():
            values.append(lab[mask])

    if not values:
        raise ValueError("no valid non-black, chromatic pixels found")
    pixels = np.concatenate(values, axis=0)
    return {
        "version": 1,
        "reference_dir": str(reference_dir),
        "reference_images": len(files),
        "resize_for_estimation": resize,
        "mask_rule": "max_rgb>20 and chroma_range>8",
        "lab_mean": pixels.mean(axis=0).round(6).tolist(),
        "lab_std": np.maximum(pixels.std(axis=0), 1.0).round(6).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a Lab profile using training reference images only"
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resize", type=int, default=256)
    args = parser.parse_args()

    profile = estimate_profile(args.reference_dir, args.resize)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

