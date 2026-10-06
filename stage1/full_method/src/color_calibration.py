"""Lab colour calibration migrated from the historical SynthDR pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LabColorProfile:
    version: int
    reference_dir: str
    reference_images: int
    resize_for_estimation: int
    mask_rule: str
    lab_mean: list[float]
    lab_std: list[float]


def load_profile(path: str | Path) -> LabColorProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "version", "reference_dir", "reference_images", "resize_for_estimation",
        "mask_rule", "lab_mean", "lab_std",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"colour profile missing fields: {sorted(missing)}")
    if payload["version"] != 1:
        raise ValueError(f"unsupported colour profile version: {payload['version']}")
    if len(payload["lab_mean"]) != 3 or len(payload["lab_std"]) != 3:
        raise ValueError("lab_mean and lab_std must each contain three values")
    if not all(np.isfinite(payload["lab_mean"])) or not all(np.isfinite(payload["lab_std"])):
        raise ValueError("colour profile contains non-finite values")
    if any(float(value) <= 0 for value in payload["lab_std"]):
        raise ValueError("lab_std values must be positive")
    return LabColorProfile(**payload)


def match_to_profile(image: Image.Image, profile: LabColorProfile) -> Image.Image:
    """Match non-black pixels to the fitted Lab mean and standard deviation.

    Black camera borders remain untouched.  The operation is global and does
    not alter geometry or the laser mask.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = rgb.max(axis=2) > 20
    if int(mask.sum()) < 256:
        raise ValueError("generated image contains too few non-black pixels for colour calibration")

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    source = lab[mask]
    source_mean = source.mean(axis=0)
    source_std = np.maximum(source.std(axis=0), 1.0)
    target_mean = np.asarray(profile.lab_mean, dtype=np.float32)
    target_std = np.asarray(profile.lab_std, dtype=np.float32)

    calibrated = lab.copy()
    calibrated[mask] = (source - source_mean) * (target_std / source_std) + target_mean
    calibrated = np.clip(calibrated, 0, 255).astype(np.uint8)
    # Explicitly restore the original border: RGB<->Lab conversion can move
    # pure black by a rounding unit.
    calibrated[~mask] = lab[~mask].astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(calibrated, cv2.COLOR_LAB2RGB), "RGB")


def calibrate_file(input_path: str | Path, output_path: str | Path,
                   profile: LabColorProfile) -> Path:
    with Image.open(input_path) as image:
        calibrated = match_to_profile(image, profile)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibrated.save(output_path)
    return output_path
