"""Prior-guided spot placement borrowed from the historical SynthDR pipeline.

The old renderer first segmented vessels, localized the optic disc/fovea, then
excluded a buffered vessel mask, the optic disc and the posterior pole around
the fovea.  This module keeps that ordering with local-only checkpoints.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class PriorConfig:
    enabled: bool = True
    device: str = "auto"
    vessel_model: Optional[str] = None
    fovea_model_dir: Optional[str] = None
    disc_margin: float = 1.5
    vessel_buffer_px: int = 8
    posterior_pole_radius: float = 160.0
    save_masks: bool = True


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _disk_mask(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return ((xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2 <= float(radius) ** 2)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0
    padded = np.pad(mask, radius, mode="constant")
    result = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            if (dx - radius) ** 2 + (dy - radius) ** 2 <= radius ** 2:
                result |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return result


def _load_vessels(image: Image.Image, cfg: PriorConfig, device: str) -> tuple[Optional[np.ndarray], str]:
    configured = cfg.vessel_model or os.environ.get("POSTGD_VESSEL_MODEL")
    if not configured:
        return None, "vessel model not configured"
    model_path = Path(configured)
    if model_path.is_file():
        model_path = model_path.parent
    if not (model_path / "model_checkpoint.pth").is_file():
        return None, f"vessel checkpoint not found: {model_path / 'model_checkpoint.pth'}"
    try:
        import torch
        from engine.models.get_model import get_arch
        from engine.utils.model_saving_loading import load_model

        model = get_arch("wnet", n_classes=1)
        model, _ = load_model(model, str(model_path), device)
        model = model.to(device).eval()
        resized = image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[-1]
            probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        mask = probability >= 0.5
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(image.size, Image.Resampling.NEAREST)
        return np.asarray(mask_image) > 0, "ok"
    except Exception as exc:
        return None, f"vessel segmentation failed: {exc}"


def _load_fovea_od(image: Image.Image, cfg: PriorConfig, device: str):
    configured = cfg.fovea_model_dir or os.environ.get("POSTGD_FOVEA_MODEL_DIR")
    if not configured:
        return None, "fovea/OD model not configured"
    model_dir = Path(configured)
    if not model_dir.is_dir():
        return None, f"fovea/OD model directory not found: {model_dir}"
    try:
        import fundus_image_toolbox as fit

        model, _ = fit.load_fovea_od_model(
            checkpoint_dir=str(model_dir), device=device, return_test_dataloader=False
        )
        prediction = np.asarray(model.predict([np.asarray(image.convert("RGB"))]), dtype=float).reshape(-1)
        if prediction.size != 4 or not np.isfinite(prediction).all():
            return None, f"invalid fovea/OD prediction shape: {prediction.shape}"
        fx, fy, ox, oy = map(float, prediction)
        return (fx, fy, ox, oy), "ok"
    except Exception as exc:
        return None, f"fovea/OD localization failed: {exc}"


def build_prior(image: Image.Image, cfg: PriorConfig, *, spot_radius: float) -> dict:
    h, w = image.height, image.width
    fov_center = (w / 2.0, h / 2.0)
    if not cfg.enabled:
        return {"enabled": False, "center": fov_center, "avoidance_mask": None,
                "vessel_mask": None, "optic_disc_mask": None, "posterior_pole_mask": None,
                "fovea": None, "optic_disc": None,
                "status": {"vessel": "disabled", "fovea_od": "disabled"}}

    device = _resolve_device(cfg.device)
    vessel, vessel_status = _load_vessels(image, cfg, device)
    fovea_od, fovea_status = _load_fovea_od(image, cfg, device)
    if fovea_od is None:
        fx, fy, ox, oy = (*fov_center, *fov_center)
    else:
        fx, fy, ox, oy = fovea_od

    od_distance = math.hypot(ox - fx, oy - fy)
    disc_radius = max(16.0, 0.22 * od_distance * max(0.8, float(cfg.disc_margin)))
    scale = min(w / 518.0, h / 395.0)
    posterior_radius = max(1.0, float(cfg.posterior_pole_radius) * scale)
    optic_disc = _disk_mask((h, w), (ox, oy), disc_radius)
    posterior = _disk_mask((h, w), (fx, fy), posterior_radius)
    vessel_buffer = int(round(float(cfg.vessel_buffer_px) + 6.0 * scale * 0.5))
    vessel_block = np.zeros((h, w), dtype=bool)
    if vessel is not None:
        vessel_block = _dilate(vessel, vessel_buffer)
        vessel_status = (
            f"{vessel_status} (mask {float(vessel.mean()):.1%}, "
            f"buffered {float(vessel_block.mean()):.1%})"
        )
    # The spot planner expands this semantic exclusion mask by the spot
    # footprint once. Keeping the footprint out here avoids double dilation.
    # Without a usable vessel prediction, fail closed rather than place spots
    # using only the non-vascular priors.
    block = (
        vessel_block | optic_disc | posterior
        if vessel is not None else np.ones((h, w), dtype=bool)
    )
    return {
        "enabled": True, "center": (ox, oy) if fovea_od is not None else fov_center,
        "avoidance_mask": block, "vessel_mask": vessel_block if vessel is not None else None,
        "optic_disc_mask": optic_disc, "posterior_pole_mask": posterior,
        "fovea": (fx, fy), "optic_disc": (ox, oy),
        "status": {"vessel": vessel_status, "fovea_od": fovea_status},
        "parameters": {"disc_radius": disc_radius, "posterior_pole_radius": posterior_radius,
                       "vessel_buffer_radius": vessel_buffer, "footprint_dilation": float(spot_radius)},
    }


def save_prior_masks(prior: dict, output_dir: Path, stem: str) -> dict:
    outputs = {}
    for key, filename in (("avoidance_mask", "_avoidance_mask.png"),
                          ("vessel_mask", "_vessel_mask.png"),
                          ("optic_disc_mask", "_optic_disc_mask.png"),
                          ("posterior_pole_mask", "_posterior_pole_mask.png")):
        mask = prior.get(key)
        if mask is None:
            continue
        path = output_dir / f"{stem}{filename}"
        Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)
        outputs[key] = str(path)
    return outputs
