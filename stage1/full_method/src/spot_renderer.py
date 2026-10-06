"""Local copy of the renderer used by the full-method pipeline.

This file is intentionally kept self-contained so ``01_full_method`` can be
copied to another machine without importing the historical project tree.
The canonical implementation is kept byte-for-byte aligned with
``02_render_only/src/spot_renderer.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter

try:
    from .prior_guidance import PriorConfig, build_prior, save_prior_masks
except ImportError:  # Script execution keeps the lightweight entry point.
    from prior_guidance import PriorConfig, build_prior, save_prior_masks

MAX_SPOT_SHAPE_SCALE = 1.28

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class ZoneProfile:
    dark_depth: float = 0.5
    dark_center: float = 0.5
    dark_to_bright: float = 0.1
    bright_zone: float = 0.2
    fade_zone: float = 0.2
    ring_mode: bool = False
    ring_center: float = 0.65
    ring_width: float = 0.06
    ring_inner_fade: float = 0.18
    ring_outer_fade: float = 0.18
    ring_shoulder_power: float = 2.5

    def validate(self) -> None:
        values = (self.dark_depth, self.dark_center, self.dark_to_bright,
                  self.bright_zone, self.fade_zone, self.ring_center,
                  self.ring_width, self.ring_inner_fade, self.ring_outer_fade,
                  self.ring_shoulder_power)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("zone parameters must be finite numbers")
        if not 0.0 <= self.dark_depth <= 1.0:
            raise ValueError("dark_depth must be in [0, 1]")
        widths = (self.dark_center, self.dark_to_bright, self.bright_zone, self.fade_zone)
        if any(float(v) < 0.0 for v in widths):
            raise ValueError("zone widths must be non-negative")
        if sum(widths) > 1.0 + 1e-9:
            raise ValueError("zone widths must sum to <= 1")
        if self.ring_mode:
            if not 0.0 <= self.ring_center <= 1.0:
                raise ValueError("ring_center must be in [0, 1]")
            if not 0.0 < self.ring_width <= 1.0:
                raise ValueError("ring_width must be in (0, 1] in ring mode")
            if not 0.0 <= self.ring_inner_fade <= 1.0:
                raise ValueError("ring_inner_fade must be in [0, 1]")
            if not 0.0 <= self.ring_outer_fade <= 1.0:
                raise ValueError("ring_outer_fade must be in [0, 1]")
            if self.ring_shoulder_power < 1.0:
                raise ValueError("ring_shoulder_power must be >= 1")

    def multiplier(self, radius_fraction: np.ndarray) -> np.ndarray:
        self.validate()
        r = np.asarray(radius_fraction, dtype=np.float32)
        out = np.zeros_like(r, dtype=np.float32)
        if self.ring_mode:
            core_start = float(self.ring_center) - 0.5 * float(self.ring_width)
            core_end = float(self.ring_center) + 0.5 * float(self.ring_width)
            inner_start = max(0.0, core_start - float(self.ring_inner_fade))
            outer_end = min(1.0, core_end + float(self.ring_outer_fade))
            power = float(self.ring_shoulder_power)

            # Build one continuous peak.  There is intentionally no constant
            # bright interval: the profile rises to one maximum at
            # ring_center and falls continuously back to the background.
            center = float(self.ring_center)
            left_span = max(center - inner_start, 1e-8)
            right_span = max(outer_end - center, 1e-8)
            left = r <= center
            right = ~left
            if np.any(left):
                t = (r[left] - inner_start) / left_span
                out[left] = _smoothstep(t) ** power
            if np.any(right):
                t = (r[right] - center) / right_span
                out[right] = (1.0 - _smoothstep(t)) ** power

            # Keep the centre slightly dark before the peak, while matching
            # the peak curve continuously at the inner shoulder.
            if self.ring_inner_fade > 0:
                dark_t = _smoothstep((r - inner_start) / float(self.ring_inner_fade))
                dark_base = -float(self.dark_depth) * (1.0 - dark_t)
                out += dark_base * (r < center)
            return out

        dark_end = float(self.dark_center)
        bright_start = dark_end + float(self.dark_to_bright)
        fade_start = bright_start + float(self.bright_zone)
        fade_end = fade_start + float(self.fade_zone)
        dark = r < dark_end
        out[dark] = -float(self.dark_depth)
        transition = (r >= dark_end) & (r < bright_start)
        if np.any(transition):
            width = max(float(self.dark_to_bright), 1e-8)
            t = (r[transition] - dark_end) / width
            out[transition] = -float(self.dark_depth) + _smoothstep(t) * (
                1.0 + float(self.dark_depth)
            )
        bright = (r >= bright_start) & (r < fade_start)
        out[bright] = 1.0
        fade = (r >= fade_start) & (r < fade_end)
        if np.any(fade) and self.fade_zone > 0:
            t = (r[fade] - fade_start) / float(self.fade_zone)
            out[fade] = 1.0 - np.clip(t, 0.0, 1.0)
        return out


@dataclass
class RenderConfig:
    num_spots: int = 160
    seed: int = 42
    spot_size: float = 1.0
    gap: float = 0.10
    gain: float = 0.20
    central_exclusion: float = 0.18
    max_attempts_per_spot: int = 250
    warm_r: float = 1.04
    warm_g: float = 1.02
    warm_b: float = 1.00
    profile: ZoneProfile = ZoneProfile()
    no_prior: bool = False
    device: str = "auto"
    vessel_model: Optional[str] = None
    fovea_model_dir: Optional[str] = None
    disc_margin: float = 1.5
    vessel_buffer_px: int = 8
    posterior_pole_radius: float = 160.0
    save_prior_masks: bool = True

    def validate(self) -> None:
        if self.num_spots < 0:
            raise ValueError("num_spots must be >= 0")
        if self.spot_size <= 0 or self.gap < 0 or self.gain < 0:
            raise ValueError("spot_size, gap and gain must be non-negative (spot_size > 0)")
        if not 0 <= self.central_exclusion < 1:
            raise ValueError("central_exclusion must be in [0, 1)")
        if self.max_attempts_per_spot < 1:
            raise ValueError("max_attempts_per_spot must be positive")
        self.profile.validate()


def _as_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB") if image.mode != "RGB" else image.copy()


def detect_fov_mask(image: Image.Image, threshold: int = 20) -> np.ndarray:
    rgb = np.asarray(_as_rgb(image), dtype=np.uint8)
    binary = (np.max(rgb, axis=2) > int(threshold)).astype(np.uint8)
    if cv2 is None:
        return binary.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _fov_geometry(fov: np.ndarray) -> tuple[float, float, float]:
    ys, xs = np.nonzero(fov)
    if len(xs) == 0:
        h, w = fov.shape
        return w / 2.0, h / 2.0, min(w, h) / 2.0
    return float(xs.mean()), float(ys.mean()), max(math.sqrt(len(xs) / math.pi), 1.0)


def _stable_seed(seed: int, image_key: str) -> int:
    digest = hashlib.sha256(image_key.encode("utf-8")).digest()
    return (int(seed) + int.from_bytes(digest[:4], "little")) % (2**32)


def plan_spots(fov_mask: np.ndarray, config: RenderConfig, *, center: Optional[tuple[float, float]] = None,
               avoidance_mask: Optional[np.ndarray] = None) -> list[tuple[float, float, float]]:
    config.validate()
    h, w = fov_mask.shape
    cx, cy, fov_radius = _fov_geometry(fov_mask)
    if center is not None:
        cx, cy = float(center[0]), float(center[1])
    rng = random.Random(int(config.seed))
    base_radius = max(2.0, 0.018 * min(h, w) * float(config.spot_size))
    min_distance = max(2.0, 2.0 * base_radius * (1.0 + float(config.gap)))
    exclusion_radius = float(config.central_exclusion) * fov_radius
    ys, xs = np.nonzero(fov_mask)
    if len(xs) == 0 or config.num_spots == 0:
        return []
    xmin, xmax, ymin, ymax = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    accepted: list[tuple[float, float, float]] = []
    avoidance_for_centers = avoidance_mask
    if avoidance_mask is not None:
        if avoidance_mask.shape != fov_mask.shape:
            raise ValueError("avoidance_mask must have the same shape as the background")
        footprint_radius = int(math.ceil(base_radius * 1.12 * MAX_SPOT_SHAPE_SCALE + 2.0))
        if footprint_radius > 0:
            if cv2 is not None:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (2 * footprint_radius + 1, 2 * footprint_radius + 1))
                avoidance_for_centers = cv2.dilate(avoidance_mask.astype(np.uint8), kernel) > 0
            else:
                size = 2 * footprint_radius + 1
                expanded = Image.fromarray(avoidance_mask.astype(np.uint8) * 255)
                avoidance_for_centers = np.asarray(
                    expanded.filter(ImageFilter.MaxFilter(size=size))
                ) > 0
    attempts = 0
    limit = max(1, int(config.num_spots) * int(config.max_attempts_per_spot))
    while len(accepted) < int(config.num_spots) and attempts < limit:
        attempts += 1
        px, py = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        ix, iy = int(round(px)), int(round(py))
        if not (0 <= ix < w and 0 <= iy < h and fov_mask[iy, ix]):
            continue
        if math.hypot(px - cx, py - cy) < exclusion_radius:
            continue
        if avoidance_for_centers is not None:
            if avoidance_for_centers[iy, ix]:
                continue
        if any(math.hypot(px - qx, py - qy) < min_distance for qx, qy, _ in accepted):
            continue
        accepted.append((px, py, rng.uniform(0.88, 1.12)))
    return accepted


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _spot_patch(
    radius: float, profile: ZoneProfile, rng: random.Random
) -> tuple[np.ndarray, np.ndarray, int]:
    pad, half = int(math.ceil(radius * 0.35)) + 2, int(math.ceil(radius + int(math.ceil(radius * 0.35)) + 2))
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    theta = np.arctan2(y, x)
    irregularity = rng.uniform(0.16, 0.25)
    shape = np.ones_like(theta)
    for harmonic, weight in ((2, 0.58), (3, 0.30), (4, 0.12)):
        phase = rng.uniform(0.0, 2.0 * math.pi)
        shape += irregularity * weight * np.sin(harmonic * theta + phase)
    shape = np.clip(shape, 2.0 - MAX_SPOT_SHAPE_SCALE, MAX_SPOT_SHAPE_SCALE)
    r = np.sqrt(x * x + y * y) / (radius * shape)
    hard = r <= 1.0
    multiplier = profile.multiplier(r)
    multiplier[~hard] = 0.0
    edge_width = rng.uniform(0.04, 0.10)
    edge = hard & (r > 1.0 - edge_width)
    multiplier[edge] *= _smoothstep((1.0 - r[edge]) / edge_width)
    return multiplier.astype(np.float32), hard, half


def render_image(image: Image.Image, points: Sequence[tuple[float, float, float]], config: RenderConfig) -> tuple[Image.Image, Image.Image]:
    config.validate()
    base = np.asarray(_as_rgb(image), dtype=np.float32).copy()
    h, w = base.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    warm = np.asarray([config.warm_r, config.warm_g, config.warm_b], dtype=np.float32)
    rng = random.Random(int(config.seed) ^ 0x5EED5EED)
    for px, py, point_scale in points:
        radius = max(2.0, 0.018 * min(h, w) * config.spot_size * float(point_scale))
        multiplier, hard, half = _spot_patch(radius, config.profile, rng)
        x0, y0 = int(round(px)) - half, int(round(py)) - half
        x1, y1 = x0 + multiplier.shape[1], y0 + multiplier.shape[0]
        bx0, by0, bx1, by1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if bx0 >= bx1 or by0 >= by1:
            continue
        sx0, sy0 = bx0 - x0, by0 - y0
        sx1, sy1 = sx0 + bx1 - bx0, sy0 + by1 - by0
        m = multiplier[sy0:sy1, sx0:sx1][..., None]
        region = base[by0:by1, bx0:bx1]
        base[by0:by1, bx0:bx1] = np.clip(region * (1.0 + float(config.gain) * m * warm), 0.0, 255.0)
        mask[by0:by1, bx0:bx1] = np.maximum(mask[by0:by1, bx0:bx1], hard[sy0:sy1, sx0:sx1].astype(np.uint8) * 255)
    return Image.fromarray(base.astype(np.uint8), "RGB"), Image.fromarray(mask, "L")


def render_background(input_path: str | Path, output_dir: str | Path, config: Optional[RenderConfig] = None,
                      *, center: Optional[tuple[float, float]] = None,
                      avoidance_mask_path: Optional[str | Path] = None) -> dict:
    config = config or RenderConfig()
    config.validate()
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as source:
        image = _as_rgb(source)
    fov = detect_fov_mask(image)
    spot_radius = max(
        2.0,
        0.018 * min(image.height, image.width) * float(config.spot_size)
        * 1.12 * MAX_SPOT_SHAPE_SCALE + 4.0,
    )
    prior = build_prior(
        image,
        PriorConfig(
            enabled=not config.no_prior,
            device=config.device,
            vessel_model=config.vessel_model,
            fovea_model_dir=config.fovea_model_dir,
            disc_margin=config.disc_margin,
            vessel_buffer_px=config.vessel_buffer_px,
            posterior_pole_radius=config.posterior_pole_radius,
            save_masks=config.save_prior_masks,
        ),
        spot_radius=spot_radius,
    )
    if config.no_prior:
        print("[prior] skipped (--no-prior)")
    else:
        print(f"[prior] vessel={prior['status']['vessel']}; fovea_od={prior['status']['fovea_od']}")
    avoidance = None
    if avoidance_mask_path is not None:
        with Image.open(avoidance_mask_path) as mask_image:
            avoidance = np.asarray(mask_image.convert("L")) > 0
        if avoidance.shape != fov.shape:
            raise ValueError("avoidance_mask must have the same shape as the background")
    if prior.get("avoidance_mask") is not None:
        avoidance = prior["avoidance_mask"] if avoidance is None else (avoidance | prior["avoidance_mask"])
    effective = RenderConfig(**{**asdict(config), "profile": config.profile})
    effective.seed = _stable_seed(config.seed, input_path.name)
    placement_center = center if center is not None else prior.get("center")
    points = plan_spots(fov, effective, center=placement_center, avoidance_mask=avoidance)
    rendered, mask = render_image(image, points, effective)
    stem = input_path.stem
    rendered_path, mask_path, meta_path = output_dir / f"{stem}_synth.png", output_dir / f"{stem}_mask.png", output_dir / f"{stem}_meta.json"
    rendered.save(rendered_path)
    mask.save(mask_path)
    prior_outputs = save_prior_masks(prior, output_dir, stem) if config.save_prior_masks else {}
    meta = {"input": str(input_path), "size": [image.width, image.height], "requested_spots": int(config.num_spots),
            "generated_spots": len(points), "seed": int(effective.seed), "profile": asdict(config.profile),
            "config": {k: v for k, v in asdict(config).items() if k != "profile"},
            "outputs": {"image": str(rendered_path), "mask": str(mask_path)},
            "prior_guided": bool(not config.no_prior),
            "prior": {"status": prior.get("status", {}), "center": prior.get("center"),
                      "fovea": prior.get("fovea"), "optic_disc": prior.get("optic_disc"),
                      "parameters": prior.get("parameters", {})}}
    meta["outputs"].update(prior_outputs)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(points) < config.num_spots:
        print(f"[render] warning: placed {len(points)}/{config.num_spots} spots")
    return meta
