"""Complete SynthDR stage one pipeline.

The pipeline is deliberately kept independent from the historical ``engine``
directory:

    background generation -> Lab calibration -> vessel-safe spot rendering
    -> circular FOV crop -> resize to the distillation input size

For a large run, backgrounds can be generated once and reused while the
renderer varies the spot profile, seed, size, gap, and requested spot count.
The actual number of spots remains an upper bound because anatomical avoidance
has priority.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw


SRC = Path(__file__).resolve().parent / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))

from color_calibration import calibrate_file, load_profile  # noqa: E402
from prior_guidance import PriorConfig, build_prior, save_prior_masks  # noqa: E402
from spot_renderer import (  # noqa: E402
    RenderConfig,
    ZoneProfile,
    detect_fov_mask,
    plan_spots,
    render_image,
)


SPOT_CASES = {
    # User-approved bright appearance without a dark centre.
    "bright_no_dark": {
        "spot_size": 2.0,
        "gap": 0.001,
        "gain": 0.12,
        "num_spots": 160,
        "profile": {
            "dark_depth": 0.0,
            "dark_center": 0.0,
            "dark_to_bright": 0.0,
            "bright_zone": 0.20,
            "fade_zone": 0.80,
            "ring_mode": False,
        },
    },
    # User-approved dark-centre profile retained from the render-only README.
    "dark_center": {
        "spot_size": 1.8,
        "gap": 0.02,
        "gain": 0.10,
        "num_spots": 160,
        "profile": {
            "dark_depth": 0.04,
            "dark_center": 0.0,
            "dark_to_bright": 0.0,
            "bright_zone": 0.0,
            "fade_zone": 0.0,
            "ring_mode": True,
            "ring_center": 0.55,
            "ring_width": 0.05,
            "ring_inner_fade": 0.25,
            "ring_outer_fade": 0.45,
            "ring_shoulder_power": 1.0,
        },
    },
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate SynthDR stage-one images for distillation"
    )
    p.add_argument("--output-dir", "--output_dir", default="./output")
    p.add_argument("--count", type=int, default=1,
                   help="number of final distillation images to create")
    p.add_argument("--background-count", type=int, default=1,
                   help="number of diffusion backgrounds; reused when count is larger")
    p.add_argument("--background-dir", "--background_dir",
                   help="reuse existing background images instead of diffusion")
    p.add_argument("--size", type=int, default=512,
                   help="diffusion/background working size")
    p.add_argument("--output-size", "--output_size", type=int, default=224,
                   help="square distillation input size")
    p.add_argument("--prompt", default="Mild Non-Proliferative Diabetic Retinopathy")
    p.add_argument("--base-model", "--sd-base", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--weight-dir", "--weight_dir",
                   default=str(PROJECT_ROOT / "models" / "sd-retina-model"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--bg-seed", "--bg_seed", type=int, default=20260833)
    p.add_argument("--render-seed", "--render_seed", type=int, default=20260834)
    p.add_argument("--seed", type=int,
                   help="convenience seed: sets bg-seed=seed and render-seed=seed+1")
    p.add_argument("--color-profile", "--color_profile",
                   default=str(PROJECT_ROOT / "configs" / "hospital_lab_template.json"))
    p.add_argument("--skip-color-cal", "--skip_color_cal", action="store_true")
    p.add_argument("--allow-base-fallback", action="store_true")
    p.add_argument("--allow-download", action="store_true")

    p.add_argument("--spot-case", choices=["bright_no_dark", "dark_center", "mixed"],
                   default="bright_no_dark",
                   help="approved appearance preset; mixed alternates the two cases")
    p.add_argument("--no-variation", "--no_perturb", action="store_true",
                   help="disable batch variation of spot size, gap, and count")
    p.add_argument("--spot-size", "--spot_size", type=float, default=None)
    p.add_argument("--gap", type=float, default=None)
    p.add_argument("--gain", type=float, default=None)
    p.add_argument("--num-spots", "--num_spots", type=int, default=None,
                   help="upper bound; placement may produce fewer spots")
    p.add_argument("--dark-depth", "--dark_depth", type=float, default=None)
    p.add_argument("--dark-center", "--dark_center", type=float, default=None)
    p.add_argument("--dark-to-bright", "--dark_to_bright", type=float, default=None)
    p.add_argument("--bright-zone", "--bright_zone", type=float, default=None)
    p.add_argument("--fade-zone", "--fade_zone", type=float, default=None)
    p.add_argument("--ring-mode", "--ring_mode", action="store_true", default=None)
    p.add_argument("--ring-center", "--ring_center", type=float, default=None)
    p.add_argument("--ring-width", "--ring_width", type=float, default=None)
    p.add_argument("--ring-inner-fade", "--ring_inner_fade", type=float, default=None)
    p.add_argument("--ring-outer-fade", "--ring_outer_fade", type=float, default=None)
    p.add_argument("--ring-shoulder-power", "--ring_shoulder_power",
                   type=float, default=None)

    p.add_argument("--no-prior", "--no_prior", action="store_true",
                   help="disable anatomical placement priors; not recommended")
    p.add_argument("--vessel-model", "--vessel_model")
    p.add_argument("--fovea-model-dir", "--fovea_model_dir")
    p.add_argument("--disc-margin", "--disc_margin", type=float, default=1.5)
    p.add_argument("--vessel-buffer", "--vessel_buffer", type=int, default=8)
    p.add_argument("--posterior-pole-radius", "--posterior_pole_radius",
                   type=float, default=160.0)
    p.add_argument("--save-prior-masks", action="store_true",
                   help="keep diagnostic prior masks in the temporary render output")
    p.add_argument("--resume", action="store_true",
                   help="keep existing final images and fill missing indices")
    return p


def _load_pipeline(args):
    try:
        import torch
        from diffusers import (AutoencoderKL, PNDMScheduler,
                               StableDiffusionPipeline,
                               UNet2DConditionModel)
        from huggingface_hub import snapshot_download
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "full_method requires torch, diffusers, transformers and huggingface_hub"
        ) from exc

    requested = args.device
    device = ("cuda" if requested == "auto" and torch.cuda.is_available() else
              "cpu" if requested == "auto" else requested)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is not available")
    args.device = device
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    unet_path = Path(args.weight_dir) / "checkpoint-60000" / "unet"
    local_finetuned = unet_path.is_dir() and any(
        (unet_path / filename).is_file()
        for filename in ("diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors")
    )
    if not local_finetuned and not args.allow_base_fallback:
        raise FileNotFoundError(
            f"fine-tuned UNet not found at {unet_path}; "
            "pass --allow-base-fallback explicitly to use base SD"
        )

    base_path = Path(args.base_model)
    if not base_path.is_dir():
        try:
            base_path = Path(snapshot_download(args.base_model, local_files_only=True))
        except Exception as exc:
            if not args.allow_download:
                raise RuntimeError(
                    f"base model {args.base_model!r} is not fully cached; "
                    "add --allow-download only when network access is intended"
                ) from exc
            base_path = Path(snapshot_download(args.base_model, local_files_only=False))

    component_dirs = {
        "tokenizer": base_path / "tokenizer",
        "text_encoder": base_path / "text_encoder",
        "vae": base_path / "vae",
        "scheduler": base_path / "scheduler",
    }
    missing = [name for name, path in component_dirs.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"base model missing components: {', '.join(missing)}")

    tokenizer = CLIPTokenizer.from_pretrained(
        str(component_dirs["tokenizer"]), local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        str(component_dirs["text_encoder"]), torch_dtype=dtype, local_files_only=True
    )
    vae = AutoencoderKL.from_pretrained(
        str(component_dirs["vae"]), torch_dtype=dtype, local_files_only=True
    )
    scheduler = PNDMScheduler.from_pretrained(
        str(component_dirs["scheduler"]), local_files_only=True
    )

    if local_finetuned:
        unet = UNet2DConditionModel.from_pretrained(
            str(unet_path), torch_dtype=dtype, local_files_only=True, use_safetensors=False
        )
    else:
        base_unet = base_path / "unet"
        if not any((base_unet / filename).is_file()
                   for filename in ("diffusion_pytorch_model.bin",
                                    "diffusion_pytorch_model.safetensors")):
            if not args.allow_download:
                raise FileNotFoundError(f"base UNet is not cached at {base_unet}")
            base_path = Path(snapshot_download(args.base_model, local_files_only=False))
        unet = UNet2DConditionModel.from_pretrained(
            str(base_path / "unet"), torch_dtype=dtype, local_files_only=True,
            use_safetensors=False
        )

    kwargs = dict(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
    )
    try:
        pipe = StableDiffusionPipeline(**kwargs, requires_safety_checker=False)
    except TypeError:
        pipe = StableDiffusionPipeline(**kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, torch


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )


def _prepare_backgrounds(args, output_dir: Path) -> list[Path]:
    background_dir = output_dir / "backgrounds"
    background_dir.mkdir(parents=True, exist_ok=True)
    requested = max(1, int(args.background_count))

    if args.background_dir:
        sources = _image_files(Path(args.background_dir))
        if not sources:
            raise FileNotFoundError(f"no backgrounds found in {args.background_dir}")
        requested = min(requested, len(sources))
        result = []
        for index, source in enumerate(sources[:requested]):
            target = background_dir / f"bg_{index:06d}.png"
            if not target.exists():
                with Image.open(source) as image:
                    image.convert("RGB").save(target)
            result.append(target)
        return result

    pipe, torch = _load_pipeline(args)
    size = max(64, int(args.size))
    size -= size % 8
    result = []
    try:
        for index in range(requested):
            target = background_dir / f"bg_{index:06d}.png"
            if args.resume and target.exists():
                result.append(target)
                continue
            generator = torch.Generator(device=args.device).manual_seed(
                int(args.bg_seed) + index
            )
            image = pipe(
                args.prompt,
                height=size,
                width=size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            image.convert("RGB").save(target)
            result.append(target)
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _calibrate_backgrounds(args, backgrounds: list[Path],
                           output_dir: Path) -> list[Path]:
    if args.skip_color_cal:
        print("[Color] skipped (--skip-color-cal)")
        return backgrounds
    profile = load_profile(args.color_profile)
    calibrated_dir = output_dir / "calibrated_backgrounds"
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for background in backgrounds:
        target = calibrated_dir / background.name
        if not (args.resume and target.exists()):
            calibrate_file(background, target, profile)
        result.append(target)
    return result


def apply_fov_hard_mask(image: Image.Image, target_size: int = 224) -> Image.Image:
    """Crop the non-black FOV to a square, apply a circular mask, then resize."""
    rgb = image.convert("RGB")
    gray = np.asarray(rgb.convert("L"))
    ys, xs = np.where(gray > 30)
    if len(xs) == 0:
        cx, cy = rgb.width // 2, rgb.height // 2
        radius = max(2, min(rgb.size) // 2 - 10)
    else:
        cx, cy = int(np.median(xs)), int(np.median(ys))
        radius = min(cx - int(xs.min()), int(xs.max()) - cx,
                     cy - int(ys.min()), int(ys.max()) - cy) - 10
        radius = max(2, radius)

    square_size = min(radius * 2, rgb.width, rgb.height)
    left = max(0, min(rgb.width - square_size, cx - square_size // 2))
    top = max(0, min(rgb.height - square_size, cy - square_size // 2))
    crop = rgb.crop((left, top, left + square_size, top + square_size))
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, square_size - 1, square_size - 1), fill=255)
    masked = Image.new("RGB", crop.size, (0, 0, 0))
    masked.paste(crop, (0, 0), mask)
    return masked.resize((int(target_size), int(target_size)), Image.Resampling.LANCZOS)


def _case_for_index(case_name: str, index: int) -> str:
    if case_name == "mixed":
        return "bright_no_dark" if index % 2 == 0 else "dark_center"
    return case_name


def _render_config(args, index: int) -> tuple[str, RenderConfig]:
    case_name = _case_for_index(args.spot_case, index)
    values = deepcopy(SPOT_CASES[case_name])
    rng = random.Random(int(args.render_seed) + index * 1009)

    if not args.no_variation:
        values["spot_size"] *= rng.uniform(0.85, 1.15)
        if case_name == "bright_no_dark":
            values["gap"] = rng.choice((0.001, 0.005, 0.015, 0.03))
        else:
            values["gap"] = rng.choice((0.01, 0.02, 0.03, 0.05))
        values["num_spots"] = rng.randint(
            max(1, int(values["num_spots"] * 0.60)),
            int(values["num_spots"]),
        )

    for key, value in {
        "spot_size": args.spot_size,
        "gap": args.gap,
        "gain": args.gain,
        "num_spots": args.num_spots,
    }.items():
        if value is not None:
            values[key] = value

    profile_values = dict(values["profile"])
    for key, value in {
        "dark_depth": args.dark_depth,
        "dark_center": args.dark_center,
        "dark_to_bright": args.dark_to_bright,
        "bright_zone": args.bright_zone,
        "fade_zone": args.fade_zone,
        "ring_center": args.ring_center,
        "ring_width": args.ring_width,
        "ring_inner_fade": args.ring_inner_fade,
        "ring_outer_fade": args.ring_outer_fade,
        "ring_shoulder_power": args.ring_shoulder_power,
    }.items():
        if value is not None:
            profile_values[key] = value
    if args.ring_mode is not None:
        profile_values["ring_mode"] = args.ring_mode

    config = RenderConfig(
        num_spots=int(values["num_spots"]),
        seed=int(args.render_seed) + index * 1009,
        spot_size=float(values["spot_size"]),
        gap=float(values["gap"]),
        gain=float(values["gain"]),
        profile=ZoneProfile(**profile_values),
        no_prior=bool(args.no_prior),
        device=args.device,
        vessel_model=args.vessel_model,
        fovea_model_dir=args.fovea_model_dir,
        disc_margin=float(args.disc_margin),
        vessel_buffer_px=int(args.vessel_buffer),
        posterior_pole_radius=float(args.posterior_pole_radius),
        save_prior_masks=bool(args.save_prior_masks),
    )
    config.validate()
    return case_name, config


def _write_manifest_line(handle, index: int, background: Path, final_path: Path,
                         case_name: str, config: RenderConfig, render_meta: dict,
                         output_size: int) -> None:
    handle.write(json.dumps({
        "index": index,
        "output": str(final_path),
        "background": str(background),
        "case": case_name,
        "seed": config.seed,
        "spot_size": config.spot_size,
        "gap": config.gap,
        "gain": config.gain,
        "requested_spots": config.num_spots,
        "generated_spots": render_meta.get("generated_spots"),
        "profile": vars(config.profile),
        "prior_guided": not config.no_prior,
        "output_size": [int(output_size), int(output_size)],
    }, ensure_ascii=False) + "\n")


def _render_with_cached_prior(
    background: Path,
    config: RenderConfig,
    args,
    cache: dict[str, tuple[Image.Image, np.ndarray, dict]],
    output_dir: Path,
) -> tuple[Image.Image, dict]:
    """Render one image while reusing the anatomical prior for a background."""
    key = str(background.resolve())
    if key not in cache:
        with Image.open(background) as source:
            image = source.convert("RGB")
        fov = detect_fov_mask(image)
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
            spot_radius=max(2.0, 0.018 * min(image.size) * config.spot_size),
        )
        cache[key] = (image, fov, prior)
        if config.save_prior_masks:
            prior_dir = output_dir / "prior_masks"
            prior_dir.mkdir(parents=True, exist_ok=True)
            save_prior_masks(prior, prior_dir, background.stem)
        if config.no_prior:
            print("[prior] skipped (--no-prior)")
        else:
            print(
                f"[prior] {background.name}: "
                f"vessel={prior['status']['vessel']}; "
                f"fovea_od={prior['status']['fovea_od']}"
            )

    image, fov, prior = cache[key]
    center = prior.get("center")
    points = plan_spots(
        fov,
        config,
        center=center,
        avoidance_mask=prior.get("avoidance_mask"),
    )
    rendered, _ = render_image(image, points, config)
    return rendered, {
        "generated_spots": len(points),
        "prior": prior.get("status", {}),
    }


def run(args: argparse.Namespace) -> Path:
    if args.seed is not None:
        args.bg_seed = int(args.seed)
        args.render_seed = int(args.seed) + 1
    if args.count < 1:
        raise ValueError("--count must be >= 1")
    if args.output_size < 1:
        raise ValueError("--output-size must be positive")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / f"distill_{int(args.output_size)}"
    final_dir.mkdir(parents=True, exist_ok=True)
    backgrounds = _prepare_backgrounds(args, output_dir)
    calibrated = _calibrate_backgrounds(args, backgrounds, output_dir)

    manifest_path = output_dir / "stage1_manifest.jsonl"
    existing = set()
    if args.resume and manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                existing.add(int(json.loads(line)["index"]))
            except (ValueError, KeyError, json.JSONDecodeError):
                continue

    manifest_mode = "a" if args.resume else "w"
    prior_cache: dict[str, tuple[Image.Image, np.ndarray, dict]] = {}
    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest:
        for index in range(args.count):
            final_path = final_dir / f"synth_{index:06d}.png"
            if args.resume and index in existing and final_path.exists():
                continue
            case_name, config = _render_config(args, index)
            background = calibrated[index % len(calibrated)]
            rendered, render_meta = _render_with_cached_prior(
                background, config, args, prior_cache, output_dir
            )
            final = apply_fov_hard_mask(rendered, args.output_size)
            final.save(final_path)
            _write_manifest_line(
                manifest, index, background, final_path, case_name, config,
                render_meta, args.output_size
            )
            if (index + 1) % 100 == 0 or index == 0:
                print(f"[stage1] {index + 1}/{args.count} -> {final_path.name}")

    meta = {
        "count_requested": int(args.count),
        "background_count": len(backgrounds),
        "backgrounds": [str(path) for path in backgrounds],
        "calibrated_backgrounds": [str(path) for path in calibrated],
        "output_dir": str(final_dir),
        "output_size": int(args.output_size),
        "spot_case": args.spot_case,
        "variation_enabled": not args.no_variation,
        "seed": {"background": int(args.bg_seed), "render": int(args.render_seed)},
        "blood_vessel_avoidance": not args.no_prior,
        "profile_presets": SPOT_CASES,
        "argv": sys.argv,
    }
    (output_dir / "_stage1_config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[stage1] complete: {final_dir}")
    return final_dir


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _parser().error(str(exc))


if __name__ == "__main__":
    main()
