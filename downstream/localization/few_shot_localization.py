"""Command line entry point for the released few-shot landmark localisation head.

The runner is deliberately dataset agnostic. Give it a manifest (see
``localization_head.load_manifest``), a frozen DINOv3 checkpoint and a task list,
and it will, per task:

1. extract backbone tokens for every image once and cache them in memory;
2. select the training epoch by **support-only** leave-one-out cross-validation
   (lower mean Euclidean distance wins);
3. fit the final head on all support images;
4. evaluate it on the test images **once** and write per-image distances.

Nothing in this loop reads a test annotation before step 4. A provenance block
(student checkpoint SHA-256, manifest SHA-256, hyper-parameters) is written next
to the metrics so results can be replayed.

Example
-------
::

    python downstream/localization/few_shot_localization.py \
      --manifest ./data/landmarks.json \
      --student-weights ./models/student/student_final.pth \
      --backbone vit_base_patch16_dinov3 \
      --image-size 1024 \
      --epochs 10 50 100 \
      --output ./runs/localization
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from localization_head import (  # noqa: E402
    COORD_SPACES,
    DEFAULT_NUM_PREFIX_TOKENS,
    LOSS_MODES,
    PREPROCESS_MODES,
    FrozenTokenExtractor,
    build_feature_sets,
    evaluate_head,
    load_manifest,
    load_sample_and_target,
    resolve_prefix_tokens,
    select_epoch,
    seed_everything,
    summarize,
    train_localization_head,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_backbone(
    name: str,
    weights: Path | None,
    device: torch.device,
    model_kwargs: Dict[str, object] | None = None,
) -> torch.nn.Module:
    """Create a frozen DINOv3 ViT and optionally load a released student.

    ``model_kwargs`` is forwarded to ``timm.create_model``. Use it to relax a
    fixed input resolution, e.g. ``{"img_size": 1024}``.
    """
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "This runner needs `timm`. Install it with "
            "`pip install -r downstream/localization/requirements.txt`."
        ) from exc

    model = timm.create_model(name, pretrained=False, num_classes=0, **(model_kwargs or {}))
    if weights is not None:
        state = torch.load(weights, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(
                "Student checkpoint does not match the backbone "
                f"(missing={missing[:3]} unexpected={unexpected[:3]}). "
                "Pass --backbone matching the checkpoint architecture."
            )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_config(path: Path) -> Dict[str, object]:
    """Read a YAML or JSON config into ``argparse`` defaults.

    Keys use the underscore spelling of the flags (``image_size`` for
    ``--image-size``). Values that are ``None`` are dropped so the flag's own
    default applies, and ``model_kwargs`` may be written as a mapping instead of
    the JSON string the flag expects.
    """
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SystemExit(
                f"Reading {path} needs PyYAML; pass a .json config instead."
            ) from exc
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must decode to a mapping of flag names to values.")

    overrides: Dict[str, object] = {}
    for key, value in payload.items():
        if key == "version" or value is None:
            continue
        if key == "model_kwargs" and isinstance(value, dict):
            value = json.dumps(value)
        overrides[str(key)] = value
    return overrides


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Resolve --config first so its values become the defaults of every other
    # flag; an explicit command-line flag still wins.
    defaults: Dict[str, object] = {}
    for index, token in enumerate(argv):
        if token == "--config" and index + 1 < len(argv):
            defaults = load_config(Path(argv[index + 1]))
        elif token.startswith("--config="):
            defaults = load_config(Path(token.split("=", 1)[1]))

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="YAML/JSON config of flag defaults")
    parser.add_argument("--manifest", type=Path, default=None, help="dataset manifest JSON (required)")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="root for relative image paths (default: the manifest directory)",
    )
    parser.add_argument("--student-weights", type=Path, default=None, help="student .pth to back the frozen backbone")
    parser.add_argument(
        "--backbone",
        default="vit_base_patch16_dinov3",
        help="timm model name (default: vit_base_patch16_dinov3)",
    )
    parser.add_argument("--image-size", type=int, default=224, help="square input resolution (default: 224)")
    parser.add_argument(
        "--preprocess-mode",
        choices=PREPROCESS_MODES,
        default="stretch",
        help="how a non-square source is mapped onto the square (default: stretch)",
    )
    parser.add_argument(
        "--coord-space",
        choices=COORD_SPACES,
        default=None,
        help="override the manifest's coord_space (normalized or pixel)",
    )
    parser.add_argument("--epochs", type=int, nargs="+", default=[10, 50, 100], help="epoch candidates")
    parser.add_argument("--aug-repeats", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--loss-mode", choices=LOSS_MODES, default="smooth_l1")
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Smooth L1 transition point in normalised units (default: 0.05)",
    )
    parser.add_argument("--l1-weight", type=float, default=0.0, help="L1 penalty on the head weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-prefix-tokens",
        type=int,
        default=None,
        help="leading tokens to drop; defaults to the backbone's num_prefix_tokens",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="subset of the manifest tasks to run (default: all)",
    )
    parser.add_argument("--output", "--out-dir", dest="output", type=Path, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda when available, else cpu)",
    )
    parser.add_argument(
        "--model-kwargs",
        default="{}",
        help='extra JSON kwargs forwarded to timm, e.g. \'{"img_size": 1024}\'',
    )
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.manifest is None:
        parser.error("--manifest is required (on the command line or in --config)")
    if args.output is None:
        parser.error("--output is required (on the command line or in --config)")
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seed_everything(args.seed)

    manifest = load_manifest(args.manifest, args.data_root)
    samples = {sample.image_id: sample for sample in manifest.samples}
    splits = manifest.splits
    if not splits:
        raise SystemExit("Manifest contains no tasks; nothing to run.")
    coord_space = args.coord_space or manifest.coord_space

    requested = set(args.tasks or list(splits))
    unknown = requested - set(splits)
    if unknown:
        raise SystemExit(f"Tasks not in the manifest: {sorted(unknown)}")

    try:
        model_kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--model-kwargs is not valid JSON: {exc}") from exc
    if not isinstance(model_kwargs, dict):
        raise SystemExit("--model-kwargs must decode to a JSON object")

    backbone = load_backbone(args.backbone, args.student_weights, device, model_kwargs)
    prefix_tokens = resolve_prefix_tokens(backbone, args.num_prefix_tokens)
    extractor = FrozenTokenExtractor(
        backbone=backbone,
        device=device,
        image_size=args.image_size,
        preprocess_mode=args.preprocess_mode,
        num_prefix_tokens=prefix_tokens,
    )

    provenance = {
        "config": str(args.config) if args.config else None,
        "config_sha256": sha256(args.config) if args.config else None,
        "backbone": args.backbone,
        "student_weights": str(args.student_weights) if args.student_weights else None,
        "student_sha256": sha256(args.student_weights) if args.student_weights else None,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "data_root": str(args.data_root) if args.data_root else None,
        "image_size": args.image_size,
        "preprocess_mode": args.preprocess_mode,
        "coord_space": coord_space,
        "epoch_candidates": args.epochs,
        "aug_repeats": args.aug_repeats,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "loss_mode": args.loss_mode,
        "beta": args.beta,
        "l1_weight": args.l1_weight,
        "seed": args.seed,
        "num_prefix_tokens": prefix_tokens,
        "model_kwargs": model_kwargs,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }

    train_kwargs = {
        "hidden_dim": args.hidden_dim,
        "loss_mode": args.loss_mode,
        "beta": args.beta,
        "l1_weight": args.l1_weight,
        "weight_decay": args.weight_decay,
        "num_prefix_tokens": prefix_tokens,
    }

    summary: Dict[str, Dict[str, object]] = {}
    for task in sorted(requested):
        split = splits[task]
        support = [i for i in split.support if i in samples]
        test = [i for i in split.test if i in samples]
        if not support:
            raise SystemExit(f"Task {task}: no usable support images in the manifest.")
        names = split.targets

        original, augmented, targets = build_feature_sets(
            samples, task, support, extractor, args.aug_repeats, args.seed, coord_space, names
        )
        features: Dict[str, torch.Tensor] = dict(original)
        for image_id in test:
            if image_id not in features:
                image, target = load_sample_and_target(
                    samples[image_id], task, args.image_size, args.preprocess_mode, coord_space, names
                )
                features[image_id] = extractor.extract(image)
                targets[image_id] = target

        if len(support) < 2:
            print(f"[{task}] single support image: skipping CV, using defaults")
            best_epoch = args.epochs[-1]
            cv_scores: Dict[int, float] = {}
        else:
            best_epoch, cv_scores = select_epoch(
                support,
                original,
                targets,
                augmented,
                device,
                args.epochs,
                args.lr,
                args.seed,
                args.image_size,
                names=names,
                **train_kwargs,
            )
        print(f"[{task}] selected epochs={best_epoch}", flush=True)

        head = train_localization_head(
            support,
            original,
            targets,
            augmented,
            device,
            best_epoch,
            args.lr,
            args.seed,
            **train_kwargs,
        )
        rows = evaluate_head(head, test, features, targets, device, args.image_size, names)
        task_dir = args.output / task
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "test_metrics.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        summary[task] = {
            "selected_epoch": best_epoch,
            "support_cv": {str(k): v for k, v in cv_scores.items()},
            "n_support": len(support),
            "n_test": len(test),
            "metrics": summarize(rows),
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {"provenance": provenance, "tasks": summary}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
