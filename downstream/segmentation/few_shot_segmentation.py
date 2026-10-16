"""Command line entry point for the released few-shot segmentation head.

The runner is deliberately dataset agnostic. Give it a manifest (see
``seg_head.load_manifest``), a frozen DINOv3 checkpoint and a task list, and it
will, per task:

1. extract backbone features for every image once and cache them in memory;
2. select the decoder epoch and the decision threshold by **support-only**
   leave-one-out cross-validation;
3. fit the final decoder on all support images;
4. evaluate it on the test images **once** and write per-image metrics.

Nothing in this loop reads a test label before step 4. A provenance block
(student checkpoint SHA-256, split digest, hyper-parameters) is written next to
the metrics so results can be replayed.

Example
-------
::

    python downstream/segmentation/few_shot_segmentation.py \
      --manifest ./data/manifest.json \
      --student-weights ./models/dinov3/student_final.pth \
      --backbone vit_base_patch16_dinov3 \
      --image-size 1024 \
      --epoch-candidates 10 50 100 150 \
      --threshold-candidates 0.3 0.5 0.7 0.9 \
      --output ./runs/segmentation
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

from seg_head import (  # noqa: E402
    DEFAULT_FEATURE_INDICES,
    FrozenFeatureExtractor,
    build_feature_sets,
    evaluate_decoder,
    load_manifest,
    load_mask_tensor,
    load_rgb,
    move_features,
    select_epoch_and_threshold,
    seed_everything,
    summarize,
    train_decoder,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_string(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
            "`pip install -r downstream/segmentation/requirements.txt`."
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, type=Path, help="dataset manifest JSON")
    parser.add_argument("--student-weights", type=Path, default=None, help="student .pth to back the frozen backbone")
    parser.add_argument(
        "--backbone",
        default="vit_base_patch16_dinov3",
        help="timm model name (default: vit_base_patch16_dinov3)",
    )
    parser.add_argument("--image-size", type=int, default=1024, help="square input resolution (default: 1024)")
    parser.add_argument(
        "--preprocess-mode",
        choices=("stretch", "center_crop", "letterbox"),
        default="stretch",
        help="how a non-square source is mapped onto the square (default: stretch)",
    )
    parser.add_argument("--epoch-candidates", type=int, nargs="+", default=[10, 50, 100, 150])
    parser.add_argument("--threshold-candidates", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--aug-repeats", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--loss-mode", choices=("foreground", "multiclass"), default="foreground")
    parser.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="subset of the manifest tasks to run (default: all)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--model-kwargs",
        default="{}",
        help='extra JSON kwargs forwarded to timm, e.g. \'{"img_size": 1024}\'',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seed_everything(args.seed)

    samples_list, splits = load_manifest(args.manifest)
    samples = {sample.image_id: sample for sample in samples_list}
    if not splits:
        raise SystemExit("Manifest contains no tasks; nothing to run.")

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
    extractor = FrozenFeatureExtractor(
        backbone=backbone,
        device=device,
        image_size=args.image_size,
        preprocess_mode=args.preprocess_mode,
        feature_indices=DEFAULT_FEATURE_INDICES,
    )

    provenance = {
        "backbone": args.backbone,
        "student_weights": str(args.student_weights) if args.student_weights else None,
        "student_sha256": sha256(args.student_weights) if args.student_weights else None,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "image_size": args.image_size,
        "preprocess_mode": args.preprocess_mode,
        "epoch_candidates": args.epoch_candidates,
        "threshold_candidates": args.threshold_candidates,
        "aug_repeats": args.aug_repeats,
        "lr": args.lr,
        "seed": args.seed,
        "dice_weight": args.dice_weight,
        "loss_mode": args.loss_mode,
        "tta": args.tta,
        "feature_indices": list(DEFAULT_FEATURE_INDICES),
        "model_kwargs": model_kwargs,
        "torch": torch.__version__,
        "python": platform.python_version(),
    }

    summary: Dict[str, Dict[str, object]] = {}
    feature_bank: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    mask_bank: Dict[str, Dict[str, torch.Tensor]] = {}

    for task in sorted(requested):
        split = splits[task]
        support = [i for i in split.support if i in samples]
        test = [i for i in split.test if i in samples]
        if not support:
            raise SystemExit(f"Task {task}: no usable support images in the manifest.")

        original, augmented = build_feature_sets(
            samples, task, support, extractor, args.aug_repeats, args.seed
        )
        for image_id in test:
            if image_id not in feature_bank.setdefault(task, {}):
                feature_bank[task][image_id] = extractor.extract(load_rgb(samples[image_id].image))
        mask_bank[task] = {
            i: load_mask_tensor(samples[i].masks[task], args.image_size, args.preprocess_mode)
            for i in support + test
        }

        if len(support) < 2:
            print(f"[{task}] single support image: skipping CV, using defaults")
            best_epoch = args.epoch_candidates[-1]
            best_threshold = 0.5
            cv_scores = {}
        else:
            best_epoch, best_threshold, cv_scores = select_epoch_and_threshold(
                support,
                original,
                mask_bank[task],
                augmented,
                device,
                args.epoch_candidates,
                args.lr,
                args.seed,
                args.threshold_candidates,
                extractor=extractor,
                samples=samples,
                tta=args.tta,
                dice_weight=args.dice_weight,
                loss_mode=args.loss_mode,
            )
        print(f"[{task}] selected epochs={best_epoch} threshold={best_threshold:.2f}", flush=True)

        decoder = train_decoder(
            support,
            original,
            mask_bank[task],
            augmented,
            device,
            best_epoch,
            args.lr,
            args.seed,
            fg_weight=None,
            dice_weight=args.dice_weight,
            loss_mode=args.loss_mode,
        )
        rows = evaluate_decoder(
            decoder,
            test,
            feature_bank[task],
            mask_bank[task],
            device,
            best_threshold,
            extractor=extractor,
            samples=samples,
            tta=args.tta,
        )
        task_dir = args.output / task
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "test_metrics.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        summary[task] = {
            "selected_epoch": best_epoch,
            "selected_threshold": best_threshold,
            "support_cv": {str(k): v for k, v in cv_scores.items()},
            "n_support": len(support),
            "n_test": len(test),
            "metrics": summarize(rows),
        }
        del decoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {"provenance": provenance, "tasks": summary}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
