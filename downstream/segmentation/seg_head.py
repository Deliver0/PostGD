"""Frozen-backbone few-shot segmentation head for PostGD.

This module implements the released Stage 3 *segmentation* protocol. It trains
only a dense-prediction decoder on frozen DINOv3 ViT features and selects the
training epoch and the decision threshold on the **support** images by
leave-one-out cross-validation. Test labels are read once, after the decoder
configuration is locked.

The code is dataset agnostic: it consumes a JSON manifest that maps image ids to
an image path and one mask path per task, plus a task -> support/test id
mapping. No clinical data is shipped with this repository.

Dependencies are deliberately limited to torch / PIL / numpy, matching the rest
of the repository (no torchvision).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps

# ImageNet statistics, matching the DINOv3 initialisation.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Stages 1..4 of the frozen backbone. The released model exposes these
# intermediate token maps in NCHW order.
DEFAULT_FEATURE_INDICES: Tuple[int, ...] = (5, 7, 9, 11)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    """Seed every RNG that can influence a run."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Image IO
# --------------------------------------------------------------------------- #
def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL RGB image -> CHW float tensor in [0, 1]."""
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:  # defensive: single-band source converted to RGB
        array = np.stack([array] * 3, axis=-1)
    raw = torch.from_numpy(array.copy())
    return raw.permute(2, 0, 1).float().div(255.0)


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """PIL mask image -> HW long tensor of {0, 1}.

    Mask values greater than 8 are foreground; lower values are background.
    """
    array = np.asarray(mask.convert("L"), dtype=np.uint8)
    return torch.from_numpy(array.copy()).gt(8).long()


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def load_mask_tensor(path: str | Path, image_size: int, mode: str = "stretch") -> torch.Tensor:
    """Load a mask and map it onto the network square.

    The mask must live on the same grid as the decoder output, so ``image_size``
    and ``mode`` have to match the values passed to the feature extractor.
    """
    mask = square_transform(load_mask(path), size=image_size, mode=mode, is_mask=True)
    return mask_to_tensor(mask)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def square_transform(image: Image.Image, size: int, mode: str, is_mask: bool) -> Image.Image:
    """Resize a square while preserving the requested geometry.

    ``mode`` selects how a non-square source is mapped onto the square:
    ``stretch`` (direct resize), ``center_crop`` or ``letterbox``.
    """
    image = image.convert("L" if is_mask else "RGB")
    resampling = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    if mode == "stretch":
        return image.resize((size, size), resampling)
    if mode == "center_crop":
        side = min(image.width, image.height)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        return image.crop((left, top, left + side, top + side)).resize((size, size), resampling)
    if mode == "letterbox":
        scale = min(size / image.width, size / image.height)
        new_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        resized = image.resize(new_size, resampling)
        fill = 0 if is_mask else (0, 0, 0)
        canvas = Image.new(image.mode, (size, size), fill)
        left = (size - new_size[0]) // 2
        top = (size - new_size[1]) // 2
        canvas.paste(resized, (left, top))
        return canvas
    raise ValueError(f"Unknown preprocess mode: {mode}")


def preprocess(
    image: Image.Image,
    mask: Image.Image | None = None,
    size: int = 224,
    mode: str = "stretch",
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Geometric transform followed by ImageNet normalisation."""
    image = square_transform(image, size=size, mode=mode, is_mask=False)
    x = pil_to_tensor(image)
    x = (x - MEAN) / STD
    out_mask = None
    if mask is not None:
        mask = square_transform(mask, size=size, mode=mode, is_mask=True)
        out_mask = mask_to_tensor(mask)
    return x, out_mask


def paired_augment(
    image: Image.Image,
    mask: Image.Image,
    rng: random.Random,
    index: int,
) -> Tuple[Image.Image, Image.Image]:
    """Geometric/photometric augmentation applied identically to image and mask.

    ``index`` picks which transforms are enabled so that a run can be replayed
    exactly from its seed.
    """
    image = image.convert("RGB")
    mask = mask.convert("L")
    if index % 2 == 1:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)
    if index % 3 == 0:
        angle = rng.uniform(-8.0, 8.0)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    if index % 4 == 0:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.95, 1.05))
    return image, mask


# --------------------------------------------------------------------------- #
# Frozen backbone interface
# --------------------------------------------------------------------------- #
class FrozenFeatureExtractor:
    """Cache-free feature extractor over a frozen DINOv3 ViT backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        image_size: int = 224,
        preprocess_mode: str = "stretch",
        feature_indices: Sequence[int] = DEFAULT_FEATURE_INDICES,
        feature_norm: bool = False,
    ):
        self.backbone = backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.image_size = image_size
        self.preprocess_mode = preprocess_mode
        self.feature_indices = list(feature_indices)
        self.feature_norm = feature_norm

    @torch.no_grad()
    def extract(self, image: Image.Image) -> List[torch.Tensor]:
        x, _ = preprocess(
            image,
            size=self.image_size,
            mode=self.preprocess_mode,
        )
        x = x.unsqueeze(0).to(self.device)
        features = self.backbone.forward_intermediates(
            x,
            indices=self.feature_indices,
            return_prefix_tokens=False,
            norm=self.feature_norm,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        return [f.detach().float().cpu() for f in features]


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #
class SegDecoder(nn.Module):
    """Four-stage DINOv3 token decoder with 512/256/128/64 channels."""

    def __init__(self, num_classes: int = 2, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.ModuleList(
            [
                nn.Conv2d(embed_dim, 64, 1),
                nn.Conv2d(embed_dim, 128, 1),
                nn.Conv2d(embed_dim, 256, 1),
                nn.Conv2d(embed_dim, 512, 1),
            ]
        )
        self.fuse_2 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fuse_1 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fuse_0 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)),
                nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)),
                nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)),
                nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)),
            ]
        )
        self.pred = nn.Conv2d(64, num_classes, 1)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        f0, f1, f2, f3 = [self.proj[i](features[i]) for i in range(4)]
        x = self.fuse_2(torch.cat([f3, f2], dim=1))
        x = self.fuse_1(torch.cat([x, f1], dim=1))
        x = self.fuse_0(torch.cat([x, f0], dim=1))
        for block in self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)
        return self.pred(x)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    target_fg = target.float()
    prob_fg = probs[:, 1]
    intersection = (prob_fg * target_fg).sum(dim=(1, 2))
    denom = prob_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    one_hot = F.one_hot(target.long(), num_classes=logits.shape[1])
    one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=probs.dtype)
    intersection = (probs * one_hot).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    fg_weight: float,
    dice_weight: float = 0.5,
    loss_mode: str = "foreground",
) -> torch.Tensor:
    """Cross entropy plus Dice.

    ``loss_mode`` selects the cross-entropy variant: ``foreground`` weights the
    positive class by a support-only estimate of foreground scarcity, while
    ``multiclass`` is the unweighted multi-class form.
    """
    if loss_mode == "multiclass":
        ce = F.cross_entropy(logits, target)
        dice = multiclass_dice_loss(logits, target)
    elif loss_mode == "foreground":
        weights = torch.tensor([1.0, fg_weight], dtype=logits.dtype, device=logits.device)
        ce = F.cross_entropy(logits, target, weight=weights)
        dice = dice_loss(logits, target)
    else:
        raise ValueError(f"Unknown segmentation loss mode: {loss_mode}")
    return ce + dice_weight * dice


def estimate_fg_weight(ids: Sequence[str], masks: Dict[str, torch.Tensor]) -> float:
    """Support-only class weight: counters imbalance without predicting fg everywhere."""
    fractions = [float(masks[image_id].float().mean().item()) for image_id in ids]
    fraction = max(sum(fractions) / max(len(fractions), 1), 1e-4)
    return float(max(2.0, min(20.0, 0.5 / fraction)))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Dice and per-class IoU.

    Binary mIoU is the mean of the background and foreground class IoUs, which
    is the quantity reported in the paper; ``foreground_iou`` is kept separately
    because it drops much faster on small targets and is a useful diagnostic.
    """
    pred = pred.bool()
    target = target.bool()
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    tn = (~pred & ~target).sum().item()
    dice = (2 * tp + 1.0) / (2 * tp + fp + fn + 1.0)
    fg_iou = (tp + 1.0) / (tp + fp + fn + 1.0)
    bg_iou = (tn + 1.0) / (tn + fp + fn + 1.0)
    mean_class_iou = (fg_iou + bg_iou) / 2.0
    return {
        "dice": float(dice),
        "foreground_iou": float(fg_iou),
        "background_iou": float(bg_iou),
        "miou": float(mean_class_iou),
        "positive_fraction": float(target.float().mean().item()),
    }


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One image and the masks that belong to it."""

    image_id: str
    image: str
    masks: Dict[str, str]


@dataclass
class TaskSplit:
    """Support/test ids for one segmentation task."""

    task: str
    support: List[str]
    test: List[str] = field(default_factory=list)


def load_manifest(path: str | Path) -> Tuple[List[Sample], Dict[str, TaskSplit]]:
    """Read the JSON manifest that describes a segmentation dataset.

    Expected layout::

        {
          "samples": [
            {"image_id": "02", "image": "images/02.jpg",
             "masks": {"laser_spots": "masks/laser/02.png", ...}},
            ...
          ],
          "tasks": {
            "laser_spots": {"support": ["02", ...], "test": [...]},
            ...
          }
        }

    Image and mask paths may be absolute or relative to the manifest directory.
    """
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent

    samples = [
        Sample(
            image_id=record["image_id"],
            image=str(_resolve(record["image"], root)),
            masks={task: str(_resolve(mask, root)) for task, mask in record["masks"].items()},
        )
        for record in payload["samples"]
    ]

    splits = {
        task: TaskSplit(
            task=task,
            support=list(spec.get("support", [])),
            test=list(spec.get("test", [])),
        )
        for task, spec in payload.get("tasks", {}).items()
    }
    return samples, splits


def _resolve(raw: str, root: Path) -> Path:
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def build_feature_sets(
    samples: Dict[str, Sample],
    task: str,
    train_ids: Sequence[str],
    extractor: FrozenFeatureExtractor,
    aug_repeats: int,
    seed: int,
) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, List[Tuple[List[torch.Tensor], torch.Tensor]]]]:
    """Extract original features plus augmented copies for the support ids."""
    original: Dict[str, List[torch.Tensor]] = {}
    augmented: Dict[str, List[Tuple[List[torch.Tensor], torch.Tensor]]] = {}
    rng = random.Random(seed)
    for image_id in train_ids:
        sample = samples[image_id]
        image = load_rgb(sample.image)
        mask = load_mask(sample.masks[task])
        original[image_id] = extractor.extract(image)
        variants: List[Tuple[List[torch.Tensor], torch.Tensor]] = []
        for index in range(aug_repeats):
            aug_image, aug_mask = paired_augment(image, mask, rng, index)
            aug_image, aug_mask = square_transform(
                aug_image, size=extractor.image_size, mode=extractor.preprocess_mode,
                is_mask=False,
            ), square_transform(
                aug_mask, size=extractor.image_size, mode=extractor.preprocess_mode,
                is_mask=True,
            )
            variants.append((extractor.extract(aug_image), mask_to_tensor(aug_mask)))
        augmented[image_id] = variants
    return original, augmented


def move_features(features: Sequence[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [f.to(device) for f in features]


def train_decoder(
    train_ids: Sequence[str],
    original_features: Dict[str, List[torch.Tensor]],
    masks: Dict[str, torch.Tensor],
    augmented: Dict[str, List[Tuple[List[torch.Tensor], torch.Tensor]]],
    device: torch.device,
    epochs: int,
    lr: float,
    seed: int,
    fg_weight: float | None = None,
    dice_weight: float = 0.5,
    loss_mode: str = "foreground",
) -> SegDecoder:
    seed_everything(seed)
    embed_dim = int(original_features[train_ids[0]][0].shape[1])
    decoder = SegDecoder(embed_dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)
    if fg_weight is None:
        fg_weight = estimate_fg_weight(train_ids, masks)
    decoder.train()
    train_items: List[Tuple[List[torch.Tensor], torch.Tensor]] = []
    for image_id in train_ids:
        train_items.append((original_features[image_id], masks[image_id]))
        train_items.extend(augmented[image_id])
    for _ in range(epochs):
        random.shuffle(train_items)
        for features, target in train_items:
            target = target.unsqueeze(0).to(device)
            logits = decoder(move_features(features, device))
            loss = segmentation_loss(
                logits, target, fg_weight, dice_weight, loss_mode
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()
    return decoder


@torch.no_grad()
def evaluate_decoder(
    decoder: SegDecoder,
    ids: Sequence[str],
    original_features: Dict[str, List[torch.Tensor]],
    masks: Dict[str, torch.Tensor],
    device: torch.device,
    threshold: float = 0.5,
    extractor: FrozenFeatureExtractor | None = None,
    samples: Dict[str, Sample] | None = None,
    tta: bool = False,
) -> List[Dict[str, object]]:
    """Evaluate one decoder on held-out ids; optional horizontal-flip TTA."""
    decoder.eval()
    rows: List[Dict[str, object]] = []
    for image_id in ids:
        logits = decoder(move_features(original_features[image_id], device))
        foreground_probs = [logits.softmax(dim=1)[:, 1][0]]
        if tta:
            if extractor is None or samples is None:
                raise ValueError("TTA requires a feature extractor and the sample table")
            image = ImageOps.mirror(load_rgb(samples[image_id].image))
            flipped_logits = decoder(move_features(extractor.extract(image), device))
            foreground_probs.append(flipped_logits.softmax(dim=1)[:, 1][0].flip(-1))
        foreground_prob = torch.stack(foreground_probs).mean(dim=0).cpu()
        pred = foreground_prob.ge(threshold)
        metrics = binary_metrics(pred, masks[image_id])
        rows.append({"id": image_id, **metrics})
    return rows


def select_epoch_and_threshold(
    train_ids: Sequence[str],
    original_features: Dict[str, List[torch.Tensor]],
    masks: Dict[str, torch.Tensor],
    augmented: Dict[str, List[Tuple[List[torch.Tensor], torch.Tensor]]],
    device: torch.device,
    epoch_candidates: Sequence[int],
    lr: float,
    seed: int,
    thresholds: Sequence[float],
    extractor: FrozenFeatureExtractor | None = None,
    samples: Dict[str, Sample] | None = None,
    tta: bool = False,
    fg_weight: float | None = None,
    dice_weight: float = 0.5,
    loss_mode: str = "foreground",
    verbose: bool = True,
) -> Tuple[int, float, Dict[int, List[float]]]:
    """Support-only leave-one-out selection of (epoch, threshold).

    Each support image is held out in turn, the decoder is fitted on the
    remaining support images, and the held-out score is recorded. **Test images
    are never consulted here.**
    """
    scores: Dict[int, List[float]] = {}
    for epochs in epoch_candidates:
        fold_scores: List[List[Dict[str, float]]] = []
        for fold, held_out in enumerate(train_ids):
            fit_ids = [x for x in train_ids if x != held_out]
            decoder = train_decoder(
                fit_ids,
                original_features,
                masks,
                augmented,
                device,
                epochs,
                lr,
                seed + 1000 * fold + epochs,
                fg_weight if fg_weight is not None else estimate_fg_weight(fit_ids, masks),
                dice_weight,
                loss_mode,
            )
            fold_rows = []
            for threshold in thresholds:
                fold_rows.append(
                    evaluate_decoder(
                        decoder,
                        [held_out],
                        original_features,
                        masks,
                        device,
                        threshold,
                        extractor=extractor,
                        samples=samples,
                        tta=tta,
                    )[0]
                )
            fold_scores.append(fold_rows)
            del decoder
            if device.type == "cuda":
                torch.cuda.empty_cache()
        scores[epochs] = [
            sum((row[j]["dice"] + row[j]["miou"]) / 2.0 for row in fold_scores)
            / len(fold_scores)
            for j in range(len(thresholds))
        ]
        if verbose:
            best_j = max(range(len(thresholds)), key=lambda j: scores[epochs][j])
            print(
                f"support-CV epochs={epochs} "
                f"threshold={thresholds[best_j]:.2f} "
                f"score={scores[epochs][best_j]:.4f}",
                flush=True,
            )
    best_epoch, best_j = max(
        ((epoch, j) for epoch, values in scores.items() for j in range(len(values))),
        key=lambda item: scores[item[0]][item[1]],
    )
    best_threshold = float(thresholds[best_j])
    if verbose:
        print(
            f"selected epochs={best_epoch} threshold={best_threshold:.2f} "
            f"support-CV={scores[best_epoch][best_j]:.4f}",
            flush=True,
        )
    return int(best_epoch), best_threshold, scores


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    """Mean of every numeric metric in ``rows``."""
    if not rows:
        return {}
    keys = [key for key in rows[0] if isinstance(rows[0][key], float)]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
