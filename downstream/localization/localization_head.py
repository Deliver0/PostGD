"""Frozen-backbone few-shot landmark localisation head for PostGD.

This module implements the released Stage 3 *landmark localisation* protocol. It
trains only a light heat-map head on frozen DINOv3 ViT tokens and selects the
training epoch on the **support** images by leave-one-out cross-validation. Test
annotations are read once, after the head configuration is locked.

The code is dataset agnostic: it consumes a JSON manifest that maps image ids to
an image path and one ``(x, y)`` annotation per landmark task, plus a task ->
support/test id mapping. No clinical data is shipped with this repository.

Coordinates live in the **network square**. An annotation stored on the original
image is mapped through the same geometry that ``square_transform`` applies to
the pixels, and the head predicts coordinates normalised to ``[0, 1]`` on that
square. Euclidean distance is therefore reported in pixels at the input
resolution (``distance * image_size``), which is the quantity in Table III.

Dependencies are deliberately limited to torch / PIL / numpy, matching the rest
of the repository (no torchvision).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance

# ImageNet statistics, matching the DINOv3 initialisation.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# timm ViTs prepend a class token; DINOv3 checkpoints add register tokens on top
# of it and expose the total through ``model.num_prefix_tokens``.
DEFAULT_NUM_PREFIX_TOKENS = 1

PREPROCESS_MODES: Tuple[str, ...] = ("stretch", "center_crop", "letterbox")
COORD_SPACES: Tuple[str, ...] = ("normalized", "pixel")
LOSS_MODES: Tuple[str, ...] = ("smooth_l1", "mse")


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


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def square_transform(image: Image.Image, size: int, mode: str = "stretch") -> Image.Image:
    """Map a source image onto the network square.

    ``mode`` selects how a non-square source is mapped: ``stretch`` (direct
    resize), ``center_crop`` or ``letterbox``. The same three modes and the same
    rounding are used by ``downstream/segmentation/seg_head.py``, so an image and
    its landmarks stay aligned across both heads.
    """
    image = image.convert("RGB")
    resampling = Image.Resampling.BILINEAR
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
        canvas = Image.new(image.mode, (size, size), (0, 0, 0))
        left = (size - new_size[0]) // 2
        top = (size - new_size[1]) // 2
        canvas.paste(resized, (left, top))
        return canvas
    raise ValueError(f"Unknown preprocess mode: {mode}")


def preprocess(image: Image.Image, size: int = 224, mode: str = "stretch") -> torch.Tensor:
    """Geometric transform followed by ImageNet normalisation."""
    image = square_transform(image, size=size, mode=mode)
    return (pil_to_tensor(image) - MEAN) / STD


def photometric_augment(image: Image.Image, rng: random.Random, index: int) -> Image.Image:
    """Brightness/contrast jitter only.

    Geometry is left untouched, so landmark targets stay valid; the horizontal
    flip is applied on the network tensor by ``FrozenTokenExtractor.extract``.
    """
    image = image.convert("RGB")
    if index % 3 == 0:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
    return image


def normalize_points(
    points: Sequence[Sequence[float]],
    width: int,
    height: int,
    mode: str = "stretch",
    image_size: int = 224,
    coord_space: str = "normalized",
) -> np.ndarray:
    """Map source-image landmarks onto the network square.

    ``points`` are ``(x, y)`` pairs, either normalised to ``[0, 1]`` on the
    source image (``coord_space="normalized"``) or in source pixels
    (``coord_space="pixel"``). The return value is ``[L, 2]`` float32, normalised
    to ``[0, 1]`` on the square. Annotations that fall outside a crop or a
    letterbox are clipped to the square, because the head cannot represent a
    coordinate it never sees.
    """
    if coord_space not in COORD_SPACES:
        raise ValueError(f"Unknown coordinate space: {coord_space}")
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Unknown preprocess mode: {mode}")

    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if coord_space == "pixel":
        array = array / np.array([float(width), float(height)], dtype=np.float64)

    if mode == "stretch":
        mapped = array.copy()
    elif mode == "center_crop":
        side = float(min(width, height))
        left = float((width - side) // 2)
        top = float((height - side) // 2)
        mapped = np.stack(
            [
                (array[:, 0] * width - left) / side,
                (array[:, 1] * height - top) / side,
            ],
            axis=1,
        )
    else:  # letterbox
        scale = min(image_size / float(width), image_size / float(height))
        new_w = max(1, round(width * scale))
        new_h = max(1, round(height * scale))
        left = float((image_size - new_w) // 2)
        top = float((image_size - new_h) // 2)
        mapped = np.stack(
            [
                (array[:, 0] * width * scale + left) / float(image_size),
                (array[:, 1] * height * scale + top) / float(image_size),
            ],
            axis=1,
        )
    return np.clip(mapped, 0.0, 1.0).astype(np.float32)


def mirror_points(points: torch.Tensor) -> torch.Tensor:
    """Mirror normalised landmark coordinates horizontally on the square."""
    mirrored = points.clone()
    mirrored[..., 0] = 1.0 - mirrored[..., 0]
    return mirrored


# --------------------------------------------------------------------------- #
# Frozen backbone interface
# --------------------------------------------------------------------------- #
def _first_tensor(output: object) -> torch.Tensor:
    """Pick the token tensor out of a timm / Hugging Face forward result."""
    if isinstance(output, dict):
        for key in ("x", "last_hidden_state", "tokens", "x_norm_patchtokens"):
            if key in output:
                output = output[key]
                break
        else:
            candidates = [value for value in output.values() if torch.is_tensor(value)]
            if not candidates:
                raise TypeError("model output dict contains no tensor")
            output = candidates[0]
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    if not torch.is_tensor(output):
        raise TypeError(f"unsupported model output type: {type(output)!r}")
    return output


def _forward_tokens(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run ``forward_features`` when the backbone has it, else the module itself."""
    if hasattr(backbone, "forward_features"):
        return backbone.forward_features(x)
    return backbone(x)


def resolve_prefix_tokens(backbone: nn.Module, override: int | None = None) -> int:
    """Number of leading tokens to drop (class token plus registers)."""
    if override is not None:
        return int(override)
    return int(getattr(backbone, "num_prefix_tokens", DEFAULT_NUM_PREFIX_TOKENS))


class FrozenTokenExtractor:
    """Cache-free token extractor over a frozen timm ViT backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        image_size: int = 224,
        preprocess_mode: str = "stretch",
        num_prefix_tokens: int | None = None,
    ):
        self.backbone = backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.image_size = int(image_size)
        self.preprocess_mode = preprocess_mode
        self.num_prefix_tokens = resolve_prefix_tokens(backbone, num_prefix_tokens)

    @torch.no_grad()
    def extract(self, image: Image.Image, flip: bool = False) -> torch.Tensor:
        """Return the ``[1, N, D]`` token sequence, on CPU, in float32.

        ``flip`` mirrors the network square horizontally after preprocessing,
        which keeps the landmark mapping exactly ``x -> 1 - x``.
        """
        x = preprocess(image, size=self.image_size, mode=self.preprocess_mode)
        if flip:
            x = torch.flip(x, dims=[-1])
        tokens = _first_tensor(_forward_tokens(self.backbone, x.unsqueeze(0).to(self.device)))
        if tokens.ndim != 3:
            raise ValueError(
                "expected a [B, N, D] token sequence from the backbone, got "
                f"{tuple(tokens.shape)}"
            )
        return tokens.detach().float().cpu()


# --------------------------------------------------------------------------- #
# Head
# --------------------------------------------------------------------------- #
def tokens_to_grid(tokens: torch.Tensor, num_prefix_tokens: int = DEFAULT_NUM_PREFIX_TOKENS) -> torch.Tensor:
    """``[B, N, D]`` token sequence -> ``[B, D, H, W]`` token grid.

    Prefix tokens (class token, register tokens) are dropped. A square input
    gives a square grid for any ViT, so a non-square token count is rejected
    instead of being guessed at.
    """
    if tokens.ndim != 3:
        raise ValueError(f"expected a [B, N, D] token sequence, got {tuple(tokens.shape)}")
    prefix = int(num_prefix_tokens)
    if prefix < 0 or prefix >= tokens.shape[1]:
        raise ValueError(
            f"num_prefix_tokens={prefix} does not fit a sequence of {tokens.shape[1]} tokens"
        )
    patches = tokens[:, prefix:]
    count = int(patches.shape[1])
    side = math.isqrt(count)
    if side * side != count:
        raise ValueError(
            f"token grid is not square: {count} patch tokens. Pass a square "
            "--image-size, or a backbone with a square patch grid."
        )
    batch, _, dim = patches.shape
    return patches.transpose(1, 2).reshape(batch, dim, side, side).contiguous()


def soft_argmax_2d(heatmaps: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Spatial expectation of each heat map -> normalised ``[B, L, 2]`` coordinates.

    Grid cell ``i`` of ``W`` is placed at ``i / (W - 1)``, so the head can reach
    both borders of the square. With a uniform heat map every landmark sits at
    the centre of the grid.
    """
    if heatmaps.ndim != 4:
        raise ValueError(f"expected [B, L, H, W] heat maps, got {tuple(heatmaps.shape)}")
    batch, landmarks, height, width = heatmaps.shape
    logits = heatmaps.reshape(batch, landmarks, height * width)
    if temperature and temperature != 1.0:
        logits = logits / float(temperature)
    probabilities = logits.softmax(dim=-1).reshape(batch, landmarks, height, width)
    xs = torch.linspace(0.0, 1.0, width, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.linspace(0.0, 1.0, height, device=heatmaps.device, dtype=heatmaps.dtype)
    x = (probabilities.sum(dim=2) * xs).sum(dim=-1)
    y = (probabilities.sum(dim=3) * ys).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


class LocalizationHead(nn.Module):
    """Token sequence -> per-landmark normalised ``(x, y)``.

    Two 3x3 convolutions over the token grid, a 1x1 heat-map projection and a
    soft-argmax. The head is the only trainable part; the backbone stays frozen.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_landmarks: int = 2,
        hidden_dim: int = 128,
        num_prefix_tokens: int = DEFAULT_NUM_PREFIX_TOKENS,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_landmarks = int(num_landmarks)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.temperature = float(temperature)
        self.trunk = nn.Sequential(
            nn.Conv2d(self.embed_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.heat = nn.Conv2d(hidden_dim, self.num_landmarks, 1)

    def forward(self, tokens: torch.Tensor, num_prefix_tokens: int | None = None) -> torch.Tensor:
        """``[B, N, D]`` tokens -> ``[B, L, 2]`` coordinates in ``[0, 1]``."""
        prefix = self.num_prefix_tokens if num_prefix_tokens is None else int(num_prefix_tokens)
        heatmaps = self.heat(self.trunk(tokens_to_grid(tokens, prefix)))
        return soft_argmax_2d(heatmaps, self.temperature)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def coordinate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mode: str = "smooth_l1",
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth L1 (default) or MSE between normalised coordinates."""
    if loss_mode == "mse":
        return F.mse_loss(pred, target)
    if loss_mode == "smooth_l1":
        return F.smooth_l1_loss(pred, target, beta=beta)
    raise ValueError(f"Unknown localisation loss mode: {loss_mode}")


def l1_penalty(module: nn.Module) -> torch.Tensor:
    """Sum of ``|w|`` over the trainable parameters, for ``--l1-weight``."""
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        return torch.zeros(())
    return sum(parameter.abs().sum() for parameter in parameters)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def euclidean_distance_px(
    pred: torch.Tensor,
    target: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Per-landmark Euclidean distance in pixels at the input resolution.

    ``pred`` and ``target`` are ``[L, 2]`` (or ``[B, L, 2]``) normalised
    coordinates; the result is the same shape with the last dimension removed.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ")
    distance = torch.linalg.vector_norm((pred.float() - target.float()), dim=-1)
    return distance * float(image_size)


def localization_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    image_size: int,
    names: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Mean / median / max ED in px plus one entry per landmark."""
    distance = euclidean_distance_px(pred, target, image_size).reshape(-1)
    labels = list(names) if names is not None else [f"landmark_{i}" for i in range(distance.numel())]
    if len(labels) != distance.numel():
        raise ValueError(f"{len(labels)} names for {distance.numel()} landmarks")
    row: Dict[str, float] = {
        "mean_ed_px": float(distance.mean().item()),
        # ``torch.median`` returns the lower of the two middle values on an even
        # count; ``quantile`` gives the conventional median instead.
        "median_ed_px": float(torch.quantile(distance, 0.5).item()),
        "max_ed_px": float(distance.max().item()),
        "mean_ed_normalized": float((distance / float(image_size)).mean().item()),
    }
    for name, value in zip(labels, distance.tolist()):
        row[f"{name}_ed_px"] = float(value)
    return row


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    """Mean of every numeric metric in ``rows``."""
    if not rows:
        return {}
    keys = [key for key in rows[0] if isinstance(rows[0][key], float)]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """One image and its landmark annotations."""

    image_id: str
    image: str
    landmarks: Dict[str, Tuple[float, float]]


@dataclass
class TaskSplit:
    """Support/test ids for one landmark task.

    ``landmarks`` lists the annotated landmark names the head predicts jointly.
    It defaults to ``[task]``, which is the common one-landmark-per-task case.
    """

    task: str
    support: List[str] = field(default_factory=list)
    test: List[str] = field(default_factory=list)
    landmarks: List[str] = field(default_factory=list)

    @property
    def targets(self) -> List[str]:
        """Landmark names this task regresses."""
        return list(self.landmarks) or [self.task]


@dataclass
class Manifest:
    """Parsed dataset manifest plus the coordinate convention it declares."""

    samples: List[Sample]
    splits: Dict[str, TaskSplit]
    coord_space: str = "normalized"
    root: str = ""


def load_manifest(path: str | Path, data_root: str | Path | None = None) -> Manifest:
    """Read the JSON manifest that describes a landmark dataset.

    Expected layout::

        {
          "coord_space": "normalized",
          "samples": [
            {"image_id": "02", "image": "images/02.jpg",
             "landmarks": {"fovea": [0.412, 0.523], "optic_disc": [0.771, 0.468]}},
            ...
          ],
          "tasks": {
            "fovea": {"support": ["02", ...], "test": [...]},
            ...
          }
        }

    ``coord_space`` is ``normalized`` (fractions of the source image, the
    default) or ``pixel``. Image paths are absolute, or relative to
    ``data_root`` when given and to the manifest directory otherwise.
    """
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = Path(data_root).resolve() if data_root is not None else path.parent

    coord_space = str(payload.get("coord_space", "normalized")).lower()
    if coord_space not in COORD_SPACES:
        raise ValueError(
            f"coord_space must be one of {COORD_SPACES}, got {coord_space!r}"
        )

    samples = [
        Sample(
            image_id=str(record["image_id"]),
            image=str(_resolve(record["image"], root)),
            landmarks={
                str(task): _point(value)
                for task, value in record.get("landmarks", {}).items()
            },
        )
        for record in payload["samples"]
    ]

    splits = {
        str(task): TaskSplit(
            task=str(task),
            support=list(spec.get("support", [])),
            test=list(spec.get("test", [])),
            landmarks=[str(name) for name in spec.get("landmarks", [])],
        )
        for task, spec in payload.get("tasks", {}).items()
    }
    return Manifest(samples=samples, splits=splits, coord_space=coord_space, root=str(root))


def _resolve(raw: str, root: Path) -> Path:
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def _point(value: Sequence[float]) -> Tuple[float, float]:
    pair = [float(item) for item in value]
    if len(pair) != 2:
        raise ValueError(f"a landmark needs exactly two values, got {len(pair)}")
    return pair[0], pair[1]


def landmark_target(
    sample: Sample,
    task: str,
    width: int,
    height: int,
    image_size: int,
    mode: str = "stretch",
    coord_space: str = "normalized",
    names: Sequence[str] | None = None,
) -> torch.Tensor:
    """``[L, 2]`` target tensor for one sample and one task.

    ``names`` selects which annotations to stack; it defaults to the task name,
    i.e. a task that regresses exactly one landmark.
    """
    wanted = list(names) if names is not None else [task]
    missing = [name for name in wanted if name not in sample.landmarks]
    if missing:
        raise KeyError(
            f"sample {sample.image_id} has no annotation for {missing} "
            f"(available: {sorted(sample.landmarks)})"
        )
    points = normalize_points(
        [sample.landmarks[name] for name in wanted],
        width,
        height,
        mode,
        image_size,
        coord_space,
    )
    return torch.from_numpy(points)


def load_sample_and_target(
    sample: Sample,
    task: str,
    image_size: int,
    mode: str = "stretch",
    coord_space: str = "normalized",
    names: Sequence[str] | None = None,
) -> Tuple[Image.Image, torch.Tensor]:
    """Open the image and build its target on the same geometry."""
    image = load_rgb(sample.image)
    target = landmark_target(
        sample, task, image.width, image.height, image_size, mode, coord_space, names
    )
    return image, target


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def build_feature_sets(
    samples: Dict[str, Sample],
    task: str,
    train_ids: Sequence[str],
    extractor: FrozenTokenExtractor,
    aug_repeats: int,
    seed: int,
    coord_space: str = "normalized",
    names: Sequence[str] | None = None,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]],
    Dict[str, torch.Tensor],
]:
    """Extract original tokens, augmented copies and targets for the support ids.

    Odd repeats mirror the network square horizontally and flip ``x`` in the
    target, so an augmentation can never move a landmark.
    """
    original: Dict[str, torch.Tensor] = {}
    augmented: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    targets: Dict[str, torch.Tensor] = {}
    rng = random.Random(seed)
    for image_id in train_ids:
        sample = samples[image_id]
        image, target = load_sample_and_target(
            sample, task, extractor.image_size, extractor.preprocess_mode, coord_space, names
        )
        targets[image_id] = target
        original[image_id] = extractor.extract(image)
        variants: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for index in range(int(aug_repeats)):
            aug_image = photometric_augment(image, rng, index)
            flip = index % 2 == 1
            aug_target = mirror_points(target) if flip else target.clone()
            variants.append((extractor.extract(aug_image, flip=flip), aug_target))
        augmented[image_id] = variants
    return original, augmented, targets


def train_localization_head(
    train_ids: Sequence[str],
    original: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    augmented: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]],
    device: torch.device,
    epochs: int,
    lr: float,
    seed: int,
    hidden_dim: int = 128,
    loss_mode: str = "smooth_l1",
    beta: float = 1.0,
    l1_weight: float = 0.0,
    weight_decay: float = 1e-4,
    num_prefix_tokens: int = DEFAULT_NUM_PREFIX_TOKENS,
) -> LocalizationHead:
    """Fit the head on ``train_ids`` only, with the backbone already frozen."""
    if not train_ids:
        raise ValueError("train_localization_head needs at least one image")
    seed_everything(seed)
    embed_dim = int(original[train_ids[0]].shape[-1])
    num_landmarks = int(targets[train_ids[0]].shape[0])
    head = LocalizationHead(
        embed_dim=embed_dim,
        num_landmarks=num_landmarks,
        hidden_dim=hidden_dim,
        num_prefix_tokens=num_prefix_tokens,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    head.train()
    train_items: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for image_id in train_ids:
        train_items.append((original[image_id], targets[image_id]))
        train_items.extend(augmented[image_id])
    for _ in range(int(epochs)):
        random.shuffle(train_items)
        for tokens, target in train_items:
            prediction = head(tokens.to(device))
            loss = coordinate_loss(
                prediction, target.unsqueeze(0).to(device), loss_mode, beta
            )
            if l1_weight:
                loss = loss + float(l1_weight) * l1_penalty(head)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
    return head


@torch.no_grad()
def evaluate_head(
    head: LocalizationHead,
    ids: Sequence[str],
    features: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    device: torch.device,
    image_size: int,
    names: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    """Score one head on held-out ids; one row per image."""
    head.eval()
    rows: List[Dict[str, object]] = []
    for image_id in ids:
        prediction = head(features[image_id].to(device))[0].cpu()
        row: Dict[str, object] = {"id": image_id}
        row.update(localization_metrics(prediction, targets[image_id], image_size, names))
        rows.append(row)
    return rows


def select_epoch(
    support_ids: Sequence[str],
    original: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    augmented: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]],
    device: torch.device,
    epoch_candidates: Sequence[int],
    lr: float,
    seed: int,
    image_size: int,
    names: Sequence[str] | None = None,
    verbose: bool = True,
    **train_kwargs: object,
) -> Tuple[int, Dict[int, float]]:
    """Support-only leave-one-out selection of the epoch count.

    Each support image is held out in turn, the head is fitted on the remaining
    support images, and the held-out mean ED is recorded. **Test images are
    never consulted here.** Lower mean ED is better, so the minimum wins.
    """
    if len(support_ids) < 2:
        raise ValueError("support-only epoch selection needs at least two support images")
    scores: Dict[int, float] = {}
    for epochs in epoch_candidates:
        fold_scores: List[float] = []
        for fold, held_out in enumerate(support_ids):
            fit_ids = [item for item in support_ids if item != held_out]
            head = train_localization_head(
                fit_ids,
                original,
                targets,
                augmented,
                device,
                epochs,
                lr,
                seed + 1000 * fold + int(epochs),
                **train_kwargs,
            )
            row = evaluate_head(
                head, [held_out], original, targets, device, image_size, names
            )[0]
            fold_scores.append(float(row["mean_ed_px"]))
            del head
            if device.type == "cuda":
                torch.cuda.empty_cache()
        scores[int(epochs)] = float(np.mean(fold_scores))
        if verbose:
            print(
                f"support-CV epochs={epochs} mean_ed_px={scores[int(epochs)]:.3f}",
                flush=True,
            )
    best_epoch = min(scores, key=lambda item: (scores[item], item))
    if verbose:
        print(
            f"selected epochs={best_epoch} support-CV={scores[best_epoch]:.3f} px",
            flush=True,
        )
    return int(best_epoch), scores
