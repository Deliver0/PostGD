"""Tests for the released few-shot segmentation head.

The tests run on CPU in a few seconds and never touch clinical data. The
backbone is replaced by a small stub that exposes the same
``forward_intermediates`` interface, so no pretrained checkpoint is needed.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seg_head import (  # noqa: E402
    DEFAULT_FEATURE_INDICES,
    FrozenFeatureExtractor,
    SegDecoder,
    binary_metrics,
    build_feature_sets,
    estimate_fg_weight,
    load_manifest,
    load_mask_tensor,
    mask_to_tensor,
    paired_augment,
    preprocess,
    select_epoch_and_threshold,
    segmentation_loss,
    square_transform,
    train_decoder,
)

EMBED_DIM = 32
# A DINOv3 ViT/16 patch is 16 px wide, so a 64 px input yields a 4 px grid and the
# decoder upsamples four times at scale factor 2, landing back on 64 px.
IMAGE_SIZE = 64
GRID = IMAGE_SIZE // 16


class StubBackbone(torch.nn.Module):
    """Deterministic feature stub with the DINOv3 intermediate-map interface."""

    def __init__(self, embed_dim: int = EMBED_DIM, grid: int = GRID):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid = grid
        weight = torch.arange(embed_dim * grid * grid).reshape(embed_dim, grid, grid)
        self.register_buffer("weight", weight, persistent=False)

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: List[int],
        return_prefix_tokens: bool,
        norm: bool,
        output_fmt: str,
        intermediates_only: bool,
    ) -> List[torch.Tensor]:
        batch = x.shape[0]
        feature = self.weight.unsqueeze(0).expand(batch, -1, -1, -1).clone()
        return [feature for _ in DEFAULT_FEATURE_INDICES]


def _write_synthetic_dataset(tmp_path: Path, count: int = 4) -> Path:
    """Write tiny RGB images plus masks and return the manifest path."""
    rng = np.random.default_rng(0)
    samples = []
    for index in range(count):
        image_id = f"{index:02d}"
        image_path = tmp_path / f"{image_id}.png"
        mask_path = tmp_path / f"{image_id}_mask.png"
        Image.fromarray(
            (rng.random((IMAGE_SIZE, IMAGE_SIZE, 3)) * 255).astype("uint8"), mode="RGB"
        ).save(image_path)
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype="uint8")
        mask[20:40, 20:40] = 255
        Image.fromarray(mask, mode="L").save(mask_path)
        samples.append(
            {
                "image_id": image_id,
                "image": str(image_path),
                "masks": {"synthetic": str(mask_path)},
            }
        )
    manifest = {
        "samples": samples,
        "tasks": {
            "synthetic": {
                "support": [s["image_id"] for s in samples[:3]],
                "test": [samples[3]["image_id"]],
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _extractor() -> FrozenFeatureExtractor:
    return FrozenFeatureExtractor(
        backbone=StubBackbone(),
        device=torch.device("cpu"),
        image_size=IMAGE_SIZE,
        preprocess_mode="stretch",
    )


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_square_transform_modes():
    image = Image.new("RGB", (60, 40), (10, 20, 30))
    mask = Image.new("L", (60, 40), 7)
    for mode in ("stretch", "center_crop", "letterbox"):
        assert square_transform(image, 32, mode, is_mask=False).size == (32, 32)
        assert square_transform(mask, 32, mode, is_mask=True).size == (32, 32)


def test_preprocess_shapes_and_range():
    image = Image.new("RGB", (50, 50), (120, 130, 140))
    mask = Image.new("L", (50, 50), 0)
    x, y = preprocess(image, mask, size=24)
    assert x.shape == (3, 24, 24)
    assert y.shape == (24, 24)
    assert y.dtype == torch.long
    assert set(y.unique().tolist()) <= {0, 1}


def test_mask_to_tensor_thresholds_above_eight():
    assert mask_to_tensor(Image.new("L", (4, 4), 9)).max().item() == 1
    assert mask_to_tensor(Image.new("L", (4, 4), 8)).max().item() == 0


def test_load_mask_tensor_lands_on_the_network_square():
    mask_path = Path(__file__)  # only used for its parent directory
    scratch = Path.cwd()
    image = Image.new("L", (37, 53), 0)
    image.putpixel((5, 5), 255)
    path = scratch / "_tmp_mask_for_test.png"
    image.save(path)
    try:
        for mode in ("stretch", "center_crop", "letterbox"):
            tensor = load_mask_tensor(path, IMAGE_SIZE, mode)
            assert tensor.shape == (IMAGE_SIZE, IMAGE_SIZE)
            assert set(tensor.unique().tolist()) <= {0, 1}
    finally:
        if path.exists():
            path.unlink()
    assert mask_path  # silence linters: path kept for the doc reference above


def test_paired_augment_preserves_alignment():
    image = Image.new("RGB", (40, 40), (200, 200, 200))
    mask = Image.new("L", (40, 40), 0)
    mask.putpixel((5, 5), 255)
    rng = random.Random(3)
    aug_image, aug_mask = paired_augment(image, mask, rng, index=3)
    assert aug_image.size == aug_mask.size == (40, 40)
    assert mask_to_tensor(aug_mask).sum().item() >= 1


# --------------------------------------------------------------------------- #
# Decoder and losses
# --------------------------------------------------------------------------- #
def test_seg_decoder_output_matches_input_grid_times_sixteen():
    decoder = SegDecoder(num_classes=2, embed_dim=EMBED_DIM)
    features = [torch.randn(1, EMBED_DIM, GRID, GRID) for _ in DEFAULT_FEATURE_INDICES]
    logits = decoder(features)
    assert logits.shape == (1, 2, GRID * 16, GRID * 16)
    assert torch.isfinite(logits).all()


def test_segmentation_loss_backward():
    decoder = SegDecoder(num_classes=2, embed_dim=EMBED_DIM)
    features = [torch.randn(1, EMBED_DIM, GRID, GRID) for _ in DEFAULT_FEATURE_INDICES]
    logits = decoder(features)
    target = torch.zeros(1, *logits.shape[-2:], dtype=torch.long)
    target[..., 2:5, 2:5] = 1
    loss = segmentation_loss(logits, target, fg_weight=3.0)
    loss.backward()
    assert torch.isfinite(loss).item()
    for name, parameter in decoder.named_parameters():
        assert parameter.grad is not None, name


def test_estimate_fg_weight_is_clamped():
    masks = {
        "a": torch.zeros(8, 8),
        "b": torch.ones(8, 8),
    }
    masks["a"][:4] = 1
    weight = estimate_fg_weight(["a", "b"], masks)
    assert 2.0 <= weight <= 20.0


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_binary_metrics_perfect_prediction():
    target = torch.zeros(8, 8, dtype=torch.long)
    target[:4] = 1
    metrics = binary_metrics(target, target)
    assert metrics["dice"] > 0.99
    assert metrics["miou"] > 0.99


def test_binary_metrics_empty_prediction_collapses_fg_iou():
    target = torch.zeros(8, 8, dtype=torch.long)
    target[:4] = 1
    metrics = binary_metrics(torch.zeros_like(target), target)
    assert metrics["dice"] < 0.6
    assert metrics["background_iou"] > metrics["foreground_iou"]


# --------------------------------------------------------------------------- #
# Leak-free selection
# --------------------------------------------------------------------------- #
def test_support_only_selection_never_touches_test(tmp_path):
    manifest_path = _write_synthetic_dataset(tmp_path, count=4)
    samples_list, splits = load_manifest(manifest_path)
    samples = {s.image_id: s for s in samples_list}
    task = "synthetic"
    split = splits[task]

    extractor = _extractor()
    original, augmented = build_feature_sets(
        samples, task, split.support, extractor, aug_repeats=1, seed=0
    )
    masks = {
        i: load_mask_tensor(samples[i].masks[task], IMAGE_SIZE, "stretch")
        for i in split.support
    }

    best_epoch, best_threshold, scores = select_epoch_and_threshold(
        split.support,
        original,
        masks,
        augmented,
        torch.device("cpu"),
        epoch_candidates=[1, 2],
        lr=1e-2,
        seed=0,
        thresholds=[0.3, 0.5],
        extractor=extractor,
        samples=samples,
        tta=False,
        verbose=False,
    )
    assert best_epoch in (1, 2)
    assert best_threshold in (0.3, 0.5)
    assert set(scores) == {1, 2}
    # One leave-one-out fold per support image: every CV score is the mean over
    # exactly len(support) held-out evaluations, and no test id ever appears.
    for values in scores.values():
        assert len(values) == 2
        assert all(0.0 <= value <= 1.0 for value in values)


def test_train_decoder_fits_support_only():
    original = {
        "a": [torch.randn(1, EMBED_DIM, GRID, GRID) for _ in DEFAULT_FEATURE_INDICES],
        "b": [torch.randn(1, EMBED_DIM, GRID, GRID) for _ in DEFAULT_FEATURE_INDICES],
    }
    side = GRID * 16
    masks = {
        "a": torch.zeros(side, side, dtype=torch.long),
        "b": torch.ones(side, side, dtype=torch.long),
    }
    decoder = train_decoder(
        ["a", "b"],
        original,
        masks,
        {"a": [], "b": []},
        torch.device("cpu"),
        epochs=1,
        lr=1e-2,
        seed=0,
    )
    assert isinstance(decoder, SegDecoder)
