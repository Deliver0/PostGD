"""Tests for the released few-shot landmark localisation head.

The tests run on CPU in a few seconds and never touch clinical data. The
backbone is replaced by a small stub that exposes the same
``forward_features`` token-sequence interface as a timm ViT, so no pretrained
checkpoint is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from few_shot_localization import load_config, parse_args  # noqa: E402
from localization_head import (  # noqa: E402
    FrozenTokenExtractor,
    LocalizationHead,
    build_feature_sets,
    coordinate_loss,
    euclidean_distance_px,
    evaluate_head,
    landmark_target,
    load_manifest,
    load_sample_and_target,
    localization_metrics,
    mirror_points,
    normalize_points,
    resolve_prefix_tokens,
    select_epoch,
    soft_argmax_2d,
    tokens_to_grid,
    train_localization_head,
)

EMBED_DIM = 32
IMAGE_SIZE = 64
GRID = IMAGE_SIZE // 16
NUM_PREFIX_TOKENS = 2  # one class token plus one register token


class StubBackbone(torch.nn.Module):
    """Deterministic token stub with the timm ``forward_features`` interface.

    The patch tokens are the fixed weight grid modulated by the pooled input, so
    a horizontally flipped image really does produce different tokens.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, grid: int = GRID):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid = grid
        self.num_prefix_tokens = NUM_PREFIX_TOKENS
        self.num_features = embed_dim
        weight = torch.arange(embed_dim * grid * grid).reshape(embed_dim, grid, grid)
        self.register_buffer("weight", weight, persistent=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.grid, self.grid))
        signal = pooled.mean(dim=1, keepdim=True)
        patches = self.weight.unsqueeze(0) * (1.0 + signal)
        patches = patches.flatten(2).transpose(1, 2)
        prefix = torch.zeros(batch, self.num_prefix_tokens, self.embed_dim)
        return torch.cat([prefix, patches], dim=1)


def _extractor() -> FrozenTokenExtractor:
    return FrozenTokenExtractor(
        backbone=StubBackbone(),
        device=torch.device("cpu"),
        image_size=IMAGE_SIZE,
        preprocess_mode="stretch",
    )


def _write_synthetic_dataset(tmp_path: Path, count: int = 4) -> Path:
    """Write tiny RGB images plus landmark annotations and return the manifest path."""
    rng = np.random.default_rng(0)
    samples = []
    for index in range(count):
        image_id = f"{index:02d}"
        image_path = tmp_path / f"{image_id}.png"
        Image.fromarray(
            (rng.random((IMAGE_SIZE, IMAGE_SIZE, 3)) * 255).astype("uint8"), mode="RGB"
        ).save(image_path)
        samples.append(
            {
                "image_id": image_id,
                "image": str(image_path),
                "landmarks": {
                    "fovea": [0.25 + 0.1 * index, 0.5],
                    "optic_disc": [0.75 - 0.1 * index, 0.5],
                },
            }
        )
    # The first three images support every task, the last one is held out.
    support = [s["image_id"] for s in samples[:-1]]
    held_out = [samples[-1]["image_id"]]
    manifest = {
        "coord_space": "normalized",
        "samples": samples,
        "tasks": {
            "fovea": {"support": support, "test": held_out},
            "both": {
                "support": support,
                "test": held_out,
                "landmarks": ["fovea", "optic_disc"],
            },
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_normalize_points_is_identity_for_square_stretch():
    points = normalize_points([[0.1, 0.2], [0.9, 0.8]], 512, 512, "stretch", 224)
    assert points.shape == (2, 2)
    assert np.allclose(points, [[0.1, 0.2], [0.9, 0.8]], atol=1e-6)


def test_normalize_points_accepts_pixel_coordinates():
    # 256 x 128 px in a 512 x 256 source is the centre of the image.
    normalised = normalize_points([[0.5, 0.5]], 512, 256, "stretch", 224, "normalized")
    pixels = normalize_points([[256.0, 128.0]], 512, 256, "stretch", 224, "pixel")
    assert np.allclose(normalised, pixels, atol=1e-6)
    assert np.allclose(pixels, [[0.5, 0.5]], atol=1e-6)


def test_normalize_points_center_crop_matches_the_pixel_crop():
    # 512x256 source -> a centred 256x256 crop; the crop's centre is the image centre.
    cropped = normalize_points([[0.5, 0.5]], 512, 256, "center_crop", 224)
    assert np.allclose(cropped, [[0.5, 0.5]], atol=1e-6)
    # A point on the crop's left border maps to 0, the source border is clipped.
    left = normalize_points([[128.0 / 512.0, 0.5]], 512, 256, "center_crop", 224)
    assert np.allclose(left, [[0.0, 0.5]], atol=1e-6)


def test_normalize_points_letterbox_preserves_aspect_ratio():
    # 256x128 source letterboxed into 224 -> 224x112 pasted at y offset 56.
    top = normalize_points([[0.5, 0.0]], 256, 128, "letterbox", 224)
    bottom = normalize_points([[0.5, 1.0]], 256, 128, "letterbox", 224)
    assert np.allclose(top, [[0.5, 56.0 / 224.0]], atol=1e-6)
    assert np.allclose(bottom, [[0.5, 168.0 / 224.0]], atol=1e-6)


def test_mirror_points_flips_only_x():
    mirrored = mirror_points(torch.tensor([[0.25, 0.4]]))
    assert torch.allclose(mirrored, torch.tensor([[0.75, 0.4]]))


def test_token_extractor_matches_the_backbone_sequence():
    extractor = _extractor()
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (120, 130, 140))
    tokens = extractor.extract(image)
    assert tokens.shape == (1, NUM_PREFIX_TOKENS + GRID * GRID, EMBED_DIM)
    assert extractor.num_prefix_tokens == NUM_PREFIX_TOKENS


def test_resolve_prefix_tokens_prefers_the_backbone_attribute():
    backbone = StubBackbone()
    assert resolve_prefix_tokens(backbone) == NUM_PREFIX_TOKENS
    assert resolve_prefix_tokens(backbone, 1) == 1


# --------------------------------------------------------------------------- #
# Head
# --------------------------------------------------------------------------- #
def test_tokens_to_grid_drops_prefix_tokens():
    tokens = torch.randn(2, NUM_PREFIX_TOKENS + GRID * GRID, EMBED_DIM)
    grid = tokens_to_grid(tokens, NUM_PREFIX_TOKENS)
    assert grid.shape == (2, EMBED_DIM, GRID, GRID)


def test_tokens_to_grid_rejects_a_non_square_grid():
    tokens = torch.randn(1, NUM_PREFIX_TOKENS + GRID * (GRID + 1), EMBED_DIM)
    try:
        tokens_to_grid(tokens, NUM_PREFIX_TOKENS)
    except ValueError as exc:
        assert "not square" in str(exc)
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("a non-square token grid must be rejected")


def test_soft_argmax_returns_the_centre_for_a_uniform_map():
    heatmaps = torch.zeros(1, 2, 5, 5)
    coordinates = soft_argmax_2d(heatmaps)
    assert coordinates.shape == (1, 2, 2)
    assert torch.allclose(coordinates, torch.full_like(coordinates, 0.5), atol=1e-6)


def test_soft_argmax_reaches_the_grid_corners():
    heatmaps = torch.full((1, 1, 3, 3), -20.0)
    heatmaps[0, 0, 0, 0] = 20.0
    top_left = soft_argmax_2d(heatmaps)
    assert torch.allclose(top_left[0, 0], torch.tensor([0.0, 0.0]), atol=1e-3)
    heatmaps = torch.full((1, 1, 3, 3), -20.0)
    heatmaps[0, 0, 2, 2] = 20.0
    bottom_right = soft_argmax_2d(heatmaps)
    assert torch.allclose(bottom_right[0, 0], torch.tensor([1.0, 1.0]), atol=1e-3)


def test_localization_head_forward_shape():
    head = LocalizationHead(
        embed_dim=EMBED_DIM, num_landmarks=2, num_prefix_tokens=NUM_PREFIX_TOKENS
    )
    tokens = torch.randn(3, NUM_PREFIX_TOKENS + GRID * GRID, EMBED_DIM)
    coordinates = head(tokens)
    assert coordinates.shape == (3, 2, 2)
    assert torch.isfinite(coordinates).all()
    assert bool((coordinates >= 0.0).all() and (coordinates <= 1.0).all())


def test_localization_head_one_landmark_matches_a_single_task_manifest():
    head = LocalizationHead(
        embed_dim=EMBED_DIM, num_landmarks=1, num_prefix_tokens=NUM_PREFIX_TOKENS
    )
    tokens = torch.randn(1, NUM_PREFIX_TOKENS + GRID * GRID, EMBED_DIM)
    assert head(tokens).shape == (1, 1, 2)


# --------------------------------------------------------------------------- #
# Loss and metrics
# --------------------------------------------------------------------------- #
def test_coordinate_loss_backward_reaches_every_parameter():
    head = LocalizationHead(
        embed_dim=EMBED_DIM, num_landmarks=2, num_prefix_tokens=NUM_PREFIX_TOKENS
    )
    tokens = torch.randn(2, NUM_PREFIX_TOKENS + GRID * GRID, EMBED_DIM)
    target = torch.tensor([[[0.2, 0.3], [0.7, 0.8]], [[0.4, 0.5], [0.6, 0.1]]])
    loss = coordinate_loss(head(tokens), target, "smooth_l1")
    loss.backward()
    assert torch.isfinite(loss).item()
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None, name


def test_euclidean_distance_is_hand_computable_in_pixels():
    # (0.5, 0.5) -> (0.5, 0.75) is a quarter of the square: 0.25 * 224 = 56 px.
    pred = torch.tensor([[0.5, 0.5]])
    target = torch.tensor([[0.5, 0.75]])
    assert torch.allclose(euclidean_distance_px(pred, target, 224), torch.tensor([56.0]))


def test_euclidean_distance_uses_the_pythagorean_norm():
    # A 3-4-5 triangle on the unit square scaled to a 1024 px input: 0.5 * 1024.
    pred = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[0.3, 0.4]])
    assert torch.allclose(
        euclidean_distance_px(pred, target, 1024), torch.tensor([512.0]), atol=1e-4
    )


def test_localization_metrics_reports_mean_median_max_and_per_landmark():
    pred = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    target = torch.tensor([[0.5, 0.75], [0.5, 0.5]])
    metrics = localization_metrics(pred, target, 224, names=["fovea", "optic_disc"])
    assert abs(metrics["fovea_ed_px"] - 56.0) < 1e-4
    assert abs(metrics["optic_disc_ed_px"] - 0.0) < 1e-4
    assert abs(metrics["mean_ed_px"] - 28.0) < 1e-4
    assert abs(metrics["max_ed_px"] - 56.0) < 1e-4
    assert abs(metrics["median_ed_px"] - 28.0) < 1e-4
    assert abs(metrics["mean_ed_normalized"] - 0.125) < 1e-6


def test_localization_metrics_rejects_a_name_count_mismatch():
    pred = torch.zeros(2, 2)
    try:
        localization_metrics(pred, pred, 224, names=["only_one"])
    except ValueError as exc:
        assert "names" in str(exc)
    else:  # pragma: no cover - the call above must raise
        raise AssertionError("a name/landmark count mismatch must be rejected")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def test_load_manifest_reads_coord_space_and_task_landmarks(tmp_path):
    manifest = load_manifest(_write_synthetic_dataset(tmp_path, count=4))
    assert manifest.coord_space == "normalized"
    assert set(manifest.splits) == {"fovea", "both"}
    assert manifest.splits["fovea"].targets == ["fovea"]
    assert manifest.splits["both"].targets == ["fovea", "optic_disc"]
    assert len(manifest.samples) == 4


def test_landmark_target_stacks_every_requested_landmark(tmp_path):
    manifest = load_manifest(_write_synthetic_dataset(tmp_path, count=1))
    sample = manifest.samples[0]
    single = landmark_target(sample, "fovea", IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE)
    both = landmark_target(
        sample, "both", IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, names=["fovea", "optic_disc"]
    )
    assert single.shape == (1, 2)
    assert both.shape == (2, 2)
    assert torch.allclose(single[0], both[0])


def test_load_sample_and_target_keeps_image_and_target_in_sync(tmp_path):
    manifest = load_manifest(_write_synthetic_dataset(tmp_path, count=1))
    image, target = load_sample_and_target(
        manifest.samples[0], "fovea", IMAGE_SIZE, "stretch", manifest.coord_space
    )
    assert image.size == (IMAGE_SIZE, IMAGE_SIZE)
    assert target.shape == (1, 2)
    assert bool((target >= 0.0).all() and (target <= 1.0).all())


# --------------------------------------------------------------------------- #
# Training / evaluation / leak-free selection
# --------------------------------------------------------------------------- #
def test_five_shot_training_one_epoch_does_not_crash(tmp_path):
    """The acceptance case: five support images, one epoch, finite loss."""
    manifest = load_manifest(_write_synthetic_dataset(tmp_path, count=6))
    samples = {sample.image_id: sample for sample in manifest.samples}
    support = [sample.image_id for sample in manifest.samples[:5]]
    extractor = _extractor()

    original, augmented, targets = build_feature_sets(
        samples, "fovea", support, extractor, aug_repeats=1, seed=0
    )
    assert len(original) == 5
    assert len(targets) == 5
    assert all(tokens.shape == (1, NUM_PREFIX_TOKENS + GRID * GRID, EMBED_DIM)
               for tokens in original.values())

    head = train_localization_head(
        support,
        original,
        targets,
        augmented,
        torch.device("cpu"),
        epochs=1,
        lr=1e-2,
        seed=0,
        num_prefix_tokens=NUM_PREFIX_TOKENS,
    )
    assert isinstance(head, LocalizationHead)
    rows = evaluate_head(
        head, support, original, targets, torch.device("cpu"), IMAGE_SIZE, ["fovea"]
    )
    assert len(rows) == 5
    for row in rows:
        assert row["id"] in support
        assert np.isfinite(row["mean_ed_px"])
        assert row["mean_ed_px"] >= 0.0


def test_flip_augmentation_keeps_the_target_aligned(tmp_path):
    """A mirrored view must be paired with a mirrored landmark."""
    manifest = load_manifest(_write_synthetic_dataset(tmp_path, count=1))
    samples = {sample.image_id: sample for sample in manifest.samples}
    extractor = _extractor()
    _, augmented, targets = build_feature_sets(
        samples, "fovea", ["00"], extractor, aug_repeats=2, seed=0
    )
    base = targets["00"]
    unchanged, flipped = augmented["00"]
    assert torch.allclose(unchanged[1], base)
    assert torch.allclose(flipped[1], mirror_points(base))
    # index 0 is unflipped, index 1 is flipped, so the token tensors must differ.
    assert not torch.allclose(unchanged[0], flipped[0])


def test_support_only_selection_never_touches_test(tmp_path):
    manifest = load_manifest(_write_synthetic_dataset(tmp_path, count=5))
    samples = {sample.image_id: sample for sample in manifest.samples}
    support = [sample.image_id for sample in manifest.samples[:3]]
    test_id = manifest.samples[4].image_id
    extractor = _extractor()

    original, augmented, targets = build_feature_sets(
        samples, "fovea", support, extractor, aug_repeats=1, seed=0
    )
    best_epoch, scores = select_epoch(
        support,
        original,
        targets,
        augmented,
        torch.device("cpu"),
        epoch_candidates=[1, 2],
        lr=1e-2,
        seed=0,
        image_size=IMAGE_SIZE,
        names=["fovea"],
        verbose=False,
        num_prefix_tokens=NUM_PREFIX_TOKENS,
    )
    assert best_epoch in (1, 2)
    assert set(scores) == {1, 2}
    assert test_id not in original
    # One leave-one-out fold per support image, each scored in pixels.
    for value in scores.values():
        assert np.isfinite(value)
        assert value >= 0.0


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def test_config_file_supplies_defaults_and_flags_win(tmp_path):
    config = tmp_path / "localization.yaml"
    config.write_text(
        "\n".join(
            [
                "version: 1",
                "image_size: 512",
                "beta: 0.25",
                "aug_repeats: 0",
                "model_kwargs:",
                "  img_size: 512",
                "num_prefix_tokens: null",
            ]
        ),
        encoding="utf-8",
    )
    base = ["--config", str(config), "--manifest", "m.json", "--output", "out"]
    defaults = parse_args(base)
    assert defaults.image_size == 512
    assert abs(defaults.beta - 0.25) < 1e-9
    assert defaults.aug_repeats == 0
    # A mapping in YAML is turned into the JSON string the flag expects.
    assert json.loads(defaults.model_kwargs) == {"img_size": 512}
    # A null in the config falls back to the flag's own default.
    assert defaults.num_prefix_tokens is None

    overridden = parse_args(base + ["--image-size", "224"])
    assert overridden.image_size == 224
    assert abs(overridden.beta - 0.25) < 1e-9


def test_json_config_is_accepted_without_yaml(tmp_path):
    # A .json config must not require PyYAML; it is read with json.loads.
    config = tmp_path / "localization.json"
    config.write_text(
        json.dumps({"version": 1, "image_size": 320, "epochs": [1]}), encoding="utf-8"
    )
    args = parse_args(
        ["--config", str(config), "--manifest", "m.json", "--output", "out"]
    )
    assert args.image_size == 320
    assert args.epochs == [1]


def test_parse_args_requires_a_manifest_and_an_output():
    for argv in (
        ["--output", "out"],
        ["--manifest", "m.json"],
    ):
        try:
            parse_args(argv)
        except SystemExit as exc:
            assert exc.code == 2
        else:  # pragma: no cover - the calls above must exit
            raise AssertionError(f"{argv} must be rejected")
