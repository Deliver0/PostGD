from pathlib import Path
import sys
import random

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import prior_guidance  # noqa: E402
import spot_renderer  # noqa: E402
from spot_renderer import (
    RenderConfig,
    ZoneProfile,
    _spot_patch,
    plan_spots,
    render_image,
)  # noqa: E402


def test_zone_profile_reaches_bright_and_background():
    profile = ZoneProfile(0.5, 0.5, 0.1, 0.2, 0.2)
    radii = np.array([0.0, 0.49, 0.55, 0.65, 0.79, 0.9, 1.0], dtype=np.float32)
    values = profile.multiplier(radii)
    assert values[0] == -0.5
    assert values[2] > 0.0
    assert values[3] == 1.0
    assert values[-1] == 0.0


def test_fade_zone_smoothly_reaches_background():
    profile = ZoneProfile(0.2, 0.0, 0.0, 0.5, 0.5)
    radii = np.array([0.5, 0.625, 0.75, 0.875, 1.0], dtype=np.float32)
    values = profile.multiplier(radii)

    assert np.allclose(values, [1.0, 0.84375, 0.5, 0.15625, 0.0], atol=1e-6)
    assert values[0] - values[1] < values[1] - values[2]
    assert values[2] - values[3] > values[3] - values[4]


def test_inner_transition_has_smooth_endpoints():
    profile = ZoneProfile(0.3, 0.2, 0.4, 0.05, 0.35)
    radii = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.775, 1.0], dtype=np.float32)
    values = profile.multiplier(radii)

    assert np.allclose(values[[0, 4, -1]], [-0.3, 1.0, 0.0], atol=1e-6)
    assert values[1] < 0.025
    assert values[3] > 0.65


def test_ring_mode_separates_narrow_core_from_soft_shoulders():
    profile = ZoneProfile(
        ring_mode=True,
        ring_center=0.6,
        ring_width=0.1,
        ring_inner_fade=0.2,
        ring_outer_fade=0.2,
        ring_shoulder_power=2.0,
    )
    radii = np.array([0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9], dtype=np.float32)
    values = profile.multiplier(radii)

    assert values[0] < 0.0
    assert values[2] < values[3] < values[4]
    assert values[4] > values[5] > values[6] > values[7]
    assert values[4] == 1.0
    assert values[-1] == 0.0


def test_ring_mode_rejects_pixel_like_widths():
    profile = ZoneProfile(ring_mode=True, ring_width=10.0)
    with np.testing.assert_raises(ValueError):
        profile.multiplier(np.array([0.5], dtype=np.float32))


def test_render_is_deterministic_and_has_no_rectangle_change():
    image = Image.fromarray(np.full((256, 256, 3), 100, dtype=np.uint8), "RGB")
    config = RenderConfig(num_spots=0, seed=7)
    points = [(128.0, 128.0, 1.0)]
    first, mask = render_image(image, points, config)
    second, _ = render_image(image, points, config)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    changed = np.any(np.asarray(first) != np.asarray(image), axis=2)
    assert np.all(~changed | (np.asarray(mask) > 0))


def test_spot_contours_are_irregular_bounded_and_repeatable():
    args = (24.0, ZoneProfile())
    multiplier_a, mask_a, half_a = _spot_patch(*args, random.Random(17))
    multiplier_b, mask_b, half_b = _spot_patch(*args, random.Random(17))
    _, mask_other, _ = _spot_patch(*args, random.Random(18))

    assert half_a == half_b
    assert np.array_equal(multiplier_a, multiplier_b)
    assert np.array_equal(mask_a, mask_b)
    assert not np.array_equal(mask_a, mask_other)

    y, x = np.nonzero(mask_a)
    radii = np.hypot(x - half_a, y - half_a)
    bins = np.floor((np.arctan2(y - half_a, x - half_a) + np.pi) * 36 / (2 * np.pi)).astype(int)
    outer = np.array([radii[bins == i].max() for i in range(36)])
    assert np.ptp(outer) > 4.0
    assert outer.max() <= 24.0 * spot_renderer.MAX_SPOT_SHAPE_SCALE + 1.0


def test_dense_vessel_prediction_is_never_discarded(monkeypatch):
    dense_vessels = np.ones((128, 128), dtype=bool)
    monkeypatch.setattr(
        prior_guidance, "_load_vessels", lambda image, cfg, device: (dense_vessels, "ok")
    )
    monkeypatch.setattr(
        prior_guidance, "_load_fovea_od", lambda image, cfg, device: (None, "unavailable")
    )
    cfg = prior_guidance.PriorConfig(device="cpu", vessel_buffer_px=0, posterior_pole_radius=1)

    prior = prior_guidance.build_prior(
        Image.new("RGB", (128, 128)), cfg, spot_radius=12
    )

    assert prior["vessel_mask"].all()
    assert prior["avoidance_mask"].all()
    assert "mask 100.0%" in prior["status"]["vessel"]


def test_missing_vessel_prediction_blocks_all_spots(monkeypatch):
    monkeypatch.setattr(
        prior_guidance, "_load_vessels", lambda image, cfg, device: (None, "checkpoint unavailable")
    )
    monkeypatch.setattr(
        prior_guidance, "_load_fovea_od", lambda image, cfg, device: (None, "unavailable")
    )
    prior = prior_guidance.build_prior(
        Image.new("RGB", (128, 128)),
        prior_guidance.PriorConfig(device="cpu"),
        spot_radius=12,
    )
    points = plan_spots(
        np.ones((128, 128), dtype=bool),
        RenderConfig(num_spots=10, seed=1),
        avoidance_mask=prior["avoidance_mask"],
    )

    assert prior["vessel_mask"] is None
    assert prior["status"]["vessel"] == "checkpoint unavailable"
    assert points == []


def _assert_rendered_spots_avoid_vessels():
    fov = np.ones((192, 192), dtype=bool)
    vessel = np.zeros_like(fov)
    vessel[40:152, 94:98] = True
    config = RenderConfig(
        num_spots=30, seed=23, spot_size=0.5, gap=0.2, central_exclusion=0
    )

    points = plan_spots(fov, config, avoidance_mask=vessel)
    _, spot_mask = render_image(Image.new("RGB", (192, 192), "white"), points, config)

    assert 0 < len(points) <= config.num_spots
    assert not np.any((np.asarray(spot_mask) > 0) & vessel)


def test_spot_footprints_avoid_vessels_with_opencv():
    if spot_renderer.cv2 is None:
        return
    _assert_rendered_spots_avoid_vessels()


def test_spot_footprints_avoid_vessels_without_opencv(monkeypatch):
    monkeypatch.setattr(spot_renderer, "cv2", None)
    _assert_rendered_spots_avoid_vessels()
