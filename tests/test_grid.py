"""PRP/GRID integration checks with deterministic synthetic anatomical priors."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=['full_method', 'render_only'])
def pipeline(request, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    src = root / 'stage1' / request.param / 'src'
    prior = load_module(f'grid_prior_{request.param}', src / 'prior_guidance.py')
    monkeypatch.setitem(sys.modules, 'prior_guidance', prior)
    renderer = load_module(f'grid_renderer_{request.param}', src / 'spot_renderer.py')
    vessel = np.zeros((395, 518), dtype=bool)
    vessel[:, 329:335] = True
    monkeypatch.setattr(prior, '_load_vessels', lambda *args: (vessel, 'ok'))
    monkeypatch.setattr(prior, '_load_fovea_od', lambda *args: ((300., 200., 180., 200.), 'ok'))
    return root, prior, renderer


@pytest.mark.parametrize('mode', ['prp', 'grid'])
def test_centers_obey_treatment_regions_and_fov(pipeline, mode):
    _, module, renderer = pipeline
    image = Image.new('RGB', (518, 395), 'white')
    cfg = renderer.RenderConfig(treatment=mode, posterior_pole_radius=120,
                                spot_size=0.5, num_spots=30, seed=12)
    prior = module.build_prior(image, module.PriorConfig(treatment=mode, device='cpu',
                                                        posterior_pole_radius=120), spot_radius=3)
    expected_center = (180., 200.) if mode == 'prp' else (300., 200.)
    assert prior['center'] == expected_center
    fov = np.ones((395, 518), dtype=bool)
    fov[:, 400:] = False
    points = renderer.plan_spots(fov, cfg, center=prior['center'], avoidance_mask=prior['avoidance_mask'])
    assert 0 < len(points) <= cfg.num_spots
    assert points == renderer.plan_spots(fov, cfg, center=prior['center'], avoidance_mask=prior['avoidance_mask'])
    for x, y, _ in points:
        ix, iy = round(x), round(y)
        assert fov[iy, ix]
        assert not prior['vessel_mask'][iy, ix]
        assert not prior['optic_disc_mask'][iy, ix]
        assert bool(prior['posterior_pole_mask'][iy, ix]) == (mode == 'grid')


def test_grid_render_exports_mode_and_foveal_center(pipeline, tmp_path):
    _, _, renderer = pipeline
    image = tmp_path / 'background.png'
    Image.new('RGB', (518, 395), (90, 50, 30)).save(image)
    cfg = renderer.RenderConfig(treatment='grid', posterior_pole_radius=120,
                                spot_size=0.5, num_spots=15, device='cpu')
    meta = renderer.render_background(image, tmp_path / 'output', cfg)
    assert meta['treatment'] == meta['config']['treatment'] == 'grid'
    assert meta['prior']['center'] == (300., 200.)
    assert meta['generated_spots'] > 0
    assert Path(meta['outputs']['image']).is_file()
    assert np.asarray(Image.open(meta['outputs']['mask'])).any()


def test_grid_requires_priors_and_prp_is_default(pipeline):
    _, prior, renderer = pipeline
    assert renderer.RenderConfig().treatment == prior.PriorConfig().treatment == 'prp'
    with pytest.raises(ValueError, match='GRID requires anatomical priors'):
        renderer.RenderConfig(treatment='grid', no_prior=True).validate()
    with pytest.raises(ValueError, match='GRID requires anatomical priors'):
        prior.build_prior(Image.new('RGB', (518, 395)),
                          prior.PriorConfig(treatment='grid', enabled=False), spot_radius=2)


@pytest.mark.parametrize('pipeline', ['full_method'], indirect=True)
def test_full_generator_records_grid_and_refuses_mixed_resume(pipeline, monkeypatch, tmp_path):
    root, prior, renderer = pipeline
    # Use the actual batch entry point with local backgrounds and no color fitting.
    monkeypatch.setitem(sys.modules, 'spot_renderer', renderer)
    monkeypatch.setitem(sys.modules, 'prior_guidance', prior)
    generator = load_module('grid_full_generator', root / 'stage1' / 'full_method' / 'generate.py')
    backgrounds = tmp_path / 'backgrounds'
    backgrounds.mkdir()
    Image.new('RGB', (518, 395), (90, 50, 30)).save(backgrounds / 'bg.png')
    output = tmp_path / 'run'
    args = generator._parser().parse_args([
        '--treatment', 'grid', '--count', '1', '--background-dir', str(backgrounds),
        '--skip-color-cal', '--device', 'cpu', '--no-variation', '--save-prior-masks',
        '--posterior-pole-radius', '120', '--output-dir', str(output),
    ])
    generator.run(args)
    record = json.loads((output / 'stage1_manifest.jsonl').read_text().strip())
    assert record['treatment'] == 'grid'
    assert record['generated_spots'] > 0
    assert json.loads((output / '_stage1_config.json').read_text())['treatment'] == 'grid'
    assert Path(record['output']).is_file()
    label = Image.open(record['mask'])
    assert label.size == Image.open(record['output']).size
    assert set(np.unique(np.asarray(label))) <= {0, 255}
    assert np.asarray(label).any()
    metadata = json.loads(Path(record['metadata']).read_text())
    assert metadata['center'] == [300., 200.]
    assert metadata['prior_masks']
    # A fully foreground label shares exactly the same circular crop as the image.
    plain = Image.new('RGB', (100, 100), 'white')
    final, mask = generator.export_fov_pair(plain, 78, Image.new('L', plain.size, 255))
    assert np.array_equal(np.asarray(final)[:, :, 0], np.asarray(mask))
    args.resume = True
    args.treatment = 'prp'
    with pytest.raises(ValueError, match='different treatment'):
        generator.run(args)
