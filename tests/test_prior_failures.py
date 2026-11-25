"""Anatomical priors must fail visibly when localization is unavailable."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest


@pytest.fixture(params=['full_method', 'render_only'])
def prior_module(request):
    path = Path(__file__).resolve().parents[1] / 'stage1' / request.param / 'src' / 'prior_guidance.py'
    name = f'prior_test_{request.param}'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_localization_stops_anatomical_generation(prior_module):
    image = Image.new('RGB', (128, 128))
    with patch.object(prior_module, '_load_vessels', return_value=(np.zeros((128, 128), dtype=bool), 'ok')), \
         patch.object(prior_module, '_load_fovea_od', return_value=(None, 'checkpoint unavailable')):
        with pytest.raises(RuntimeError, match='fovea/OD localization is unavailable.*checkpoint unavailable'):
            prior_module.build_prior(image, prior_module.PriorConfig(device='cpu'), spot_radius=2)


def test_no_prior_does_not_load_models(prior_module):
    with patch.object(prior_module, '_load_vessels', side_effect=AssertionError('must not load')), \
         patch.object(prior_module, '_load_fovea_od', side_effect=AssertionError('must not load')):
        prior = prior_module.build_prior(Image.new('RGB', (128, 128)),
                                         prior_module.PriorConfig(enabled=False), spot_radius=2)
    assert not prior['enabled']
    assert prior['avoidance_mask'] is None


def test_posterior_region_uses_fovea_and_prp_rings_use_optic_disc(prior_module):
    with patch.object(prior_module, '_load_vessels', return_value=(np.zeros((128, 128), dtype=bool), 'ok')), \
         patch.object(prior_module, '_load_fovea_od', return_value=((90., 64., 30., 64.), 'ok')):
        prior = prior_module.build_prior(Image.new('RGB', (128, 128)),
                                         prior_module.PriorConfig(device='cpu', posterior_pole_radius=20),
                                         spot_radius=2)
    assert prior['center'] == (30., 64.)
    assert prior['posterior_pole_mask'][64, 90]
    assert not prior['posterior_pole_mask'][64, 30]
