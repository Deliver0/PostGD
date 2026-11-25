"""Image-to-feature-to-head checks without patient data or pretrained weights."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'downstream'))
import extract_features as exporter
import adapt_head as adapter


class TinyStudent(torch.nn.Module):
    def forward_features(self, images):
        return {'x_norm_clstoken': images.mean(dim=(2, 3))}


@pytest.mark.parametrize('candidate', adapter.DEFAULT_CANDIDATES)
def test_saved_head_reproduces_predictions(candidate):
    rng = np.random.default_rng(10)
    features = rng.normal(size=(10, 6)).astype(np.float32)
    labels = np.array([0, 1] * 5)
    query = rng.normal(size=(3, 6)).astype(np.float32)
    fitted = adapter.fit_head(candidate, features, labels)
    head = json.loads(json.dumps(fitted))
    assert np.allclose(adapter.predict_head(head, query), adapter.predict_head(fitted, query))


def test_image_export_and_head_cli_roundtrip(tmp_path, monkeypatch):
    samples = []
    for i in range(6):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (24, 32), (220, 40, 30) if i % 2 else (30, 40, 220)).save(path)
        samples.append({'image_id': str(i), 'image': path.name, 'label': i % 2})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'samples': samples, 'support': ['0', '1', '2', '3'],
                                    'query': ['4', '5']}))
    weights = tmp_path / 'student.pth'
    weights.write_bytes(b'test-only initialization identifier')
    monkeypatch.setattr(exporter, 'load_backbone', lambda *args: TinyStudent())
    npz = tmp_path / 'features.npz'
    assert exporter.main(['--manifest', str(manifest), '--student-weights', str(weights),
                          '--output', str(npz), '--device', 'cpu', '--image-size', '32']) == 0
    arrays = np.load(npz, allow_pickle=False)
    assert arrays['support_features'].shape == (4, 3)
    assert arrays['query_features'].shape == (2, 3)
    result_path = tmp_path / 'result.json'
    adapter.main(['--support-npz', str(npz), '--output', str(result_path), '--cv-repeats', '1'])
    result = json.loads(result_path.read_text())
    head = json.loads(Path(result['head_checkpoint']).read_text())
    probabilities = adapter.predict_head(head, arrays['query_features'])
    assert adapter.score(arrays['query_labels'], probabilities) == result['query_metrics']
    assert result['feature_provenance']['support_ids'] == ['0', '1', '2', '3']
    # Altering query labels cannot change support-only head selection.
    selected, _ = adapter.select_candidate(arrays['support_features'], arrays['support_labels'],
                                           adapter.DEFAULT_CANDIDATES, 1)
    assert selected == result['selected_candidate']
