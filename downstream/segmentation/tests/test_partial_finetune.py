"""CPU integration checks for selective backbone adaptation and fold isolation."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import partial_finetune as partial
import few_shot_segmentation as runner


class TinyBackbone(torch.nn.Module):
    embed_dim = 4

    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Conv2d(3, 4, 16, stride=16)
        self.blocks = torch.nn.ModuleList([torch.nn.Conv2d(4, 4, 1) for _ in range(12)])
        self.norm = torch.nn.LayerNorm(4)

    def forward_intermediates(self, x, indices, **kwargs):
        x = self.patch_embed(x)
        maps = []
        for index, block in enumerate(self.blocks):
            x = x + 0.1 * torch.tanh(block(x))
            if index in indices:
                maps.append(x)
        return maps


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('task,blocks', [('optic_disc', 1), ('optic_cup', 2)])
def test_only_selected_blocks_and_decoder_update(task, blocks):
    backbone = TinyBackbone()
    initial = copy.deepcopy(backbone.state_dict())
    original = {'a': (torch.randn(3, 32, 32), torch.zeros(32, 32, dtype=torch.long))}
    original['a'][1][8:20, 8:20] = 1
    model = partial.fit_partial(
        backbone, partial.task_finetune_blocks(task), ['a'], original, {'a': []},
        torch.device('cpu'), 1, 3e-4, 1e-5, 7,
    )
    assert all(torch.equal(initial[k], value) for k, value in backbone.state_dict().items())
    changed = {name for name, value in model.backbone.state_dict().items()
               if not torch.equal(initial[name], value)}
    expected = {name for name, _ in model.backbone.named_parameters()
                if any(name.startswith(f'blocks.{i}.') for i in range(12 - blocks, 12))}
    assert changed == expected
    for name, param in model.backbone.named_parameters():
        assert param.requires_grad == (name in expected)
        assert (param.grad is not None) == (name in expected)
    assert all(p.grad is not None for p in model.decoder.parameters())
    groups = model.parameter_groups(3e-4, 1e-5)
    assert [group['lr'] for group in groups] == [3e-4, 1e-5]
    assert not model.backbone.patch_embed.training
    assert model.backbone.blocks[-1].training


def test_aliases_and_frozen_override():
    for task in ['OD', 'optic-disc', 'optic disc']:
        assert partial.task_finetune_blocks(task) == 1
    assert partial.task_finetune_blocks('OC') == 2
    assert partial.task_finetune_blocks('vessel') == 0
    assert partial.task_finetune_blocks('laser_scars') == 0
    args = runner.parse_args(['--manifest', 'x.json', '--output', 'out', '--finetune-blocks', '0'])
    assert args.finetune_blocks == 0


def test_cli_resets_each_fold_delays_test_labels_and_saves_task_weights(tmp_path, monkeypatch):
    torch.manual_seed(3)
    backbone = TinyBackbone()
    initial = copy.deepcopy(backbone.state_dict())
    records = []
    for index in range(3):
        image = tmp_path / f'{index}.png'
        mask = tmp_path / f'{index}_mask.png'
        Image.fromarray(np.random.default_rng(index).integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(image)
        target = np.zeros((32, 32), dtype=np.uint8)
        target[8:20, 8:20] = 255
        Image.fromarray(target).save(mask)
        records.append({'image_id': str(index), 'image': str(image), 'masks': {'optic_disc': str(mask)}})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'samples': records,
                                   'tasks': {'optic_disc': {'support': ['0', '1'], 'test': ['2']}}}))
    monkeypatch.setattr(runner, 'load_backbone', lambda *args: backbone)
    fit_calls = []
    original_fit = partial.fit_partial

    def checked_fit(base, blocks, ids, *args, **kwargs):
        assert all(torch.equal(initial[k], value) for k, value in base.state_dict().items())
        assert '2' not in ids
        fit_calls.append(list(ids))
        return original_fit(base, blocks, ids, *args, **kwargs)

    original_load = partial.load_mask

    def checked_mask(path):
        if Path(path).name == '2_mask.png':
            assert fit_calls == [['1'], ['0'], ['0', '1']]
        return original_load(path)

    monkeypatch.setattr(partial, 'fit_partial', checked_fit)
    monkeypatch.setattr(partial, 'load_mask', checked_mask)
    output = tmp_path / 'output'
    assert runner.main(['--manifest', str(manifest), '--output', str(output),
                        '--image-size', '32', '--device', 'cpu', '--epoch-candidates', '1',
                        '--threshold-candidates', '0.5', '--aug-repeats', '0']) == 0
    report = json.loads((output / 'summary.json').read_text())
    assert report['tasks']['optic_disc']['finetune_blocks'] == 1
    state = torch.load(output / 'optic_disc' / 'task_model.pth', map_location='cpu', weights_only=True)
    restored = partial.PartialSegmentationModel(backbone, state['finetune_blocks'])
    restored.backbone.load_state_dict(state['backbone'], strict=True)
    restored.decoder.load_state_dict(state['decoder'], strict=True)
    probs = partial.probability(restored, partial.load_rgb(records[2]['image']),
                                torch.device('cpu'), 32, 'stretch')
    _, target = partial.preprocess(partial.load_rgb(records[2]['image']),
                                   original_load(records[2]['masks']['optic_disc']), size=32)
    rows = [json.loads(line) for line in (output / 'optic_disc' / 'test_metrics.jsonl').read_text().splitlines()]
    assert rows[0]['dice'] == partial.binary_metrics(probs.ge(state['selected_threshold']), target)['dice']


def test_rejects_overlapping_split_before_reading_images(tmp_path):
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='disjoint'):
        partial.run_partial_task(TinyBackbone(), 1, 'od', {}, ['x'], ['x'],
                                 SimpleNamespace(), torch.device('cpu'))
