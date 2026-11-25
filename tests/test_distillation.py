"""Numerical loss and initialization checks for the public trainer."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

path = Path(__file__).resolve().parents[1] / 'distillation' / 'distill.py'
spec = importlib.util.spec_from_file_location('tested_distill', path)
distill = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = distill
spec.loader.exec_module(distill)


def test_epoch_consistency_uses_raw_features_and_teacher_stays_fixed():
    torch.manual_seed(42)
    student = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 3))
    teacher = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 5))
    projector = torch.nn.Linear(3, 5)
    v1, v2 = torch.randn(2, 3, 2, 2), torch.randn(2, 3, 2, 2)
    with torch.no_grad():
        expected = distill.consistency_loss(student(v1), student(v2)).item()
        projected = distill.consistency_loss(projector(student(v1)), projector(student(v2))).item()
    assert abs(expected - projected) > 1e-3
    teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
    student_before = student[1].weight.detach().clone()
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(projector.parameters()), lr=1e-3)
    cfg = distill.DistillConfig(data_dir='unused', device='cpu', batch_size=2)
    result = distill.train_one_epoch(student, teacher, projector,
                                    DataLoader(TensorDataset(v1, v2), batch_size=2),
                                    optimizer, None, None, cfg, 0)
    assert result['cons'] == pytest.approx(expected, abs=1e-6)
    assert not torch.equal(student_before, student[1].weight)
    assert all(torch.equal(value, teacher_before[k]) for k, value in teacher.state_dict().items())
    assert all(p.grad is None for p in teacher.parameters())


def test_incomplete_checkpoint_is_rejected(tmp_path):
    path = tmp_path / 'incomplete.pth'
    torch.save({'weight': torch.ones(3, 2)}, path)
    with pytest.raises(ValueError, match='checkpoint does not match'):
        distill._load_weights(torch.nn.Linear(2, 3), str(path))
