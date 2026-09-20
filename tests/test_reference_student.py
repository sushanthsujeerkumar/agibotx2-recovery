"""Protect the experimental student deployment/action interface."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_reference_student import Student, action_from_target
from x2_recovery.common import ModelInfo


def test_teacher_targets_roundtrip_through_asymmetric_student_action_space():
    info = ModelInfo(physics_profile='guarded_v2')
    rng = np.random.default_rng(481)
    for _ in range(100):
        target = rng.uniform(info.lower + .05, info.upper - .05)
        action = action_from_target(info, target)
        assert np.max(np.abs(action)) <= 1.
        np.testing.assert_allclose(info.targets(action), target, atol=1e-14)
        # Torch actors return float32; quantify the deployment conversion error.
        np.testing.assert_allclose(info.targets(action.astype(np.float32)), target, atol=2e-7)


def test_export_retains_normalization_and_explicit_phase_contract(tmp_path):
    torch.manual_seed(71)
    model = Student(torch.randn(107), torch.rand(107) + .1).eval()
    x = torch.randn(9, 107)
    scripted = torch.jit.script(model)
    path = tmp_path / 'student.pt'
    torch.jit.save(scripted, str(path))
    loaded = torch.jit.load(str(path)).eval()
    with torch.inference_mode():
        assert torch.equal(model(x), loaded(x))
        assert loaded(x).shape == (9, 31)
        assert loaded(x).abs().max() <= 1
    # The old 106-input runtime must not silently accept a phase-based policy.
    with pytest.raises((RuntimeError, torch.jit.Error)):
        loaded(torch.zeros(1, 106))


def test_masked_student_cannot_copy_previous_action():
    torch.manual_seed(72)
    model = Student(torch.zeros(107), torch.ones(107), drop_previous_action=True).eval()
    x = torch.randn(4, 107)
    changed = x.clone()
    changed[:, 75:106] = torch.randn(4, 31) * 10
    with torch.inference_mode():
        assert torch.equal(model(x), model(changed))
        changed[:, 106] += .3
        assert not torch.equal(model(x), model(changed))
