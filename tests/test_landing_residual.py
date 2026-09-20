"""Numerical contracts for the small episodic PPO correction experiment."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_landing_residual import BOUND, STD, Residual, gae, landing_window, save_checkpoint


def test_terminal_gae_has_no_bootstrap_beyond_episode():
    values = np.array([.5, .25, -.1], np.float32)
    advantage, returns = gae([1., 2., 3.], values, gamma=.9, lam=1.)
    np.testing.assert_allclose(returns, [5.23, 4.7, 3.], atol=1e-6)
    np.testing.assert_allclose(advantage, returns - values, atol=1e-6)
    _, fault_returns = gae([0., -5.], [7., 9.], gamma=.9, lam=1.)
    np.testing.assert_allclose(fault_returns, [-4.5, -5.], atol=1e-6)


def test_correction_is_zero_outside_landing_and_bounded_inside():
    for t in [0., 5., 8.99, 9., 11., 15.]:
        assert landing_window(t) == 0.
    assert landing_window(10.) == 1.
    assert 0 < landing_window(9.15) < 1
    assert 0 < landing_window(10.85) < 1
    for t in np.linspace(0., 15., 501):
        assert 0 <= landing_window(t) <= 1
    model = Residual()
    assert torch.equal(model(torch.randn(7, 78)), torch.zeros(7, 2))
    with torch.no_grad():
        model.actor[-1].bias[:] = torch.tensor([100., -100.])
    np.testing.assert_allclose(model(torch.zeros(1, 78)).detach().numpy(), [[BOUND, -BOUND]], atol=1e-8)


def test_worker_and_optimizer_log_probabilities_agree():
    mean = np.array([.12, -.07], np.float32)
    raw = np.array([.41, -.2], np.float32)
    worker = np.sum(-.5 * ((raw - mean) / STD) ** 2 - math.log(STD) - .5 * math.log(2 * math.pi))
    optimizer = torch.distributions.Normal(torch.from_numpy(mean), STD).log_prob(torch.from_numpy(raw)).sum()
    np.testing.assert_allclose(worker, float(optimizer), atol=1e-6)


def test_exported_normalized_offsets_match_checkpoint_with_documented_scale(tmp_path):
    torch.manual_seed(27)
    model = Residual().eval()
    with torch.no_grad():
        model.actor[-1].weight.normal_(0, .01)
    save_checkpoint(model, tmp_path, 2, {'updates': 2})
    exported = torch.jit.load(str(tmp_path / 'update_002_actor.pt')).eval()
    x = torch.randn(7, 78)
    with torch.inference_mode():
        np.testing.assert_allclose((exported(x) * BOUND).numpy(), model(x).numpy(), atol=1e-8)
