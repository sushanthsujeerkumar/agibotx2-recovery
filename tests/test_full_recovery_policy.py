"""Clock contract and self-contained residual policy export parity."""
import copy
import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from x2_recovery.full_recovery import MotionPrior, ResidualPolicy
from x2_recovery.train import ppo_config, export_actor


def test_prior_phase_contract_and_end_clamping():
    actions = torch.linspace(-.9, .9, 750)[:, None].expand(750, 31).clone()
    policy = torch.jit.script(MotionPrior(actions))
    obs = torch.zeros(3, 107)
    obs[:, -1] = torch.tensor([0., 749./750., 1.])
    torch.testing.assert_close(policy(obs), actions[[0, 749, 749]])
    with pytest.raises((RuntimeError, torch.jit.Error)):
        policy(torch.zeros(3, 106))


def test_checkpoint_export_contains_prior_and_feedback(tmp_path):
    torch.manual_seed(89)
    cfg = ppo_config(variant='stance')
    actor_cfg = copy.deepcopy(cfg['actor'])
    actor_cfg.pop('class_name')
    sample = TensorDict({'actor': torch.zeros(1, 107), 'critic': torch.zeros(1, 107)}, batch_size=[1])
    actor = MLPModel(sample, cfg['obs_groups'], 'actor', 31, **actor_cfg).eval()
    actions = torch.rand(750, 31)*1.8-.9
    checkpoint = dict(train_cfg=cfg, actor_state_dict=actor.state_dict(),
                      observation_shapes={'actor': [107], 'critic': [107]}, num_actions=31,
                      motion_prior_actions=actions, environment_cfg={'feedback_bound': .01})
    path = tmp_path/'checkpoint.pt'
    torch.save(checkpoint, path)
    exported = torch.jit.load(str(export_actor(path, tmp_path/'actor.pt')))
    expected = ResidualPolicy(MotionPrior(actions), actor.as_jit(), .01).eval()
    obs = torch.randn(20, 107)
    obs[:, -1] = torch.linspace(0., 1., 20)
    with torch.inference_mode():
        torch.testing.assert_close(exported(obs), expected(obs), rtol=0, atol=0)
        assert (exported(obs)-MotionPrior(actions)(obs)).abs().max() <= .010001
