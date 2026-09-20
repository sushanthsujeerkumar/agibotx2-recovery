"""Full-episode reference-guided policy: supervised motion prior plus PPO feedback.

The prior is explicit and frozen. PPO learns state-dependent corrections over the
entire recovery. Do not describe the frozen reference as behavior discovered by RL.
"""
import numpy as np
import torch

from .common import ROOT

REFERENCE = ROOT/'artifacts/submission/final_recovery/reference/trajectory.json'
PREPARED = ROOT/'artifacts/submission/final_recovery/prepared'


class ReferenceController:
    """Own validated hybrid controller, used only to collect training examples."""
    def __init__(self, runtime, record):
        self.runtime = runtime
        self.record = record
        self.prior = torch.jit.load(str(REFERENCE.parent/'standing_actor.pt')).eval()
        self.initial = runtime.data.qpos[runtime.info.qadr].copy()
        self.bridge_initial = None

    def __call__(self, obs):
        info = self.runtime.info
        t = float(self.runtime.data.time)
        prefix = self.record['prefix']
        offset = prefix['times'][-1]
        if t < offset-1e-8:
            times = prefix['times']
            points = [self.initial, np.asarray(prefix['points'][0]), np.asarray(prefix['points'][0]),
                      np.asarray(prefix['points'][1]), np.asarray(prefix['points'][1])]
        elif t < offset+self.record['times'][-1]-1e-8:
            if self.bridge_initial is None:
                self.bridge_initial = self.runtime.data.qpos[info.qadr].copy()
            t -= offset
            times = self.record['times']
            points = [self.bridge_initial]+[np.asarray(q) for q in self.record['points']]+[np.asarray(self.record['goal'])]
        else:
            phase = min(self.record['prior_phase']+(t-offset-self.record['times'][-1])*self.record['prior_rate'], 16.)/16.
            return self.prior(torch.cat([obs, torch.full((len(obs), 1), phase, dtype=obs.dtype)], dim=1))
        k = min(max(int(np.searchsorted(times, t, side='right'))-1, 0), len(times)-2)
        u = np.clip((t-times[k])/(times[k+1]-times[k]), 0., 1.)
        u = u*u*(3.-2.*u)
        q = points[k]*(1-u)+points[k+1]*u
        delta = q-info.nominal
        action = (delta/np.where(delta >= 0., info.action_positive, info.action_negative)).clip(-1., 1.)
        return torch.tensor(action, dtype=torch.float32)[None]


class MotionPrior(torch.nn.Module):
    """Phase-conditioned linear spline, fitted by least squares to own examples.

    Accepts the 106 physical observations plus elapsed episode time / 15.
    This supervised prior is not claimed as an RL result.
    """
    def __init__(self, actions):
        super().__init__()
        self.register_buffer('actions', actions.clone().float())

    def forward(self, obs):
        if obs.dim() != 2 or obs.shape[1] != 107:
            raise RuntimeError('Full recovery requires 106 physical observations plus episode phase')
        position = (obs[:, -1]*750.).clamp(0., float(self.actions.shape[0]-1))
        left = position.floor().long()
        right = (left+1).clamp(max=self.actions.shape[0]-1)
        fraction = (position-left.float()).unsqueeze(-1)
        return (self.actions[left]*(1.-fraction)+self.actions[right]*fraction).clamp(-1., 1.)


class ResidualPolicy(torch.nn.Module):
    def __init__(self, prior, feedback, bound=.01):
        super().__init__()
        self.prior = prior
        self.feedback = feedback
        self.bound = bound

    def forward(self, obs):
        return (self.prior(obs)+self.bound*torch.tanh(self.feedback(obs))).clamp(-1., 1.)


def evaluate_actor(actor, seed):
    from .runtime import RecoveryRuntime
    runtime = RecoveryRuntime('full_recovery', actor, physics_profile='guarded_v2')
    runtime.reset(seed)
    try:
        for step in range(750):
            state = runtime.step()
            if state['invalid'] or not runtime.limits.ok:
                break
        return dict(seed=seed, passed=bool(step == 749 and not state['invalid'] and runtime.limits.ok and runtime.clean_hold_time >= 2.-1e-8),
                    hold=runtime.clean_hold_time, sim_time=float(runtime.data.time), limits=runtime.limits.report(),
                    final_checks=state['checks'], final_stance=state['stance'])
    finally:
        runtime.close()
