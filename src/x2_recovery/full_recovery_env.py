"""Supine-start PPO environment with an explicit frozen own motion prior."""
import torch
from tensordict import TensorDict

from .env import X2RecoveryEnv
from .full_recovery import MotionPrior


class FullRecoveryEnv(X2RecoveryEnv):
    def __init__(self, prior_actions, feedback_bound=.01, publisher=None, **kwargs):
        self.publisher = None
        super().__init__(physics_profile='guarded_v2', reset_mode='supine', reward_version=3, **kwargs)
        self.motion_prior_actions = prior_actions.detach().to(self.device).clone()
        self.prior = MotionPrior(self.motion_prior_actions).to(self.device).eval()
        self.feedback_bound = feedback_bound
        self.num_obs = self.num_privileged_obs = 107
        self.cfg.update(method='own_frozen_motion_prior_plus_full_episode_PPO_feedback',
                        feedback_bound=feedback_bound, phase_input='elapsed_episode_seconds / 15',
                        external_teacher=False, resume_environment='fresh supine episodes')
        self.publisher = publisher
        if publisher:
            publisher.publish(self, force=True)

    def get_observations(self):
        obs = super().get_observations()['actor']
        phase = (self.episode_length_buf.float()/750.).clamp(0., 1.)[:, None]
        obs = torch.cat([obs, phase], dim=-1)
        return TensorDict({'actor': obs, 'critic': obs}, batch_size=[self.num_envs])

    def step(self, actions):
        with torch.inference_mode():
            actual = (self.prior(self.get_observations()['actor'])+
                      self.feedback_bound*torch.tanh(actions)).clamp(-1., 1.)
        result = super().step(actual)
        if self.publisher:
            self.publisher.publish(self)
        return result

    def close(self):
        if self.publisher:
            self.publisher.close()
            self.publisher = None
        super().close()
