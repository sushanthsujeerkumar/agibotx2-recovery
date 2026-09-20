"""Batched X2 recovery using mjlab's MuJoCo Warp simulation backend."""
from __future__ import annotations

import numpy as np
import mujoco
import torch
import warp as wp
from tensordict import TensorDict
from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
from .common import (ModelInfo, CONTROL_DT, EPISODE_SECONDS, HOLD_SECONDS,
                     SUCCESS_TILT, SUCCESS_LINEAR_SPEED, SUCCESS_ANGULAR_SPEED,
                     CONTACT_FORCE, reset_cpu)
from .stance import MIN_WIDTH, MAX_WIDTH, MAX_HEADING, MAX_HIP_YAW, MAX_SOLE_TILT


class X2RecoveryEnv:
    def __init__(self, num_envs=256, device="cuda:0", seed=0, reward_version=1,
                 physics_profile="legacy", reset_mode="supine"):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.info = ModelInfo(physics_profile=physics_profile)
        if reset_mode not in {"supine", "balance", "crouch"}:
            raise ValueError("reset_mode must be supine, balance or crouch")
        if reset_mode != "supine" and physics_profile != "guarded_v2":
            raise ValueError("Standing/crouch curriculum requires guarded_v2")
        self.reset_mode = reset_mode
        self.strict_limits = physics_profile == "guarded_v2"
        self.num_actions = self.info.model.nu
        self.num_obs = self.num_actions * 3 + 13
        self.num_privileged_obs = self.num_obs
        self.max_episode_length = round(EPISODE_SECONDS / CONTROL_DT)
        if reward_version not in (1, 2, 3):
            raise ValueError("reward_version must be 1 (baseline), 2 (stability) or 3 (stance)")
        self.reward_version = reward_version
        self.cfg = {"physics_dt": self.info.model.opt.timestep, "control_dt": CONTROL_DT,
                    "episode_seconds": EPISODE_SECONDS, "hold_seconds": HOLD_SECONDS,
                    "reset": "settled_supine" if reset_mode == "supine" else f"training_only_{reset_mode}",
                    "reset_mode": reset_mode, "physics_profile": physics_profile,
                    "strict_trajectory_limits": self.strict_limits, "seed": seed, "num_envs": num_envs,
                    "reward_version": reward_version, "is_finite_horizon": False}
        torch.manual_seed(seed)
        np.random.seed(seed)
        wp.init()
        if self.device.type == "cuda":
            # Use Warp's owned non-default stream for both libraries and graph capture.
            self._torch_stream = torch.cuda.ExternalStream(wp.get_stream(str(self.device)).cuda_stream, device=self.device)
            torch.cuda.set_stream(self._torch_stream)
        cfg = SimulationCfg(nconmax=128, njmax=512,
            mujoco=MujocoCfg(timestep=self.info.model.opt.timestep,
                            iterations=50 if self.strict_limits else 30, ls_iterations=10, tolerance=1e-6,
                            integrator="implicitfast", cone="pyramidal", solver="newton"))
        self.sim = Simulation(num_envs=num_envs, cfg=cfg, model=self.info.model, device=str(self.device))
        self.qpos = wp.to_torch(self.sim.wp_data.qpos)
        self.qvel = wp.to_torch(self.sim.wp_data.qvel)
        self.ctrl = wp.to_torch(self.sim.wp_data.ctrl)
        self.xmat = wp.to_torch(self.sim.wp_data.xmat)
        self.geom_xpos = wp.to_torch(self.sim.wp_data.geom_xpos)
        self.geom_xmat = wp.to_torch(self.sim.wp_data.geom_xmat)
        self.sensors = wp.to_torch(self.sim.wp_data.sensordata)
        self.qadr = torch.as_tensor(self.info.qadr, device=self.device, dtype=torch.long)
        self.vadr = torch.as_tensor(self.info.vadr, device=self.device, dtype=torch.long)
        for name in ["lower", "upper", "effort", "velocity", "kp", "kd", "nominal", "action_scale", "action_positive", "action_negative"]:
            setattr(self, name, torch.as_tensor(getattr(self.info, name), device=self.device, dtype=torch.float32))
        m = self.info.model
        meta = self.info.metadata
        self.foot_sensor = torch.as_tensor([m.sensor_adr[i] for i in meta["foot_sensor_indices"]], device=self.device)
        self.other_sensor = torch.as_tensor([m.sensor_adr[i] for i in meta["nonfoot_sensor_indices"]], device=self.device)
        self.pelvis_id = m.body("pelvis").id
        self.foot_geoms = [m.geom(f'{side}_ankle_roll_link_collision_0').id for side in ('left', 'right')]
        self.hip_yaw_qadr = [int(self.info.qadr[self.info.names.index(f'{side}_hip_yaw_joint')]) for side in ('left', 'right')]
        reference = mujoco.MjData(m)
        reference.qpos[:] = self.info.standing
        mujoco.mj_forward(m, reference)
        self.nominal_foot_width = float((reference.geom_xpos[self.foot_geoms[0]] - reference.geom_xpos[self.foot_geoms[1]]) @ reference.xmat[self.pelvis_id].reshape(3, 3)[:, 1])
        if reward_version == 3:
            self.cfg['stance'] = {'min_width_m': MIN_WIDTH, 'max_width_m': MAX_WIDTH,
                                 'max_heading_rad': MAX_HEADING, 'max_hip_yaw_rad': MAX_HIP_YAW,
                                 'max_sole_tilt_rad': MAX_SOLE_TILT,
                                 'nominal_foot_width_m': self.nominal_foot_width,
                                 'termination': 'original recovery plus clean stance held for 2s',
                                 'foot_bracing': 'geometry proxy in training; exact force audit in CPU evaluation'}
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.hold_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.clean_hold_buf = torch.zeros_like(self.hold_buf)
        self.recovered_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.limit_failed = torch.zeros_like(self.recovered_buf)
        self.previous_action = torch.zeros((num_envs, self.num_actions), device=self.device)
        self.episode_reward = torch.zeros(num_envs, device=self.device)
        self.previous_height = torch.zeros(num_envs, device=self.device)
        self.max_height = torch.zeros(num_envs, device=self.device)
        self.reset_q, self.reset_v = self._reset_bank(seed)
        self.reset()

    def _reset_bank(self, seed):
        d = mujoco.MjData(self.info.model)
        qs, vs = [], []
        for i in range(16):
            if self.reset_mode in {"balance", "crouch"}:
                from .physics import reset_balance, reset_crouch
                resetter = reset_balance if self.reset_mode == "balance" else reset_crouch
                resetter(self.info, d, seed+i)
            else:
                reset_cpu(self.info, d, seed+i)
            qs.append(d.qpos.copy())
            vs.append(d.qvel.copy())
        return (torch.as_tensor(np.stack(qs), dtype=torch.float32, device=self.device),
                torch.as_tensor(np.stack(vs), dtype=torch.float32, device=self.device))

    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        self.sim.reset(ids)
        choice = torch.randint(len(self.reset_q), (len(ids),), device=self.device)
        self.qpos[ids] = self.reset_q[choice]
        self.qvel[ids] = self.reset_v[choice]
        self.ctrl[ids] = 0
        self.previous_action[ids] = 0
        self.episode_length_buf[ids] = 0
        self.hold_buf[ids] = 0
        self.clean_hold_buf[ids] = 0
        self.recovered_buf[ids] = False
        self.limit_failed[ids] = False
        self.episode_reward[ids] = 0
        self.previous_height[ids] = self.qpos[ids, 2]
        self.max_height[ids] = self.qpos[ids, 2]
        self.sim.forward()
        return self.get_observations(), {}

    def support_forces(self):
        return torch.cat([self.sensors[:, self.foot_sensor],
                          self.sensors[:, self.other_sensor].sum(dim=-1, keepdim=True)], dim=-1)

    def get_observations(self):
        r = self.xmat[:, self.pelvis_id].reshape(-1, 3, 3)
        gravity = -r[:, 2, :]
        local_vel = torch.bmm(r.transpose(1, 2), self.qvel[:, :3, None]).squeeze(-1)
        obs = torch.cat([self.qpos[:, self.qadr] - self.nominal,
                         self.qvel[:, self.vadr] * 0.1, gravity, local_vel,
                         self.qvel[:, 3:6] * .25, self.qpos[:, 2:3],
                         (self.support_forces() > CONTACT_FORCE).float(), self.previous_action], dim=-1)
        obs = torch.nan_to_num(obs, nan=0., posinf=10., neginf=-10.).clamp(-10, 10)
        return TensorDict({"actor": obs, "critic": obs}, batch_size=[self.num_envs])

    def step(self, actions):
        actions = actions.detach().clamp(-1, 1)
        scale = torch.where(actions >= 0, self.action_positive, self.action_negative)
        target = (self.nominal + actions * scale).clamp(self.lower, self.upper)
        for _ in range(self.info.substeps):
            v = self.qvel[:, self.vadr]
            tau = (self.kp * (target-self.qpos[:, self.qadr]) - self.kd*v).clamp(-self.effort, self.effort)
            tau = torch.where(((v >= self.velocity) & (tau > 0)) | ((v <= -self.velocity) & (tau < 0)), 0., tau)
            if self.strict_limits:
                from .physics import guarded_torque_torch
                tau = guarded_torque_torch(self.qpos[:, self.qadr], v, target,
                                           self.lower, self.upper, self.kp, self.kd,
                                           self.velocity, self.effort)
            self.ctrl.copy_(tau)
            self.sim.step()
            if self.strict_limits:
                from .physics import POSITION_TOLERANCE, RATIO_TOLERANCE
                q = self.qpos[:, self.qadr]
                bad = ((q < self.lower - POSITION_TOLERANCE) |
                       (q > self.upper + POSITION_TOLERANCE) |
                       (self.qvel[:, self.vadr].abs() > self.velocity * (1. + RATIO_TOLERANCE))).any(-1)
                bad |= ~torch.isfinite(q).all(-1) | ~torch.isfinite(self.qvel).all(-1)
                self.limit_failed |= bad
        self.episode_length_buf += 1
        height = self.qpos[:, 2]
        up = self.xmat[:, self.info.torso_id].reshape(-1, 3, 3)[:, 2, 2]
        forces = self.support_forces()
        feet = (forces[:, :2] > CONTACT_FORCE).all(dim=-1)
        other_clear = forces[:, 2] <= CONTACT_FORCE
        linear = self.qvel[:, :3].norm(dim=-1)
        angular = self.qvel[:, 3:6].norm(dim=-1)
        standing = ((height >= self.info.height_threshold) & (up >= np.cos(SUCCESS_TILT)) &
                    feet & other_clear & (linear < SUCCESS_LINEAR_SPEED) & (angular < SUCCESS_ANGULAR_SPEED))
        self.hold_buf = torch.where(standing, self.hold_buf+1, 0)
        original_success = self.hold_buf >= round(HOLD_SECONDS/CONTROL_DT)
        if self.strict_limits:
            original_success &= ~self.limit_failed
        self.recovered_buf |= original_success
        success = original_success
        if self.reward_version == 3:
            pelvis_rotation = self.xmat[:, self.pelvis_id].reshape(-1, 3, 3)
            foot_delta = self.geom_xpos[:, self.foot_geoms[0]] - self.geom_xpos[:, self.foot_geoms[1]]
            pelvis_forward = torch.nn.functional.normalize(pelvis_rotation[:, :2, 0], dim=-1, eps=1e-8)
            pelvis_left = torch.stack([-pelvis_forward[:, 1], pelvis_forward[:, 0]], dim=-1)
            width = (foot_delta[:, :2] * pelvis_left).sum(-1)
            foot_rotation = self.geom_xmat[:, self.foot_geoms].reshape(-1, 2, 3, 3)
            foot_forward = foot_rotation[:, :, :2, 0]
            foot_forward = torch.nn.functional.normalize(foot_forward, dim=-1, eps=1e-8)
            heading_cos = (foot_forward * pelvis_forward[:, None]).sum(-1).clamp(-1, 1)
            hip_yaw = self.qpos[:, self.hip_yaw_qadr]
            posture_ok = ((width >= MIN_WIDTH) & (width <= MAX_WIDTH) &
                          (heading_cos >= np.cos(MAX_HEADING)).all(-1) &
                          (foot_rotation[:, :, 2, 2] >= np.cos(MAX_SOLE_TILT)).all(-1) &
                          (hip_yaw.abs() <= MAX_HIP_YAW).all(-1))
            self.clean_hold_buf = torch.where(standing & posture_ok, self.clean_hold_buf+1, 0)
            success = self.clean_hold_buf >= round(HOLD_SECONDS/CONTROL_DT)
            if self.strict_limits:
                success &= ~self.limit_failed
        upright = ((up+1.)*.5).clamp(0, 1)
        h = (height/self.info.standing[2]).clamp(0, 1.1)
        progress = (height-self.previous_height).clamp(-.1, .1)/CONTROL_DT
        effort = ((self.ctrl/self.effort)**2).mean(dim=-1)
        action_change = ((actions-self.previous_action)**2).mean(dim=-1)
        speed_excess = ((self.qvel[:,self.vadr].abs()/self.velocity-1).clamp_min(0)**2).mean(dim=-1)
        joint_limit = (((self.lower+.03-self.qpos[:,self.qadr]).clamp_min(0) +
                        (self.qpos[:,self.qadr]-self.upper+.03).clamp_min(0))**2).mean(dim=-1)
        reward = (2.*h*upright + upright + .5*progress + 1.*feet.float()*h*upright +
                  4.*standing.float() - .03*effort - .03*action_change - .1*speed_excess - joint_limit)*CONTROL_DT
        if self.reward_version >= 2:
            # The first run stood up but kept hopping/travelling. Give a smooth
            # gradient toward rest before the strict binary success threshold.
            support_gate = h * upright * feet.float() * other_clear.float()
            stability = torch.exp(-1.5*linear.square() - .3*angular.square())
            posture_error = ((self.qpos[:,self.qadr]-self.nominal)**2).mean(dim=-1)
            reward += (4.*support_gate*stability - .15*h*upright*posture_error)*CONTROL_DT
        if self.reward_version == 3:
            # Only shape posture near upright standing; leave the supine roll free.
            phase = ((h-.55)/.35).clamp(0, 1) * ((up-.5)/.5).clamp(0, 1)
            width_error = (width-self.nominal_foot_width)/.18
            heading_error = (1.-heading_cos).mean(-1)
            sole_error = (1.-foot_rotation[:, :, 2, 2]).mean(-1)
            quality = torch.exp(-width_error.square() - 2.*heading_error - 2.*sole_error)
            narrow_or_crossed = ((MIN_WIDTH-width)/.18).clamp(0, 2)
            reward += phase*(6.*quality - 2.*narrow_or_crossed - heading_error - sole_error - .5*hip_yaw.square().mean(-1))*CONTROL_DT
        reward += 10.*success.float()
        invalid = (~torch.isfinite(self.qpos).all(dim=-1) | ~torch.isfinite(self.qvel).all(dim=-1) |
                   (self.qvel.abs().amax(dim=-1)>150) | (height<-.05))
        if self.strict_limits:
            invalid |= self.limit_failed
        reward = torch.nan_to_num(reward, nan=-1., posinf=-1., neginf=-1.) - invalid.float()
        timeout = self.episode_length_buf >= self.max_episode_length
        done = success | timeout | invalid
        self.episode_reward += reward
        self.max_height = torch.maximum(self.max_height, torch.nan_to_num(height, nan=0., posinf=0., neginf=0.))
        self.previous_height.copy_(height)
        self.previous_action.copy_(actions)
        extras = {"time_outs": timeout & ~success & ~invalid}
        ids = torch.nonzero(done, as_tuple=False).flatten()
        if len(ids):
            extras["log"] = {"recovery/success_rate": success[ids].float(),
                              "recovery/invalid_rate": invalid[ids].float(),
                              "recovery/max_pelvis_height": self.max_height[ids],
                              "recovery/episode_return": self.episode_reward[ids],
                              "recovery/episode_seconds": self.episode_length_buf[ids]*CONTROL_DT,
                              "recovery/terminal_linear_speed": linear[ids],
                              "recovery/terminal_angular_speed": angular[ids]}
            if self.reward_version == 3:
                extras['log'].update({'recovery/original_recovery_rate': self.recovered_buf[ids].float(),
                                      'stance/clean_success_rate': success[ids].float(),
                                      'stance/terminal_signed_width': width[ids],
                                      'stance/terminal_sole_error': sole_error[ids],
                                      'stance/terminal_heading_error': heading_error[ids]})
            if self.strict_limits:
                extras['log']['physics/limit_failure_rate'] = self.limit_failed[ids].float()
            self.reset(ids)
        return self.get_observations(), reward, done.long(), extras

    def close(self):
        pass
