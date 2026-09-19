"""Single-episode CPU MuJoCo runtime used by ROS, evaluation and the viewer."""
from __future__ import annotations

import numpy as np
import mujoco
from .common import (ModelInfo, CONTROL_DT, HOLD_SECONDS, ground_forces,
                     sensor_forces, observation_numpy, success_conditions, reset_cpu)


class RecoveryRuntime:
    control_dt = CONTROL_DT

    def __init__(self, controller="scripted", checkpoint=None, render=False, seed=0):
        if controller not in {"scripted", "policy"}:
            raise ValueError("controller must be scripted or policy")
        self.info = ModelInfo()
        self.model = self.info.model
        # Same integration/solver settings used by the training backend.
        self.model.opt.iterations = 30
        self.model.opt.ls_iterations = 10
        self.model.opt.tolerance = 1e-6
        self.data = mujoco.MjData(self.model)
        self.controller = controller
        self.policy = None
        self.viewer = None
        if controller == "policy":
            if not checkpoint:
                raise ValueError("policy controller requires an exported TorchScript checkpoint")
            import torch
            torch.set_num_threads(2)
            self.policy = torch.jit.load(str(checkpoint), map_location="cpu").eval()
        self.reset(seed)
        if render:
            from mujoco import viewer as mujoco_viewer
            self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 2.5
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -20
            self.viewer.cam.lookat[:] = [0, 0, .45]
            self.viewer.opt.geomgroup[3] = 0

    def reset(self, seed=0):
        reset_cpu(self.info, self.data, seed)
        self.previous_action = np.zeros(self.model.nu)
        self.hold_time = 0.
        self.max_height = float(self.data.qpos[2])
        self.last_conditions = {}
        return self.snapshot(False, False)

    def _scripted_target(self):
        """Untrained bounded keyframe attempt. No direct base motion or assistance."""
        target = self.info.nominal.copy()
        def pose(values):
            q = target.copy()
            for name, value in values.items():
                if name in self.info.names:
                    q[self.info.names.index(name)] = value
            return np.clip(q, self.info.lower, self.info.upper)
        tuck = pose({"left_hip_pitch_joint": -1.5, "right_hip_pitch_joint": -1.5,
                     "left_knee_joint": 2.1, "right_knee_joint": 2.1,
                     "left_shoulder_pitch_joint": -.8, "right_shoulder_pitch_joint": -.8,
                     "left_elbow_joint": -1., "right_elbow_joint": -1.})
        roll = tuck.copy()
        for name, value in {"waist_roll_joint": .25, "left_hip_roll_joint": .3,
                            "right_hip_roll_joint": .2, "left_shoulder_roll_joint": .7,
                            "right_shoulder_roll_joint": -.3}.items():
            if name in self.info.names: roll[self.info.names.index(name)] = value
        crouch = pose({"left_hip_pitch_joint": -1.1, "right_hip_pitch_joint": -1.1,
                       "left_knee_joint": 2., "right_knee_joint": 2.,
                       "left_ankle_pitch_joint": -.6, "right_ankle_pitch_joint": -.6,
                       "left_shoulder_pitch_joint": -.9, "right_shoulder_pitch_joint": -.9})
        times = [0., 1.5, 3., 5., 8., 15.]
        poses = [self.info.supine[self.info.qadr], tuck, roll, crouch, target, target]
        t = float(self.data.time)
        k = min(int(np.searchsorted(times, t, side="right"))-1, len(times)-2)
        u = np.clip((t-times[k])/(times[k+1]-times[k]), 0, 1)
        u = u*u*(3-2*u)
        return np.clip((1-u)*poses[k] + u*poses[k+1], self.info.lower, self.info.upper)

    def step(self):
        if self.policy is not None:
            import torch
            obs = observation_numpy(self.info, self.data, self.previous_action, sensor_forces(self.info, self.data))
            with torch.inference_mode():
                action = self.policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
            if action.shape != (self.model.nu,) or not np.isfinite(action).all():
                raise RuntimeError("Policy returned invalid actions")
            action = np.clip(action, -1, 1)
            target = self.info.targets(action)
        else:
            target = self._scripted_target()
            delta = target-self.info.nominal
            scale = np.where(delta>=0, self.info.action_positive, self.info.action_negative)
            action = np.clip(delta/np.maximum(scale, 1e-6), -1, 1)
        for _ in range(self.info.substeps):
            self.data.ctrl[:] = self.info.torque(self.data.qpos[self.info.qadr], self.data.qvel[self.info.vadr], target)
            mujoco.mj_step(self.model, self.data)
        self.previous_action = action.copy()
        invalid = not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        invalid = invalid or np.max(np.abs(self.data.qvel)) > 150 or self.data.qpos[2] < -.05
        forces = ground_forces(self.info, self.data)
        self.last_conditions = success_conditions(self.info, self.data, forces)
        self.hold_time = self.hold_time + CONTROL_DT if all(self.last_conditions.values()) else 0.
        self.max_height = max(self.max_height, float(self.data.qpos[2]))
        success = self.hold_time >= HOLD_SECONDS - 1e-8
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
        return self.snapshot(success, bool(invalid))

    def snapshot(self, success, invalid):
        return {"joint_names": list(self.info.names),
                "joint_positions": self.data.qpos[self.info.qadr].tolist(),
                "sim_time": float(self.data.time), "success": bool(success), "invalid": bool(invalid),
                "pelvis_height": float(self.data.qpos[2]), "max_pelvis_height": self.max_height,
                "standing_hold_seconds": self.hold_time, "checks": dict(self.last_conditions)}

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
