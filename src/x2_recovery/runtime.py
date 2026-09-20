"""Single-episode CPU MuJoCo runtime used by ROS, evaluation and the viewer."""
from __future__ import annotations

import copy
import numpy as np
import mujoco
from .common import (ModelInfo, CONTROL_DT, HOLD_SECONDS, ground_forces,
                     sensor_forces, observation_numpy, success_conditions, reset_cpu)


class RecoveryRuntime:
    control_dt = CONTROL_DT

    def __init__(self, controller="scripted", checkpoint=None, render=False, seed=0, assess_stance=False,
                 physics_profile="legacy", reset_mode="supine"):
        if controller not in {"scripted", "policy", "full_recovery"}:
            raise ValueError("controller must be scripted, policy or full_recovery")
        if controller == 'full_recovery' and physics_profile != 'guarded_v2':
            raise ValueError('full_recovery requires guarded_v2 physics')
        self.info = ModelInfo(physics_profile=physics_profile)
        self.model = self.info.model
        # Same integration/solver settings used by the training backend.
        self.model.opt.iterations = 50 if physics_profile == "guarded_v2" else 30
        self.model.opt.ls_iterations = 10
        self.model.opt.tolerance = 1e-6
        self.data = mujoco.MjData(self.model)
        self.controller = controller
        self.assess_stance = assess_stance or controller == "full_recovery"
        self.display_label = "SCRIPTED BASELINE" if controller == "scripted" else "TRAINED POLICY"
        if controller == "full_recovery":
            self.display_label = "MOTION PRIOR + PPO FEEDBACK"
        self.policy = None
        self.viewer = None
        self._viewer_threads = []
        if controller in {"policy", "full_recovery"}:
            if not checkpoint:
                raise ValueError("policy controller requires an exported TorchScript checkpoint")
            import torch
            torch.set_num_threads(2)
            self.policy = torch.jit.load(str(checkpoint), map_location="cpu").eval()
        self.reset(seed, reset_mode=reset_mode)
        if render:
            import threading
            from mujoco import viewer as mujoco_viewer
            previous_threads = set(threading.enumerate())
            # The passive viewer calls mj_forward and accepts GUI edits. Keep
            # both away from the controller's solver state and touch sensors.
            self._view_model = copy.copy(self.model)
            self._view_data = mujoco.MjData(self._view_model)
            mujoco.mj_copyData(self._view_data, self._view_model, self.data)
            self.viewer = mujoco_viewer.launch_passive(self._view_model, self._view_data)
            # MuJoCo 3.11's close() requests exit without joining its daemon GUI
            # thread. Keep the render thread created by this launch alive through
            # teardown, before interpreter-level GLFW cleanup can run.
            self._viewer_threads = [t for t in threading.enumerate()
                                    if t not in previous_threads and
                                    t.name.endswith('(_launch_internal)')]
            self.viewer.cam.distance = 2.5
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -20
            self.viewer.cam.lookat[:] = [0, 0, .45]
            self.viewer.opt.geomgroup[3] = 0

    def reset(self, seed=0, reset_mode="supine"):
        self.reset_mode = reset_mode
        if reset_mode == "supine":
            reset_cpu(self.info, self.data, seed)
        elif reset_mode in {"balance", "crouch", "deep_crouch"} and self.info.physics_profile == "guarded_v2":
            from .physics import reset_balance, reset_crouch, reset_deep_crouch
            resetter = {"balance": reset_balance, "crouch": reset_crouch, "deep_crouch": reset_deep_crouch}[reset_mode]
            resetter(self.info, self.data, seed)
        else:
            raise ValueError("Only guarded_v2 supports training-only balance/crouch resets")
        self.previous_action = np.zeros(self.model.nu)
        self.hold_time = 0.
        self.clean_hold_time = 0.
        self.max_clean_hold_time = 0.
        self.stance = {}
        self.max_height = float(self.data.qpos[2])
        self.last_conditions = {}
        from .physics import TrajectoryLimits
        self.limits = TrajectoryLimits(self.info)
        # Include the attained initial state, but not the untimed settling path.
        self.limits.observe(self.data)
        return self.snapshot(False, False)

    def _scripted_target(self):
        """Bounded asymmetric roll-and-crouch attempt, not a proven recovery.

        URDF signs: hip flexion and elbow flexion are negative; knee flexion
        is positive. Axial waist yaw rotates around the spine during a side
        roll. All movement still comes from the shared torque-limited PD loop.
        """
        if float(self.data.time) < self.control_dt or not hasattr(self, "_scripted_start"):
            # Start at the physically settled joint state to avoid a reset jump.
            self._scripted_start = self.data.qpos[self.info.qadr].copy()

        def pose(values):
            q = self.info.nominal.copy()
            for name, value in values.items():
                q[self.info.names.index(name)] = value
            return np.clip(q, self.info.lower, self.info.upper)

        side_roll = pose({
            "waist_yaw_joint": -2.0, "waist_pitch_joint": .2,
            "left_hip_pitch_joint": -1.2, "right_hip_pitch_joint": -.6,
            "left_knee_joint": 1.8, "right_knee_joint": 1.3,
            "left_hip_yaw_joint": -.8,
            "left_hip_roll_joint": -.15, "right_hip_roll_joint": -.25,
            # Left forearm reaches across the chest; right arm seeks support.
            "left_shoulder_pitch_joint": -.6, "right_shoulder_pitch_joint": 1.8,
            "left_shoulder_yaw_joint": -1.5,
            "left_shoulder_roll_joint": 0., "right_shoulder_roll_joint": -.2,
            "left_elbow_joint": -1.8, "right_elbow_joint": -.3,
        })
        crouch = pose({
            "left_hip_pitch_joint": -1.3, "right_hip_pitch_joint": -1.3,
            "left_knee_joint": 1.9, "right_knee_joint": 1.9,
            "left_ankle_pitch_joint": -.7, "right_ankle_pitch_joint": -.7,
            "left_shoulder_pitch_joint": -.8, "right_shoulder_pitch_joint": -.8,
            "left_elbow_joint": -.8, "right_elbow_joint": -.8,
        })
        # Brief holds allow contacts to settle before unwinding and extending.
        times = [0., 1., 2., 4., 6., 8., 15.]
        poses = [self._scripted_start, side_roll, side_roll, crouch, crouch,
                 self.info.nominal, self.info.nominal]
        t = float(self.data.time)
        k = min(max(int(np.searchsorted(times, t, side="right")) - 1, 0), len(times) - 2)
        u = np.clip((t - times[k]) / (times[k + 1] - times[k]), 0, 1)
        u = u * u * (3 - 2 * u)
        return np.clip((1 - u) * poses[k] + u * poses[k + 1], self.info.lower, self.info.upper)

    def step(self):
        if self.policy is not None:
            import torch
            obs = observation_numpy(self.info, self.data, self.previous_action, sensor_forces(self.info, self.data))
            if self.controller == "full_recovery":
                obs = np.concatenate([obs, np.array([min(float(self.data.time)/15., 1.)], dtype=np.float32)])
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
            self.limits.observe(self.data)
        self.previous_action = action.copy()
        invalid = not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        invalid = invalid or np.max(np.abs(self.data.qvel)) > 150 or self.data.qpos[2] < -.05
        forces = ground_forces(self.info, self.data)
        self.last_conditions = success_conditions(self.info, self.data, forces)
        self.hold_time = self.hold_time + CONTROL_DT if all(self.last_conditions.values()) else 0.
        self.max_height = max(self.max_height, float(self.data.qpos[2]))
        success = self.hold_time >= HOLD_SECONDS - 1e-8
        if self.assess_stance:
            from .stance import stance_metrics
            self.stance = stance_metrics(self.info, self.data)
            clean = all(self.last_conditions.values()) and self.stance['posture_ok']
            self.clean_hold_time = self.clean_hold_time + CONTROL_DT if clean else 0.
            self.max_clean_hold_time = max(self.max_clean_hold_time, self.clean_hold_time)
        if self.viewer is not None and self.viewer.is_running() and round(self.data.time/CONTROL_DT) % 2 == 0:
            with self.viewer.lock():
                mujoco.mj_copyData(self._view_data, self._view_model, self.data)
            extra_title = '\nClean stance' if self.assess_stance else ''
            extra_value = f'\n{self.clean_hold_time:.2f} / {HOLD_SECONDS:.1f} s' if self.assess_stance else ''
            display_success = self.clean_hold_time >= HOLD_SECONDS - 1e-8 if self.assess_stance else success
            outcome = 'LIMIT FAILURE' if not self.limits.ok else 'SUCCEEDED' if display_success else 'RUNNING'
            self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,
                                   mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   f"X2 {'supine recovery' if self.reset_mode == 'supine' else self.reset_mode + ' curriculum'} evaluation\nController\nEpisode time\nPelvis height\nStable standing\nOutcome" + extra_title,
                                   f"\n{self.display_label}\n{self.data.time:.1f} s\n{self.data.qpos[2]:.3f} m\n{self.hold_time:.2f} / {HOLD_SECONDS:.1f} s\n{outcome}" + extra_value))
            self.viewer.sync(state_only=True)
        return self.snapshot(success, bool(invalid))

    def snapshot(self, success, invalid):
        result = {"joint_names": list(self.info.names),
                "joint_positions": self.data.qpos[self.info.qadr].tolist(),
                "sim_time": float(self.data.time), "success": bool(success), "invalid": bool(invalid),
                "pelvis_height": float(self.data.qpos[2]), "max_pelvis_height": self.max_height,
                "standing_hold_seconds": self.hold_time, "checks": dict(self.last_conditions)}
        result.update(physics_profile=self.info.physics_profile, reset_mode=self.reset_mode,
                      trajectory_limits=self.limits.report(),
                      validated_success=bool(success and self.limits.ok))
        if self.assess_stance:
            result.update(stance=self.stance, clean_stance_hold_seconds=self.clean_hold_time,
                          max_clean_stance_hold_seconds=self.max_clean_hold_time,
                          clean_stance_success=self.clean_hold_time >= HOLD_SECONDS - 1e-8)
        return result

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            for thread in self._viewer_threads:
                thread.join(timeout=5.)
                if thread.is_alive():
                    raise RuntimeError('MuJoCo viewer did not finish closing within 5 seconds')
            self._viewer_threads = []
            self.viewer = None
