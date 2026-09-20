"""Versioned simulation assumptions and read-only trajectory compliance checks."""
import numpy as np

PROFILES = ("legacy", "guarded_v2")
POSITION_TOLERANCE = 1e-6
RATIO_TOLERANCE = 1e-6


def configure_model(model, joint_ids, profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown physics profile: {profile}")
    if profile == "guarded_v2":
        # Earlier limit engagement, not a change to the published joint range.
        # These are engineering assumptions, not identified hardware parameters.
        model.opt.timestep = .001
        model.opt.iterations = 50
        model.opt.ls_iterations = 10
        model.opt.tolerance = 1e-6
        model.jnt_solref[joint_ids] = [.002, 1.]
        model.jnt_solimp[joint_ids] = [.999, .999, .001, .5, 2.]
        model.jnt_margin[joint_ids] = .04


def guarded_torque(info, q, v, target):
    """Effort-limited velocity servo with soft approach to position boundaries.

    This does not guarantee speed bounds against contact impulses. The independent
    trajectory monitor must reject excursions. No state or contact is overwritten.
    """
    target = np.clip(target, info.lower + .05, info.upper - .05)
    velocity_target = np.clip((info.kp / info.kd) * (target - q),
                              -.4 * info.velocity, .4 * info.velocity)
    velocity_target = np.clip(velocity_target,
                              -np.maximum(0., q - info.lower - .025) / .1,
                              np.maximum(0., info.upper - .025 - q) / .1)
    damping = np.maximum(info.kd, info.effort / (.25 * info.velocity))
    return np.clip(damping * (velocity_target - v), -info.effort, info.effort)


def guarded_torque_torch(q, v, target, lower, upper, kp, kd, velocity, effort):
    """Batched counterpart, tested against the NumPy deployment controller."""
    import torch
    target = target.clamp(lower + .05, upper - .05)
    desired = ((kp / kd) * (target - q)).clamp(-.4 * velocity, .4 * velocity)
    desired = desired.clamp(-(q - lower - .025).clamp_min(0.) / .1,
                            (upper - .025 - q).clamp_min(0.) / .1)
    damping = torch.maximum(kd, effort / (.25 * velocity))
    return (damping * (desired - v)).clamp(-effort, effort)


class TrajectoryLimits:
    """Latch violations at every physics step; later recovery cannot erase them."""
    def __init__(self, info):
        self.info = info
        self.max_position_error = 0.
        self.max_speed_ratio = 0.
        self.max_effort_ratio = 0.
        self.first_violation = None
        self.steps = 0

    @property
    def ok(self):
        return self.first_violation is None

    def observe(self, data):
        i = self.info
        q, v = data.qpos[i.qadr], data.qvel[i.vadr]
        position = np.maximum(i.lower - q, q - i.upper).clip(0.)
        speed = np.abs(v) / i.velocity
        effort = np.abs(data.ctrl) / i.effort
        self.steps += 1
        if not (np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(effort).all()):
            if self.ok:
                self.first_violation = {"time_s": float(data.time), "kind": "nonfinite"}
            return
        self.max_position_error = max(self.max_position_error, float(position.max()))
        self.max_speed_ratio = max(self.max_speed_ratio, float(speed.max()))
        self.max_effort_ratio = max(self.max_effort_ratio, float(effort.max()))
        if self.ok:
            for kind, values, threshold in [("position", position, POSITION_TOLERANCE),
                                             ("speed", speed, 1. + RATIO_TOLERANCE),
                                             ("effort", effort, 1. + RATIO_TOLERANCE)]:
                if values.max() > threshold:
                    j = int(values.argmax())
                    self.first_violation = {"time_s": float(data.time), "kind": kind,
                                            "joint": i.names[j], "value": float(values[j])}
                    break

    def report(self):
        return {"ok": self.ok, "physics_steps": self.steps,
                "max_position_violation_rad": self.max_position_error,
                "max_speed_ratio": self.max_speed_ratio,
                "max_commanded_effort_ratio": self.max_effort_ratio,
                "first_violation": self.first_violation}


def reset_balance(info, data, seed):
    """Training-only standing perturbations, never used for recovery evaluation."""
    import mujoco
    rng = np.random.default_rng(seed)
    mujoco.mj_resetData(info.model, data)
    data.qpos[:] = info.standing
    data.qpos[:2] += rng.uniform(-.02, .02, 2)
    data.qpos[2] += .01
    data.qpos[info.qadr] += rng.uniform(-.01, .01, len(info.qadr))
    rotation = np.array([rng.uniform(-.015, .015), rng.uniform(-.015, .015), 0.])
    mujoco.mju_quatIntegrate(data.qpos[3:7], rotation, 1.)
    mujoco.mj_forward(info.model, data)
    audit = TrajectoryLimits(info)
    for _ in range(round(.4 / info.model.opt.timestep)):
        data.ctrl[:] = info.torque(data.qpos[info.qadr], data.qvel[info.vadr], info.nominal)
        mujoco.mj_step(info.model, data)
        audit.observe(data)
    if not audit.ok:
        raise RuntimeError(f"Invalid balance reset: {audit.report()}")
    data.time = 0.
    mujoco.mj_forward(info.model, data)


def reset_crouch(info, data, seed):
    """Reach a moderate crouch through controlled physics, for curriculum tests."""
    import mujoco
    reset_balance(info, data, seed)
    target = info.nominal.copy()
    for side in ("left", "right"):
        for part, value in [("hip_pitch", -.7), ("knee", 1.4), ("ankle_pitch", -.7)]:
            target[info.names.index(f"{side}_{part}_joint")] = value
    audit = TrajectoryLimits(info)
    for step in range(round(3.5 / info.model.opt.timestep)):
        phase = min(step * info.model.opt.timestep / 3., 1.)
        phase = phase * phase * (3. - 2. * phase)
        command = info.nominal * (1. - phase) + target * phase
        data.ctrl[:] = info.torque(data.qpos[info.qadr], data.qvel[info.vadr], command)
        mujoco.mj_step(info.model, data)
        audit.observe(data)
    if not audit.ok or data.qpos[2] < .45:
        raise RuntimeError(f"Invalid crouch curriculum reset: {audit.report()}")
    data.time = 0.
    mujoco.mj_forward(info.model, data)


def deep_crouch_target(info):
    """Validated deeper command; each joint stays inside guarded target margins."""
    target = info.nominal.copy()
    for side in ("left", "right"):
        for part, value in [("hip_pitch", -.9), ("knee", 1.6), ("ankle_pitch", -.7)]:
            target[info.names.index(f"{side}_{part}_joint")] = value
    if np.any(target < info.lower + .05) or np.any(target > info.upper - .05):
        raise ValueError('Deep crouch reference exceeds guarded command bounds')
    return target


def reset_deep_crouch(info, data, seed):
    """Physical 4 s lowering plus 1 s hold, matching the audited deeper reference."""
    import mujoco
    from .common import CONTROL_DT
    if info.physics_profile != 'guarded_v2':
        raise ValueError('Deep crouch requires guarded_v2')
    reset_balance(info, data, seed)
    target = deep_crouch_target(info)
    audit = TrajectoryLimits(info)
    audit.observe(data)
    for step in range(round(5. / CONTROL_DT)):
        phase = min(step * CONTROL_DT / 4., 1.)
        phase = phase * phase * (3. - 2. * phase)
        command = info.nominal * (1. - phase) + target * phase
        for _ in range(info.substeps):
            data.ctrl[:] = info.torque(data.qpos[info.qadr], data.qvel[info.vadr], command)
            mujoco.mj_step(info.model, data)
            audit.observe(data)
            if not audit.ok or not np.isfinite(data.qpos).all() or data.qpos[2] < .4:
                raise RuntimeError(f'Invalid deep crouch reset: {audit.report()}')
    data.time = 0.
    mujoco.mj_forward(info.model, data)
