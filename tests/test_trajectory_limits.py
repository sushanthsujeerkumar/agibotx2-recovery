import mujoco
import numpy as np
import torch
from x2_recovery.common import ModelInfo
from x2_recovery.physics import TrajectoryLimits, guarded_torque_torch, reset_balance


def test_transient_joint_violation_is_latched_after_state_returns_to_bounds():
    info = ModelInfo()
    data = mujoco.MjData(info.model)
    data.qpos[:] = info.standing
    monitor = TrajectoryLimits(info)
    monitor.observe(data)
    assert monitor.ok
    j = info.names.index('left_wrist_roll_joint')
    data.qvel[info.vadr[j]] = 1.01 * info.velocity[j]
    monitor.observe(data)
    data.qvel[:] = 0.
    monitor.observe(data)
    assert not monitor.ok
    assert monitor.first_violation['joint'] == 'left_wrist_roll_joint'
    assert monitor.first_violation['kind'] == 'speed'
    np.testing.assert_allclose(monitor.max_speed_ratio, 1.01)


def test_numpy_and_training_torque_match_and_respect_effort():
    info = ModelInfo(physics_profile='guarded_v2')
    rng = np.random.default_rng(37)
    q = rng.uniform(info.lower, info.upper, (100, info.model.nu))
    v = rng.uniform(-2 * info.velocity, 2 * info.velocity, q.shape)
    target = rng.uniform(info.lower, info.upper, q.shape)
    expected = info.torque(q, v, target)
    actual = guarded_torque_torch(*(torch.tensor(a) for a in [q,v,target,info.lower,info.upper,info.kp,info.kd,info.velocity,info.effort]))
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)
    assert np.all(abs(expected) <= info.effort)
    # Over-speed motion must receive braking torque, even at a target boundary.
    assert np.all(expected[v > info.velocity] < 0.)
    assert np.all(expected[v < -info.velocity] > 0.)


def test_guarded_profile_preserves_physical_source_and_valid_balance_reset():
    old = ModelInfo()
    new = ModelInfo(physics_profile='guarded_v2')
    for field in ['body_mass','body_inertia','actuator_ctrlrange','jnt_range']:
        np.testing.assert_array_equal(getattr(old.model,field),getattr(new.model,field))
    assert new.substeps == 20
    data = mujoco.MjData(new.model)
    reset_balance(new,data,1001)
    monitor = TrajectoryLimits(new)
    monitor.observe(data)
    assert monitor.ok
    assert data.qpos[2] > .6
