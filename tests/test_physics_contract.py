"""Checks that guard against physically misleading training/evaluation results."""
import numpy as np
import mujoco
from x2_recovery.common import ModelInfo, reset_cpu, ground_forces, success_conditions


def test_actions_reach_full_joint_limits_and_respect_effort():
    i = ModelInfo()
    np.testing.assert_allclose(i.targets(np.ones(i.model.nu)), i.upper)
    np.testing.assert_allclose(i.targets(-np.ones(i.model.nu)), i.lower)
    tau = i.torque(i.lower, np.zeros(i.model.nu), i.upper)
    assert np.all(np.abs(tau) <= i.effort)
    tau = i.torque(i.lower, i.velocity+1., i.upper)
    assert np.all(tau <= 0)


def test_supine_reset_is_finite_supported_and_not_successful():
    i = ModelInfo()
    d = mujoco.MjData(i.model)
    reset_cpu(i, d, seed=1001)
    assert np.isfinite(d.qpos).all()
    assert d.qpos[2] < .25
    assert np.linalg.norm(d.qvel) < .2
    assert min([c.dist for c in d.contact[:d.ncon]] or [0.]) > -.003
    forces = ground_forces(i, d)
    assert forces[2] > 2
    assert not all(success_conditions(i,d,forces).values())


def test_success_rejects_extra_support_or_airborne_feet():
    i = ModelInfo()
    d = mujoco.MjData(i.model)
    d.qpos[:] = i.standing
    mujoco.mj_forward(i.model,d)
    assert not success_conditions(i,d,np.array([100.,100.,20.]))["no_other_support"]
    assert not success_conditions(i,d,np.array([0.,0.,0.]))["both_feet"]
