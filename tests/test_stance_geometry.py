"""Posture checks must distinguish the learned inward stance from neutral stance."""
import mujoco
import numpy as np
from x2_recovery.common import ModelInfo
from x2_recovery.stance import stance_metrics


def test_nominal_stance_is_invariant_to_world_heading():
    info = ModelInfo()
    data = mujoco.MjData(info.model)
    data.qpos[:] = info.standing
    mujoco.mj_forward(info.model, data)
    original = stance_metrics(info, data)
    assert original['posture_ok']
    data.qpos[3:7] = [np.cos(.7), 0, 0, np.sin(.7)]
    mujoco.mj_forward(info.model, data)
    rotated = stance_metrics(info, data)
    assert rotated['posture_ok']
    np.testing.assert_allclose(rotated['signed_foot_width_m'], original['signed_foot_width_m'], atol=1e-8)
    np.testing.assert_allclose(rotated['foot_heading_error_rad'], original['foot_heading_error_rad'], atol=1e-7)


def test_inward_twisted_hips_are_rejected_as_clean_stance():
    info = ModelInfo()
    data = mujoco.MjData(info.model)
    data.qpos[:] = info.standing
    for side, angle in [('left', -1.24), ('right', 1.02)]:
        data.qpos[info.qadr[info.names.index(f'{side}_hip_yaw_joint')]] = angle
    mujoco.mj_forward(info.model, data)
    result = stance_metrics(info, data)
    assert not result['posture_ok']
    assert not result['checks']['hip_yaw']
    assert not result['checks']['foot_heading']
