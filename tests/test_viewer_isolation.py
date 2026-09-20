"""Opening or manipulating a display must not change the evaluated dynamics."""
import contextlib
from types import SimpleNamespace

import mujoco
import numpy as np
from mujoco import viewer

from x2_recovery.runtime import RecoveryRuntime


def test_viewer_model_data_and_forward_calls_are_isolated(monkeypatch):
    class PerturbingDisplay:
        def __init__(self, model, data):
            self.model, self.data = model, data
            self.cam = SimpleNamespace(lookat=np.zeros(3))
            self.opt = SimpleNamespace(geomgroup=np.ones(6))
            self.closed = False
            # Actual launch_passive calls mj_forward before opening the window.
            mujoco.mj_forward(model, data)
            data.qpos[0] += .5
            model.opt.gravity[0] = 9.

        def lock(self):
            return contextlib.nullcontext()

        def is_running(self):
            return not self.closed

        def set_texts(self, _texts):
            pass

        def sync(self, state_only=False):
            assert state_only
            self.data.qvel[0] += 1.
            self.data.sensordata[:] = 1e6
            mujoco.mj_forward(self.model, self.data)

        def close(self):
            self.closed = True

    monkeypatch.setattr(viewer, 'launch_passive', PerturbingDisplay)
    headless = RecoveryRuntime(seed=123, physics_profile='guarded_v2')
    displayed = RecoveryRuntime(seed=123, physics_profile='guarded_v2', render=True)
    assert displayed.model is not displayed.viewer.model
    assert displayed.data is not displayed.viewer.data
    try:
        for _ in range(30):
            assert headless.step() == displayed.step()
            for field in ['qpos', 'qvel', 'qacc_warmstart', 'sensordata', 'ctrl']:
                np.testing.assert_array_equal(getattr(headless.data, field), getattr(displayed.data, field))
    finally:
        headless.close()
        displayed.close()
