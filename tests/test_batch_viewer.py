"""Training state must remain unchanged even if the display is manipulated."""
import contextlib
from types import SimpleNamespace

import mujoco
from mujoco import viewer
import numpy as np

from x2_recovery.batch_viewer import BatchViewer, grid_offsets
from x2_recovery.common import ModelInfo


def test_grid_is_centered_and_has_distinct_positions():
    for count in (1, 16, 256):
        offsets = grid_offsets(count)
        assert len(np.unique(offsets, axis=0)) == count
        np.testing.assert_allclose(offsets.mean(axis=0), 0., atol=1e-12)
        assert np.all(offsets[:, 2] == 0.)


def test_all_256_states_render_without_modifying_input(monkeypatch):
    class Display:
        def __init__(self, model, data, key_callback, **kwargs):
            self.data = data
            self.cam = SimpleNamespace(lookat=np.zeros(3))
            self.opt = mujoco.MjvOption()
            self.user_scn = mujoco.MjvScene(model, maxgeom=20000)
        def lock(self): return contextlib.nullcontext()
        def set_texts(self, texts): pass
        def sync(self, state_only=False): self.data.qpos[0] = 999.
        def close(self): pass
    monkeypatch.setattr(viewer, 'launch_passive', Display)
    info = ModelInfo(physics_profile='guarded_v2')
    source = np.tile(info.supine, (256, 1))
    source[:, 0] += np.arange(256)*.001
    original = source.copy()
    display = BatchViewer()
    try:
        count, sixteen = display.update(source)
        assert count == 16 and sixteen > 0
        display.key(ord('V'))
        count, all_geoms = display.update(source)
        assert count == 256 and all_geoms*15 == sixteen*255
        np.testing.assert_array_equal(source, original)
        assert display.model is not info.model
        assert display.data.qpos[0] == 999.
    finally:
        display.close()
