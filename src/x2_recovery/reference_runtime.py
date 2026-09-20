"""ROS bridge for an explicitly external teacher plus trained PPO corrections."""
import copy
import threading
import time
from pathlib import Path

import mujoco
import torch

from .landing_residual import Residual, episode
from .vendor_controller import VendorGraph


class SimulationDurationTimeout(Exception):
    pass


class ReferenceDisplay:
    """Display a separate model/data copy; never modify the controlled state."""
    def __init__(self):
        self.viewer = None
        self.threads = []

    def update(self, info, data, hold, audit):
        if self.viewer is None:
            from mujoco import viewer
            self.model = copy.copy(info.model)
            self.data = mujoco.MjData(self.model)
            mujoco.mj_copyData(self.data, self.model, data)
            before = set(threading.enumerate())
            self.viewer = viewer.launch_passive(self.model, self.data)
            self.threads = [t for t in threading.enumerate()
                            if t not in before and t.name.endswith('(_launch_internal)')]
            self.viewer.cam.distance = 2.9
            self.viewer.cam.azimuth = 125
            self.viewer.cam.elevation = -20
            self.viewer.opt.geomgroup[3] = 0
        if self.viewer.is_running():
            with self.viewer.lock():
                mujoco.mj_copyData(self.data, self.model, data)
                self.viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], .55]
            self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                'EXTERNAL TEACHER + TRAINED PPO CORRECTION\nTeacher provides recovery\nEpisode time\nClean standing hold\nLimits',
                f'\n\n{data.time:.2f} s\n{hold:.2f} s\n' + ('PASS' if audit.ok else 'FAIL')))
            self.viewer.sync(state_only=True)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            for thread in self.threads:
                thread.join(timeout=5.)
                if thread.is_alive():
                    raise RuntimeError('Reference viewer failed to close')


def run_reference_episode(config, publish_frame):
    """Publish genuine simulator state and succeed only after the full 15 s audit."""
    if config.get('physics_profile') != 'guarded_v2':
        raise ValueError('reference_residual requires guarded_v2 physics')
    torch.set_num_threads(1)
    graph = VendorGraph(Path(config['vendor_assets']))
    checkpoint = torch.load(config['checkpoint'], weights_only=True, map_location='cpu')
    model = Residual().eval()
    model.load_state_dict(checkpoint['model'])
    display = ReferenceDisplay() if config['render'] else None
    start = time.monotonic()

    def observe(info, data, step, hold, audit):
        publish_frame(dict(joint_names=list(info.names),
                           joint_positions=data.qpos[info.qadr].tolist(),
                           sim_time=float(data.time), success=False, invalid=not audit.ok))
        # A shorter configured timeout must not become a successful prefix.
        limit = config['max_sim_duration_s']
        if data.time > limit + 1e-8 or (step < 749 and data.time >= limit - 1e-8):
            raise SimulationDurationTimeout()
        if display is not None:
            display.update(info, data, hold, audit)
        if config['realtime']:
            time.sleep(max(0., (step + 1) * .02 - (time.monotonic() - start)))

    try:
        result, _, _ = episode(config['seed'], model, graph=graph, on_step=observe)
        reason = ('full 15-second clean-stance and trajectory-limit checks passed'
                  if result['passed'] else 'trajectory limits failed'
                  if not result['limits']['ok'] else 'full episode ended without stable standing')
        return dict(success=result['passed'], reason=reason, reference_result=result)
    except SimulationDurationTimeout:
        return dict(success=False, reason='simulation duration timeout')
    finally:
        if display is not None:
            display.close()
