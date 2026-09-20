"""Show/record the experimental teacher-derived student without its teacher."""
import argparse
import copy
import contextlib
import json
import threading
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from train_reference_student import rollout, sha


class Recording:
    def __init__(self, output, live):
        self.output = output
        self.live = live
        self.renderer = self.viewer = self.writer = None
        self.threads = []
        self.frames = 0

    def __call__(self, info, data, step, hold, monitor):
        if self.renderer is None:
            # launch_passive runs mj_forward. Isolate its data/model so opening
            # the window cannot alter the policy's touch observations or solver.
            self.view_model = copy.copy(info.model)
            self.view_data = mujoco.MjData(self.view_model)
            mujoco.mj_copyData(self.view_data, self.view_model, data)
            self.renderer = mujoco.Renderer(self.view_model, height=480, width=640)
            self.camera = mujoco.MjvCamera()
            self.camera.distance = 2.9
            self.camera.azimuth = 125
            self.camera.elevation = -20
            self.options = mujoco.MjvOption()
            self.options.geomgroup[3] = 0
            self.writer = imageio.get_writer(self.output / 'student_recovery.mp4', fps=25)
            if self.live:
                from mujoco import viewer
                existing = set(threading.enumerate())
                self.viewer = viewer.launch_passive(self.view_model, self.view_data)
                self.threads = [t for t in threading.enumerate()
                                if t not in existing and t.name.endswith('(_launch_internal)')]
                self.viewer.cam.distance = 2.9
                self.viewer.cam.azimuth = 125
                self.viewer.cam.elevation = -20
                self.viewer.opt.geomgroup[3] = 0
            self.start = time.monotonic()
        with self.viewer.lock() if self.viewer is not None else contextlib.nullcontext():
            mujoco.mj_copyData(self.view_data, self.view_model, data)
        self.camera.lookat[:] = [data.qpos[0], data.qpos[1], .55]
        if step % 2 == 1:
            self.renderer.update_scene(self.view_data, camera=self.camera, scene_option=self.options)
            frame = Image.new('RGB', (640, 592), (16, 23, 31))
            frame.paste(Image.fromarray(self.renderer.render()), (0, 80))
            draw = ImageDraw.Draw(frame)
            font = ImageFont.load_default(size=16)
            draw.text((12, 7), 'LEARNED STUDENT | EXTERNAL-TEACHER IMITATION', fill=(255, 220, 95), font=font)
            draw.text((12, 29), 'Student alone at inference | real MuJoCo dynamics', fill='white', font=font)
            draw.text((12, 51), f'Time {data.time:5.2f} s | clean hold {hold:.2f} s | limits '
                      + ('PASS' if monitor.ok else 'FAIL'), fill='white', font=font)
            draw.text((12, 568), 'Supervised training; zero PPO updates | no state teleporting', fill=(255, 220, 95), font=font)
            self.writer.append_data(np.asarray(frame))
            self.frames += 1
            if self.frames in [1, 150, 190, 230, 290, 375]:
                frame.save(self.output / f'frame_{self.frames:04}.png')
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.cam.lookat[:] = self.camera.lookat
            self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  'LEARNED STUDENT\nExternal-teacher imitation, not PPO\nSimulation time\nClean standing hold\nTrajectory limits',
                                  f'\n\n{data.time:.2f} s\n{hold:.2f} s\n' + ('PASS' if monitor.ok else 'FAIL')))
            self.viewer.sync(state_only=True)
        if self.live:
            time.sleep(max(0., (step + 1) * .02 - (time.monotonic() - self.start)))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            for thread in self.threads:
                thread.join(timeout=5.)
                if thread.is_alive():
                    raise RuntimeError('Viewer failed to close')
        if self.renderer is not None:
            self.renderer.close()
        if self.writer is not None:
            self.writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--live', action='store_true')
    p.add_argument('--expected', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    actor = torch.jit.load(str(a.actor)).eval()
    recording = Recording(a.output, a.live)
    try:
        result, _, _, _ = rollout(a.seed, actor=actor, on_step=recording)
    finally:
        recording.close()
    result['actor_sha256'] = sha(a.actor)
    expected = json.loads(a.expected.read_text())
    keys = ['passed', 'terminal_hold', 'max_clean_hold', 'max_height', 'limits',
            'final_checks', 'final_stance', 'actor_sha256', 'sim_time']
    checks = {k: result[k] == expected[k] for k in keys}
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (a.output / 'checks.json').write_text(json.dumps(dict(checks=checks, frames=recording.frames,
                                                        physical_resimulation=True), indent=2) + '\n')
    assert all(checks.values()), checks
    print(json.dumps(dict(checks=checks, viewer_closed=True)), flush=True)


if __name__ == '__main__':
    main()
