"""Read-only batch display in a separate process; V toggles 16/all environments."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import mujoco
import numpy as np

from .common import ModelInfo


def grid_offsets(count, spacing=2.5):
    if count < 1:
        raise ValueError('At least one robot is required')
    columns = math.ceil(math.sqrt(count))
    offsets = np.zeros((count, 3))
    for index in range(count):
        row, column = divmod(index, columns)
        offsets[index, :2] = column*spacing, row*spacing
    offsets[:, :2] -= offsets[:, :2].mean(axis=0)
    return offsets


class BatchPublisher:
    """Only copies GPU data out; the viewer cannot write back into training."""
    def __init__(self, directory, visible=16, fps=5.):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory/'batch_snapshot.npz'
        self.period = 1./fps
        self.last_update = 0.
        self.sequence = 0
        self.copy_seconds = 0.
        self.log = (self.directory/'batch_viewer.log').open('w')
        self.process = subprocess.Popen([sys.executable, '-m', 'x2_recovery.batch_viewer',
                                        '--snapshot', str(self.path), '--visible', str(visible)],
                                       stdout=self.log, stderr=subprocess.STDOUT)

    def publish(self, env, force=False):
        now = time.monotonic()
        if not force and now-self.last_update < self.period:
            return
        self.last_update = now
        self.sequence += 1
        qpos = env.qpos.detach().cpu().numpy().copy()
        steps = env.episode_length_buf.detach().cpu().numpy().copy()
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, qpos=qpos, episode_steps=steps, sequence=self.sequence,
                     published_unix=time.time(), total_envs=env.num_envs)
        temporary.replace(self.path)
        self.copy_seconds += time.monotonic()-now

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.log.close()
        (self.directory/'batch_publisher_stats.json').write_text(json.dumps({
            'snapshots': self.sequence, 'copy_and_write_seconds': self.copy_seconds,
            'viewer_returncode': self.process.returncode,
            'display_has_no_writeback_channel': True}, indent=2)+'\n')


class BatchViewer:
    def __init__(self, visible=16):
        from mujoco import viewer
        self.info = ModelInfo(physics_profile='guarded_v2')
        self.model = self.info.model  # Independent model, owned only by this process.
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        self.data.qpos[:] = self.info.supine
        mujoco.mj_forward(self.model, self.data)
        self.visible = visible
        self.show_all = False
        self.last_count = 0
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.perturb = mujoco.MjvPerturb()
        previous = set(threading.enumerate())
        self.viewer = viewer.launch_passive(self.model, self.data, key_callback=self.key,
                                            show_left_ui=False, show_right_ui=False)
        self.threads = [t for t in threading.enumerate() if t not in previous and t.name.endswith('(_launch_internal)')]
        self.viewer.opt.geomgroup[3] = 0
        self.viewer.cam.azimuth = 115
        self.viewer.cam.elevation = -35

    def key(self, keycode):
        if keycode in (ord('v'), ord('V')):
            self.show_all = not self.show_all

    def update(self, qpos, sequence=0, age=0.):
        count = len(qpos) if self.show_all else min(self.visible, len(qpos))
        offsets = grid_offsets(count)
        with self.viewer.lock():
            self.data.qpos[:] = qpos[0]
            self.data.qpos[:3] += offsets[0]
            mujoco.mj_forward(self.model, self.data)
            scene = self.viewer.user_scn
            scene.ngeom = 0
            for index in range(1, count):
                self.scratch.qpos[:] = qpos[index]
                self.scratch.qpos[:3] += offsets[index]
                mujoco.mj_forward(self.model, self.scratch)
                mujoco.mjv_addGeoms(self.model, self.scratch, self.option, self.perturb,
                                  mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError('Batch display geometry capacity exceeded')
            if count != self.last_count:
                self.viewer.cam.lookat[:] = [0., 0., .3]
                self.viewer.cam.distance = max(5., math.ceil(math.sqrt(count))*3.3)
                self.last_count = count
        self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               'Training snapshots (read only)\nDisplayed / training\nSnapshot\nSnapshot age\nV key',
                               f'{"LIVE" if age < 2. else "PAUSED / STALE"}\n{count} / {len(qpos)}\n{sequence}\n{age:.1f} s\nToggle 16 / all'))
        self.viewer.sync(state_only=True)
        return count, int(self.viewer.user_scn.ngeom)

    def close(self):
        self.viewer.close()
        for thread in self.threads:
            thread.join(timeout=5.)
            if thread.is_alive():
                raise RuntimeError('Batch viewer thread failed to close')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--visible', type=int, default=16)
    parser.add_argument('--seconds', type=float, default=0.)
    parser.add_argument('--exercise-all', action='store_true')
    args = parser.parse_args()
    alive = True
    def stop(*_):
        nonlocal alive
        alive = False
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    display = None
    start = time.monotonic()
    last_sequence = -1
    frames = 0
    counts = set()
    render_seconds = 0.
    try:
        while alive and (not args.seconds or time.monotonic()-start < args.seconds):
            if not args.snapshot.exists():
                time.sleep(.1)
                continue
            with np.load(args.snapshot, allow_pickle=False) as frame:
                qpos = frame['qpos'].copy()
                sequence = int(frame['sequence'])
                age = time.time()-float(frame['published_unix'])
            if display is None:
                display = BatchViewer(args.visible)
                print('BATCH_VIEWER_OPEN', flush=True)
            if not display.viewer.is_running():
                break
            if args.exercise_all:
                display.show_all = 3. < time.monotonic()-start < 7.
            tick = time.monotonic()
            count, geoms = display.update(qpos, sequence, age)
            render_seconds += time.monotonic()-tick
            frames += 1
            counts.add(count)
            last_sequence = sequence
            time.sleep(max(0., .2-(time.monotonic()-tick)))
    finally:
        if display is not None:
            display.close()
        stats = dict(frames=frames, counts=sorted(counts), last_sequence=last_sequence,
                     render_seconds=render_seconds, pid=os.getpid())
        args.snapshot.with_name('batch_viewer_stats.json').write_text(json.dumps(stats, indent=2)+'\n')
        print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
