#!/usr/bin/env python3
"""Fit an explicit full-episode motion prior from own physically valid examples."""
import argparse
import concurrent.futures
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import torch

from x2_recovery.full_recovery import REFERENCE, ReferenceController, MotionPrior, evaluate_actor
from x2_recovery.runtime import RecoveryRuntime


def collect(seed):
    torch.set_num_threads(1)
    record = json.loads(REFERENCE.read_text())
    runtime = RecoveryRuntime('scripted', physics_profile='guarded_v2', assess_stance=True)
    runtime.reset(seed)
    runtime.policy = ReferenceController(runtime, record)
    actions = []
    try:
        for step in range(750):
            state = runtime.step()
            actions.append(runtime.previous_action.copy())
            if state['invalid'] or not runtime.limits.ok:
                break
        passed = bool(step == 749 and runtime.limits.ok and runtime.clean_hold_time >= 2.-1e-8)
        row = dict(seed=seed, passed=passed, hold=runtime.clean_hold_time, limits=runtime.limits.report())
        return row, np.asarray(actions)
    finally:
        runtime.close()


def check(job):
    path, seed = job
    torch.set_num_threads(1)
    return evaluate_actor(path, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/runs/reprepared'))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Choose a new output directory to preserve earlier evidence')
    start = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(4, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(collect, range(17001, 17009)))
        successful = [actions for row, actions in results if row['passed']]
        if not successful:
            raise RuntimeError('No physically valid reference examples; do not train')
        # For samples on matching time knots, the least-squares fit is their mean.
        actions = np.mean(np.stack(successful), axis=0).astype(np.float32)
        prior = MotionPrior(torch.from_numpy(actions)).eval()
        torch.jit.script(prior).save(str(output/'motion_prior.pt'))
        np.savez_compressed(output/'reference_examples.npz', actions=np.stack(successful), fitted_actions=actions)
        validation = list(pool.map(check, [(output/'motion_prior.pt', seed) for seed in range(17101, 17111)]))
    report = dict(method='least_squares_phase_spline_from_own_physical_examples',
                  scope='Supervised reference prior; no new full-recovery PPO training yet',
                  observation_size=107, phase_input='elapsed_episode_seconds / 15',
                  external_teacher=False, collection=[r[0] for r in results], validation=validation,
                  passes=sum(r['passed'] for r in validation), episodes=len(validation), seconds=time.monotonic()-start)
    (output/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['collection', 'validation']}), flush=True)
