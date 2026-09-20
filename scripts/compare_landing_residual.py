"""Paired, deterministic assessment of a learned residual and its zero baseline."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path

import numpy as np
import torch

from train_landing_residual import Residual, initialize_worker, worker_job, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    jobs, names = [], []
    for name, path in [('baseline', a.baseline), ('residual', a.checkpoint)]:
        (a.output / name).mkdir()
        state = torch.load(path, weights_only=True, map_location='cpu')['model']
        for seed in a.seeds:
            jobs.append((state, seed, False, 0)); names.append(name)
    ctx = multiprocessing.get_context('spawn')
    results = {'baseline': {}, 'residual': {}}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=ctx,
            initializer=initialize_worker, initargs=(str(a.asset_dir.resolve()),)) as pool:
        futures = {pool.submit(worker_job, job): name for job, name in zip(jobs, names)}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            result, _, states = future.result()
            path = a.baseline if name == 'baseline' else a.checkpoint
            result['checkpoint_sha256'] = sha(path)
            results[name][result['seed']] = result
            (a.output / name / f'{result["seed"]}.json').write_text(json.dumps(result, indent=2) + '\n')
            np.savez_compressed(a.output / name / f'{result["seed"]}.npz', qpos=states)
            print(json.dumps({'controller': name, 'seed': result['seed'], 'passed': result['passed'],
                              'speed': result['limits']['max_speed_ratio']}), flush=True)
    summary = dict(seeds=a.seeds, selection_uses_these_seeds=False,
                   baseline_checkpoint_sha256=sha(a.baseline), residual_checkpoint_sha256=sha(a.checkpoint),
                   external_teacher_required_by_both=True, controllers={}, pairs=[])
    for name, rows in results.items():
        summary['controllers'][name] = dict(successes=sum(r['passed'] for r in rows.values()),
            episodes=len(rows), faults=sum(not r['limits']['ok'] for r in rows.values()),
            max_speed_ratio=max(r['limits']['max_speed_ratio'] for r in rows.values()))
    for seed in a.seeds:
        summary['pairs'].append(dict(seed=seed, baseline=results['baseline'][seed]['passed'],
                                     residual=results['residual'][seed]['passed']))
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['controllers']), flush=True)


if __name__ == '__main__':
    main()
