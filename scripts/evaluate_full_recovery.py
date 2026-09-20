#!/usr/bin/env python3
"""Evaluate a full exported policy, including its clock and strict physical limits."""
import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import torch
from x2_recovery.full_recovery import evaluate_actor


def evaluate(job):
    torch.set_num_threads(1)
    return evaluate_actor(*job)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=18001)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    with concurrent.futures.ProcessPoolExecutor(4, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(evaluate, [(args.actor, seed) for seed in range(args.seed, args.seed+args.episodes)]))
    report = dict(actor=str(args.actor.resolve()), passes=sum(r['passed'] for r in results),
                  episodes=len(results), results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}), flush=True)

    raise SystemExit(0 if report['passes'] == report['episodes'] else 1)
