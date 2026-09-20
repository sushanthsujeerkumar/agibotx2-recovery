#!/usr/bin/env python3
"""Train full-episode PPO feedback on the independently prepared motion prior."""
import argparse
import json
from pathlib import Path
import random
import signal
import subprocess
import sys

import numpy as np
import torch

from x2_recovery.batch_viewer import BatchPublisher
from x2_recovery.full_recovery import PREPARED
from x2_recovery.full_recovery_env import FullRecoveryEnv
from x2_recovery.train import RecoveryRunner, TrainingStopped, ppo_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=1800.)
    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--visible', type=int, default=16)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--evaluate-final', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--prepared', type=Path, default=PREPARED)
    parser.add_argument('--seed', type=int, default=17201)
    args = parser.parse_args()
    if args.seconds <= 0 or args.iterations <= 0 or not 1 <= args.visible <= args.num_envs:
        parser.error('Use positive duration/iterations and 1 <= visible <= num-envs')
    args.output = args.output.resolve()
    if (args.output/'progress.jsonl').exists() and not args.resume:
        parser.error('Existing run: use --resume or choose a fresh output directory')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cfg = ppo_config(seed=args.seed, save_interval=25, variant='stance')
    cfg['actor']['distribution_cfg'].update(init_std=.1, std_range=(.02, .25))
    cfg['algorithm'].update(learning_rate=1e-4, entropy_coef=0., clip_param=.05)
    cfg['freeze_actor_normalization'] = True
    bound = .01
    if args.resume:
        check = torch.load(args.resume, map_location='cpu', weights_only=False)
        actions = check['motion_prior_actions']
        cfg = check['train_cfg']
        bound = check['environment_cfg']['feedback_bound']
    else:
        report = json.loads((args.prepared/'preparation.json').read_text())
        if report['passes'] == 0:
            raise RuntimeError('Prepared motion prior has no successful validation episodes')
        actions = torch.from_numpy(np.load(args.prepared/'reference_examples.npz')['fitted_actions'])
    publisher = None
    env = None
    runner = None
    try:
        if not args.headless:
            publisher = BatchPublisher(args.output, visible=args.visible)
        env = FullRecoveryEnv(actions, feedback_bound=bound, publisher=publisher,
                              num_envs=args.num_envs, device='cuda:0', seed=args.seed)
        runner = RecoveryRunner(env, cfg, str(args.output), 'cuda:0', args.seconds, print_interval=25)
        # Numerical logs and checkpoint metadata are the reproducibility record.
        runner.logger._store_code_state = lambda: []
        if args.resume:
            runner.load(str(args.resume))
        else:
            last = [m for m in runner.alg.get_policy().mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
            torch.nn.init.zeros_(last.weight)
            torch.nn.init.zeros_(last.bias)
            runner.initialization = dict(method='zero_feedback_on_own_supervised_motion_prior',
                                         full_episode_feedback=True, external_teacher=False)
        runner.alg.get_policy().obs_normalizer.until = 0
        (args.output/'config.json').write_text(json.dumps(dict(training=cfg, environment=env.cfg,
                                            prior_preparation=str(args.prepared/'preparation.json')), indent=2)+'\n')
        def request_stop(*_):
            runner.stop_requested = True
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        runner.status('RUNNING')
        try:
            runner.learn(max(0, args.iterations-runner.completed_iterations), init_at_random_ep_len=False)
        except TrainingStopped:
            pass
        runner.save(str(args.output/f'model_{runner.completed_iterations:06d}.pt'))
        runner.status('PAUSED', note='Bounded run finished; long training requires an explicit launch')
    except BaseException as error:
        if runner:
            runner.status('FAILED', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if runner and runner.logger.writer:
            runner.logger.writer.close()
        if env:
            env.close()
        elif publisher:
            publisher.close()
    if args.evaluate_final and runner and not runner.stop_requested:
        subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate_full_recovery.py')),
                        '--actor', str(args.output/'actor_latest.pt'),
                        '--output', str(args.output/'evaluation.json'),
                        '--episodes', '10', '--seed', str(args.seed+10000)], check=True)


if __name__ == '__main__':
    main()
