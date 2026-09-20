"""Bounded PPO landing corrections around an explicitly external pretrained teacher.

The teacher supplies recovery. Our separately initialized PPO learns two small
ankle-target offsets, applied only during a fixed touchdown interval. This is
not an independently learned stand-up policy and does not replace the historical ROS actor.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from .vendor_controller import Teacher
from .common import (CONTROL_DT, ModelInfo, ground_forces, observation_numpy,
                     reset_cpu, sensor_forces, success_conditions)
from .physics import TrajectoryLimits
from .stance import stance_metrics

CONTRACT = 'landing_residual_v1: shared_observation[:75] + elapsed/15 + base_ankle_actions2 -> ankle_target_offsets_rad2'
EXPORT_CONTRACT = 'TorchScript: same 78 inputs -> normalized tanh corrections2; multiply by 0.04 radians and landing_window(elapsed_seconds)'
JOINTS = ['left_ankle_pitch_joint', 'right_ankle_pitch_joint']
BOUND = .04
STD = .2
GAMMA = .995
LAMBDA = .95
GRAPH = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action_from_target(info, target):
    delta = target - info.nominal
    return np.clip(delta / np.where(delta >= 0, info.action_positive, info.action_negative), -1, 1)


def landing_window(t):
    def smooth(u):
        u = float(np.clip(u, 0, 1))
        return u * u * (3 - 2 * u)
    return smooth((t - 9.) / .3) * smooth((11. - t) / .3)


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(78, 64), nn.ELU(), nn.Linear(64, 64),
                                   nn.ELU(), nn.Linear(64, 2))
        self.critic = nn.Sequential(nn.Linear(78, 64), nn.ELU(), nn.Linear(64, 64),
                                    nn.ELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, observation):
        return torch.tanh(self.actor(observation)) * BOUND


def gae(rewards, values, gamma=GAMMA, lam=LAMBDA):
    """Finite 15 s task horizon and fault stops are terminal, not truncations."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    advantage = np.zeros_like(rewards)
    carry = 0.
    for t in range(len(rewards) - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < len(values) else 0.
        delta = rewards[t] + gamma * next_value - values[t]
        carry = delta + gamma * lam * carry
        advantage[t] = carry
    return advantage, advantage + values


def initialize_worker(asset_dir):
    global GRAPH
    from .vendor_controller import VendorGraph
    torch.set_num_threads(1)
    GRAPH = VendorGraph(asset_dir)


def episode(seed, model, stochastic=False, action_seed=0, on_step=None, graph=None):
    graph = graph if graph is not None else GRAPH
    if graph is None:
        raise RuntimeError('External teacher assets are required for this composite controller')
    rng = np.random.default_rng(action_seed)
    info = ModelInfo(physics_profile='guarded_v2')
    data = mujoco.MjData(info.model)
    reset_cpu(info, data, seed)
    teacher = Teacher(graph, info, data, timescale=1.2)
    indices = np.array([info.names.index(n) for n in JOINTS])
    previous = np.zeros(31, np.float32)
    audit = TrajectoryLimits(info)
    audit.observe(data)
    buffer = {k: [] for k in ['observations', 'raw_actions', 'logprob', 'values', 'rewards', 'active']}
    rows, states = [], []
    hold = maxhold = max_residual = 0.
    maxheight = float(data.qpos[2])
    start = time.monotonic()
    for step in range(750):
        t = step * CONTROL_DT
        base = teacher.target(info, data, t)
        base_action = action_from_target(info, base)
        obs = np.concatenate([observation_numpy(info, data, previous, sensor_forces(info, data))[:75],
                              [t / 15.], base_action[indices]]).astype(np.float32)
        with torch.inference_mode():
            tensor = torch.from_numpy(obs).unsqueeze(0)
            mean = model.actor(tensor).numpy()[0]
            value = float(model.critic(tensor)[0, 0])
        raw = (mean + STD * rng.standard_normal(2)).astype(np.float32) if stochastic else mean
        logprob = float(np.sum(-.5 * ((raw - mean) / STD) ** 2 - math.log(STD) - .5 * math.log(2 * math.pi)))
        window = landing_window(t)
        offset = np.tanh(raw) * BOUND * window
        command = base.copy()
        command[indices] += offset
        command = np.clip(command, info.lower + .05, info.upper - .05)
        max_residual = max(max_residual, float(np.abs(offset).max()))
        previous = action_from_target(info, command).astype(np.float32)
        step_peak = 0.
        for _ in range(info.substeps):
            data.ctrl[:] = info.torque(data.qpos[info.qadr], data.qvel[info.vadr], command)
            mujoco.mj_step(info.model, data)
            audit.observe(data)
            step_peak = max(step_peak, float(np.max(np.abs(data.qvel[info.vadr]) / info.velocity)))
            if not audit.ok:
                break
        checks = success_conditions(info, data, ground_forces(info, data))
        stance = stance_metrics(info, data)
        clean = all(checks.values()) and stance['posture_ok'] and audit.ok
        hold = hold + CONTROL_DT if clean else 0.
        maxhold = max(maxhold, hold)
        maxheight = max(maxheight, float(data.qpos[2]))
        up = float(data.xmat[info.torso_id].reshape(3, 3)[2, 2])
        reward = CONTROL_DT * (float(np.clip(data.qpos[2] / info.standing[2], 0, 1.2))
                              + float(np.clip(up, 0, 1)) + .5 * checks['both_feet'] + 2. * clean
                              - .5 * max(step_peak - .75, 0.) ** 2
                              - .001 * float(np.sum(np.tanh(raw) ** 2)) * window)
        if not audit.ok:
            reward -= 5.
        elif step == 749 and hold >= 2.:
            reward += 5.
        buffer['observations'].append(obs)
        buffer['raw_actions'].append(raw)
        buffer['logprob'].append(logprob)
        buffer['values'].append(value)
        buffer['rewards'].append(reward)
        buffer['active'].append(window > 0.)
        if step % 5 == 0:
            rows.append(dict(time=float(data.time), height=float(data.qpos[2]), clean_hold=hold,
                             ankle_offsets_rad=offset.tolist(), step_speed_ratio=step_peak))
            states.append(data.qpos.copy())
        if on_step is not None:
            on_step(info, data, step, hold, audit)
        if not audit.ok:
            break
    passed = bool(step == 749 and hold >= 2. and audit.ok)
    result = dict(seed=seed, passed=passed, terminal_hold=hold, max_clean_hold=maxhold,
                  max_height=maxheight, sim_time=float(data.time), limits=audit.report(),
                  final_checks=checks, final_stance=stance, reward=sum(buffer['rewards']),
                  max_residual_rad=max_residual, stochastic=stochastic, action_seed=action_seed,
                  contract=CONTRACT, reference_timescale=1.2, external_teacher_at_inference=True,
                  teacher_policy_sha256=sha(graph.path), state_edits_after_reset=False,
                  wall_seconds=time.monotonic() - start, trace=rows)
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in buffer.items()}
    arrays['advantages'], arrays['returns'] = gae(arrays['rewards'], arrays['values'])
    return result, arrays, np.asarray(states)


def worker_job(job):
    weights, seed, stochastic, action_seed = job
    model = Residual().eval()
    model.load_state_dict(weights)
    result, arrays, states = episode(seed, model, stochastic, action_seed)
    return result, arrays if stochastic else None, states


def batch(pool, model, seeds, stochastic=False, update=0):
    weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    jobs = [(weights, seed, stochastic, 51000 + update * 100 + i) for i, seed in enumerate(seeds)]
    return list(pool.map(worker_job, jobs))


def save_checkpoint(model, output, update, metadata):
    path = output / f'update_{update:03}.pt'
    torch.save(dict(model=model.state_dict(), metadata=metadata), path)
    actor = nn.Sequential(model.actor, nn.Tanh(), nn.Hardtanh(-1., 1.))
    # The export returns normalized corrections; multiply by 0.04 and the window.
    torch.jit.save(torch.jit.script(actor.eval()), str(output / f'update_{update:03}_actor.pt'))
    (output / f'update_{update:03}_actor.json').write_text(json.dumps(dict(
        contract=EXPORT_CONTRACT, joints=JOINTS, bound_rad=BOUND, window_seconds=[9., 11.],
        teacher_required=True, checkpoint_sha256=sha(path)), indent=2) + '\n')
    return path


def train(args):
    torch.manual_seed(args.seed)
    model = Residual()
    initial_actor = {k: v.detach().clone() for k, v in model.actor.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metadata = dict(method='PPO_residual_on_external_pretrained_recovery', contract=CONTRACT,
                    joints=JOINTS, bound_rad=BOUND, gaussian_std=STD, landing_interval=[9., 11.],
                    gamma=GAMMA, gae_lambda=LAMBDA, clip=.1, epochs=4, minibatch_size=512,
                    learning_rate=1e-4, actor_active_only_in_landing_window=True,
                    seed=args.seed, updates=0, environment_steps=0,
                    note='External teacher supplies recovery; only the bounded ankle corrections are trained here.')
    ctx = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
            initializer=initialize_worker, initargs=(str(args.asset_dir.resolve()),)) as pool:
        rows, validations = [], []
        start = time.monotonic()
        save_checkpoint(model, args.output, 0, metadata.copy())
        for update in range(args.updates + 1):
            if update > 0:
                # Known failures are explicit development/training cases, not held-out evidence.
                seeds = [9503, 9513, 9501, 9401] + list(range(11000 + (update - 1) * 4, 11004 + (update - 1) * 4))
                episodes = batch(pool, model, seeds, True, update)
                arrays = {k: torch.from_numpy(np.concatenate([e[1][k] for e in episodes]))
                          for k in episodes[0][1]}
                active = arrays['active'].bool()
                if not active.any():
                    raise RuntimeError('No landing decisions survived to train on')
                advantages = arrays['advantages']
                normalized = (advantages - advantages[active].mean()) / (advantages[active].std() + 1e-8)
                last_loss = None
                kl_values = []
                for _ in range(4):
                    permutation = torch.randperm(len(advantages))
                    for ids in permutation.split(512):
                        obs = arrays['observations'][ids]
                        distribution = torch.distributions.Normal(model.actor(obs), STD)
                        logprob = distribution.log_prob(arrays['raw_actions'][ids]).sum(-1)
                        ratio = (logprob - arrays['logprob'][ids]).exp()
                        use = active[ids]
                        if use.any():
                            surrogate = torch.minimum(ratio[use] * normalized[ids][use],
                                                      ratio[use].clamp(.9, 1.1) * normalized[ids][use])
                            policy_loss = -surrogate.mean()
                            with torch.no_grad():
                                kl_values.append(float(((ratio[use] - 1) - torch.log(ratio[use])).mean()))
                        else:
                            policy_loss = logprob.sum() * 0
                        value_loss = (model.critic(obs).squeeze(-1) - arrays['returns'][ids]).square().mean()
                        loss = policy_loss + .5 * value_loss
                        optimizer.zero_grad(); loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        last_loss = float(loss.detach())
                metadata['updates'] = update
                metadata['environment_steps'] += len(advantages)
                metadata['wall_seconds'] = time.monotonic() - start
                actor_delta = sum(float((v - initial_actor[k]).square().sum())
                                  for k, v in model.actor.state_dict().items()) ** .5
                row = dict(update=update, training_successes=sum(e[0]['passed'] for e in episodes),
                           episodes=len(episodes), mean_reward=float(np.mean([e[0]['reward'] for e in episodes])),
                           limit_faults=sum(not e[0]['limits']['ok'] for e in episodes),
                           total_environment_steps=metadata['environment_steps'], active_decisions=int(active.sum()),
                           actor_weight_change_l2=actor_delta, mean_approx_kl=float(np.mean(kl_values)),
                           final_minibatch_loss=last_loss, wall_seconds=metadata['wall_seconds'])
                rows.append(row)
                print(json.dumps(row), flush=True)
                for result, _, _ in episodes:
                    (args.output / f'train_{update:03}_{result["seed"]}.json').write_text(json.dumps(result, indent=2) + '\n')
                save_checkpoint(model, args.output, update, metadata.copy())
                (args.output / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
            if update in {0, 2, 4, args.updates}:
                evaluated = batch(pool, model, [11201, 11202, 11203, 11204, 11205])
                for result, _, states in evaluated:
                    (args.output / f'validation_{update:03}_{result["seed"]}.json').write_text(json.dumps(result, indent=2) + '\n')
                    np.savez_compressed(args.output / f'validation_{update:03}_{result["seed"]}.npz', qpos=states)
                row = dict(update=update, successes=sum(e[0]['passed'] for e in evaluated), episodes=5,
                           faults=sum(not e[0]['limits']['ok'] for e in evaluated),
                           mean_reward=float(np.mean([e[0]['reward'] for e in evaluated])),
                           max_speed_ratio=max(e[0]['limits']['max_speed_ratio'] for e in evaluated))
                validations.append(row)
                print(json.dumps({'validation': row}), flush=True)
                (args.output / 'validation_summary.json').write_text(json.dumps(validations, indent=2) + '\n')
        metadata['total_wall_seconds'] = time.monotonic() - start
        metadata['actual_updates'] = args.updates
        metadata['training_seed_rule'] = 'Four known development cases + four fresh training seeds per update'
        metadata['validation_seeds'] = [11201, 11202, 11203, 11204, 11205]
        (args.output / 'training.json').write_text(json.dumps(metadata, indent=2) + '\n')


def evaluate(args):
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    model = Residual().eval(); model.load_state_dict(checkpoint['model'])
    ctx = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx,
            initializer=initialize_worker, initargs=(str(args.asset_dir.resolve()),)) as pool:
        episodes = batch(pool, model, args.seeds)
    for result, _, states in episodes:
        result['checkpoint_sha256'] = sha(args.checkpoint)
        (args.output / f'{result["seed"]}.json').write_text(json.dumps(result, indent=2) + '\n')
        np.savez_compressed(args.output / f'{result["seed"]}.npz', qpos=states)
    summary = dict(seeds=args.seeds, successes=sum(e[0]['passed'] for e in episodes), episodes=len(episodes),
                   limit_faults=sum(not e[0]['limits']['ok'] for e in episodes), checkpoint_sha256=sha(args.checkpoint),
                   maximum_speed_ratio=max(e[0]['limits']['max_speed_ratio'] for e in episodes),
                   mean_reward=float(np.mean([e[0]['reward'] for e in episodes])), external_teacher_at_inference=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'evaluate'])
    p.add_argument('--asset-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--updates', type=int, default=6)
    p.add_argument('--seed', type=int, default=51)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--seeds', type=int, nargs='+', default=[11301, 11302, 11303, 11304, 11305])
    a = p.parse_args()
    if not 1 <= a.workers <= 4 or a.updates < 1:
        p.error('Use 1–4 workers and at least one update')
    if a.mode == 'evaluate' and a.checkpoint is None:
        p.error('Evaluation requires --checkpoint')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    globals()[a.mode](a)


if __name__ == '__main__':
    main()
