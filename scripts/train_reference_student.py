"""Local teacher-guided student experiment; external-teacher provenance is retained.

This experimental actor has an explicit 107-input contract (the usual 106 plus
elapsed episode time / 15). It is not compatible with the default 106-input PPO
or ROS actor. Dataset collection and evaluation both use uninterrupted dynamics.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from x2_recovery.common import (CONTROL_DT, ModelInfo, ground_forces,
                                observation_numpy, reset_cpu, sensor_forces,
                                success_conditions)
from x2_recovery.physics import TrajectoryLimits
from x2_recovery.stance import stance_metrics

CONTRACT = 'x2_phase_student_v1: shared106 + elapsed_seconds/15 -> normalized_joint_targets31'
EXPECTED_POLICY_SHA256 = '3faf3df8f9616448f9ae580e22c2fa28fcb25c1eb0c9a338a0b0c5b9a520522f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action_from_target(info, target):
    delta = target - info.nominal
    return np.clip(delta / np.where(delta >= 0, info.action_positive, info.action_negative), -1, 1)


class Teacher:
    def __init__(self, graph, info, data, timescale=1.2):
        import yaml
        self.graph = graph
        self.order = np.array([info.names.index(n) for n in graph.names])
        self.initial = data.qpos[info.qadr].copy()
        self.init = info.nominal.copy()
        cfg = yaml.safe_load((graph.directory / 'ground_init_config.yaml').read_text())
        for name, value in zip(cfg['BaseConfig']['action_seq'], cfg['ACConfig']['robot']['default_dof_pos']):
            self.init[info.names.index(name)] = value
        self.previous = np.zeros(29, np.float32)
        self.timescale = timescale

    def target(self, info, data, t):
        if t < 5:
            u = min(t / 3., 1.)
            u = u * u * (3 - 2 * u)
            command = (1 - u) * self.initial + u * self.init
        else:
            g = self.graph
            frame = min(int((t - 5) / CONTROL_DT / self.timescale), len(g.references['joint_pos']) - 1)
            rotation = data.xmat[info.model.body('pelvis').id].reshape(3, 3)
            obs = np.concatenate([g.references['joint_pos'][frame],
                                  g.references['joint_vel'][frame] / self.timescale,
                                  rotation.T @ np.array([0., 0., -1.]), data.qvel[3:6] * .25,
                                  data.qpos[info.qadr[self.order]] - g.defaults,
                                  data.qvel[info.vadr[self.order]] * .05, self.previous]).astype(np.float32)
            self.previous = g.action(obs)
            command = info.nominal.copy()
            command[self.order] = g.defaults + g.scale * self.previous
        return np.clip(command, info.lower + .05, info.upper - .05)


class Student(nn.Module):
    def __init__(self, mean, std, drop_previous_action=False, phase_harmonics=0):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        mask = torch.ones_like(mean)
        if drop_previous_action:
            mask[75:106] = 0.
        self.register_buffer('input_mask', mask)
        self.register_buffer('phase_frequencies', torch.arange(1, phase_harmonics + 1, device=mean.device) * (2 * torch.pi))
        self.net = nn.Sequential(nn.Linear(107 + 2 * phase_harmonics, 512), nn.ELU(), nn.Linear(512, 256),
                                 nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 31))

    def raw_action(self, observation):
        phase = observation[:, 106:107] * self.phase_frequencies
        features = torch.cat([((observation - self.mean) / self.std) * self.input_mask,
                              torch.sin(phase), torch.cos(phase)], dim=-1)
        return self.net(features)

    def forward(self, observation):
        return self.raw_action(observation).clamp(-1., 1.)


def rollout(seed, graph=None, actor=None, teacher_fraction=0., on_step=None, prepare_target=None):
    info = ModelInfo(physics_profile='guarded_v2')
    data = mujoco.MjData(info.model)
    reset_cpu(info, data, seed)
    initial = data.qpos[info.qadr].copy()
    teacher = Teacher(graph, info, data) if graph is not None else None
    if actor is None and teacher is None:
        raise ValueError('A controller is required')
    previous = np.zeros(31, np.float32)
    monitor = TrajectoryLimits(info)
    monitor.observe(data)
    observations, actions, states, trace = [], [], [], []
    hold = maxhold = 0.
    maxheight = float(data.qpos[2])
    max_inverse_error = 0.
    start = time.monotonic()
    for step in range(round(15. / CONTROL_DT)):
        t = step * CONTROL_DT
        obs = np.append(observation_numpy(info, data, previous, sensor_forces(info, data)),
                        t / 15.).astype(np.float32)
        target = teacher.target(info, data, t) if teacher is not None else None
        label = action_from_target(info, target) if target is not None else None
        if label is not None:
            observations.append(obs)
            actions.append(label.astype(np.float32))
            max_inverse_error = max(max_inverse_error, float(np.abs(info.targets(label) - target).max()))
        if actor is None:
            command = target
            previous = label.astype(np.float32)
        else:
            with torch.inference_mode():
                student_action = actor(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
            previous = student_action if label is None else (
                teacher_fraction * label + (1 - teacher_fraction) * student_action).astype(np.float32)
            command = info.targets(previous)
            if prepare_target is not None and t < 5:
                u = min(t / 3., 1.)
                u = u * u * (3 - 2 * u)
                command = np.clip((1 - u) * initial + u * prepare_target, info.lower + .05, info.upper - .05)
                previous = action_from_target(info, command).astype(np.float32)
        for _ in range(info.substeps):
            data.ctrl[:] = info.torque(data.qpos[info.qadr], data.qvel[info.vadr], command)
            mujoco.mj_step(info.model, data)
            monitor.observe(data)
            if not monitor.ok:
                break
        checks = success_conditions(info, data, ground_forces(info, data))
        stance = stance_metrics(info, data)
        valid = all(checks.values()) and stance['posture_ok'] and monitor.ok
        hold = hold + CONTROL_DT if valid else 0.
        maxhold = max(maxhold, hold)
        maxheight = max(maxheight, float(data.qpos[2]))
        if step % 5 == 0:
            states.append(data.qpos.copy())
            trace.append(dict(time=float(data.time), height=float(data.qpos[2]),
                              clean_hold=hold, torso_up=float(data.xmat[info.torso_id].reshape(3, 3)[2, 2])))
        if on_step is not None:
            on_step(info, data, step, hold, monitor)
        if not monitor.ok or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            break
    full = step == 749 and monitor.ok and np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    result = dict(seed=seed, passed=bool(full and hold >= 2), sim_time=float(data.time),
                  terminal_hold=hold, max_clean_hold=maxhold, max_height=maxheight,
                  limits=monitor.report(), final_checks=checks, final_stance=stance,
                  contract=CONTRACT, controller='teacher' if actor is None else 'student',
                  teacher_queried=teacher is not None, teacher_fraction=teacher_fraction,
                  scripted_preparation_seconds=5. if prepare_target is not None else 0.,
                  state_edits_after_reset=False, max_action_inverse_error=max_inverse_error,
                  wall_seconds=time.monotonic() - start, trace=trace)
    return result, np.asarray(observations), np.asarray(actions), np.asarray(states)


def collect(args):
    from vendor_reference_probe import VendorGraph
    torch.set_num_threads(2)  # Match this script's CLI, despite the probe module's setting.
    graph = VendorGraph(args.asset_dir)
    if sha(graph.path) != EXPECTED_POLICY_SHA256:
        raise ValueError('Teacher differs from the student experiment provenance pin')
    actor = torch.jit.load(str(args.actor)).eval() if args.actor else None
    xs, ys, episode_ids, results = [], [], [], []
    for episode, seed in enumerate(args.seeds):
        result, x, y, states = rollout(seed, graph, actor, args.teacher_fraction)
        results.append(result)
        # Pure demonstration data includes only fully compliant successful runs.
        # DAgger also needs failure-state labels, up to the first audit fault.
        keep = actor is not None or result['passed']
        if keep:
            xs.append(x); ys.append(y); episode_ids.extend([episode] * len(x))
        np.savez_compressed(args.output / f'states_{seed}.npz', qpos=states)
        (args.output / f'{seed}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ['seed', 'passed', 'terminal_hold', 'max_height', 'wall_seconds']}
                         | {'kept': keep}), flush=True)
    if xs:
        np.savez_compressed(args.output / 'dataset.npz', observations=np.concatenate(xs),
                            actions=np.concatenate(ys), episode_ids=np.asarray(episode_ids),
                            seeds=np.asarray(args.seeds))
    (args.output / 'collection.json').write_text(json.dumps(dict(
        contract=CONTRACT, external_teacher_sha256=EXPECTED_POLICY_SHA256,
        policy_source='https://x2-aimdk.agibot.com/downloads/mc-x86-v1.0.0-20260522.zip',
        reference_timescale=1.2, seeds=args.seeds, passes=sum(r['passed'] for r in results),
        selection='Only successful complete teacher demonstrations; all student-state labels retained when aggregating',
        actor_sha256=sha(args.actor) if args.actor else None, teacher_fraction=args.teacher_fraction,
        samples=sum(map(len, xs)), original_vendor_assets_in_repository=False), indent=2) + '\n')


def fit(args):
    torch.manual_seed(args.seed)
    datasets = [np.load(path) for path in args.datasets]
    x = torch.tensor(np.concatenate([d['observations'] for d in datasets]), device=args.device)
    y = torch.tensor(np.concatenate([d['actions'] for d in datasets]), device=args.device)
    seeds = np.concatenate([d['seeds'][d['episode_ids']] for d in datasets])
    valid = torch.tensor(np.isin(seeds, args.validation_seeds), device=args.device)
    if x.shape[1:] != (107,) or y.shape != (len(x), 31) or not valid.any() or valid.all():
        raise ValueError('Invalid dataset contract or train/validation split')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or y.abs().max() > 1:
        raise ValueError('Nonfinite or unbounded demonstrations')
    mean = x[~valid].mean(0)
    std = x[~valid].std(0).clamp_min(.1)
    std[72:75] = 1.  # Contact switches cannot become arbitrarily large normalized inputs.
    model = Student(mean, std, args.drop_previous_action, args.phase_harmonics).to(args.device)
    if args.parent:
        parent = torch.load(args.parent, weights_only=True, map_location=args.device)
        parent['model'].setdefault('input_mask', model.input_mask)
        parent['model'].setdefault('phase_frequencies', model.phase_frequencies)
        model.load_state_dict(parent['model'])
    optimizer = torch.optim.Adam(model.net.parameters(), lr=args.learning_rate)
    train_ids = torch.where(~valid)[0]
    start = time.monotonic()
    best, best_state, best_step, rows = float('inf'), None, 0, []
    for step in range(1, args.steps + 1):
        ids = train_ids[torch.randint(len(train_ids), (1024,), device=args.device)]
        noisy = x[ids].clone()
        if args.noise:
            noise = torch.randn_like(noisy) * args.noise * model.std
            noise[:, 72:75] = 0.; noise[:, 106] = 0.
            noisy += noise
        prediction = model.raw_action(noisy) if args.unclamped_loss else model(noisy)
        loss = (prediction - y[ids]).square().mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == args.steps:
            with torch.no_grad():
                val = (model(x[valid]) - y[valid]).square().mean().item()
            row = dict(step=step, train_mse=loss.item(), validation_mse=val,
                       wall_seconds=time.monotonic() - start)
            rows.append(row); print(json.dumps(row), flush=True)
            if val < best:
                best = val; best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.cpu().load_state_dict(best_state)
    model.eval()
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, str(args.output / 'actor.pt'))
    with torch.inference_mode():
        assert torch.equal(model(x[:32].cpu()), scripted(x[:32].cpu()))
    provenance = dict(method='supervised_external_teacher_distillation', contract=CONTRACT,
                      initialization='random' if not args.parent else 'previous_student',
                      parent_sha256=sha(args.parent) if args.parent else None,
                      datasets=[dict(path=str(p), sha256=sha(p)) for p in args.datasets],
                      external_teacher_sha256=EXPECTED_POLICY_SHA256,
                      training_seeds=np.unique(seeds[~valid.cpu().numpy()]).tolist(),
                      validation_seeds=np.unique(seeds[valid.cpu().numpy()]).tolist(),
                      requested_validation_seeds=args.validation_seeds, gradient_steps=args.steps,
                      best_step=best_step, best_validation_mse=best, seed=args.seed,
                      samples=len(x), device=args.device, normalized_noise=args.noise,
                      previous_action_masked=bool(torch.all(model.input_mask[75:106] == 0)),
                      phase_harmonics=int(model.phase_frequencies.numel()),
                      loss_uses_unclamped_outputs=args.unclamped_loss,
                      wall_seconds=time.monotonic() - start, ppo_updates=0,
                      deployment_torch_threads=2,
                      actor_sha256=sha(args.output / 'actor.pt'),
                      note='Teacher-derived experimental policy. Physical evaluation is required; MSE is not recovery success.')
    torch.save(dict(model=best_state, provenance=provenance), args.output / 'checkpoint.pt')
    (args.output / 'fit.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (args.output / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')


def evaluate(args):
    actor = torch.jit.load(str(args.actor)).eval()
    results = []
    for seed in args.seeds:
        result, _, _, states = rollout(seed, actor=actor)
        result['actor_sha256'] = sha(args.actor)
        results.append(result)
        (args.output / f'{seed}.json').write_text(json.dumps(result, indent=2) + '\n')
        np.savez_compressed(args.output / f'states_{seed}.npz', qpos=states)
        print(json.dumps({k: result[k] for k in ['seed', 'passed', 'terminal_hold', 'max_height', 'limits']}), flush=True)
    (args.output / 'summary.json').write_text(json.dumps(dict(actor_sha256=sha(args.actor),
        passes=sum(r['passed'] for r in results), episodes=len(results), seeds=args.seeds,
        contract=CONTRACT, external_teacher_at_inference=False), indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['collect', 'fit', 'evaluate'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--asset-dir', type=Path)
    p.add_argument('--actor', type=Path)
    p.add_argument('--seeds', type=int, nargs='+', default=[9501])
    p.add_argument('--teacher-fraction', type=float, default=0.)
    p.add_argument('--datasets', type=Path, nargs='+')
    p.add_argument('--validation-seeds', type=int, nargs='+', default=[9513, 9514, 9515, 9516])
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--seed', type=int, default=12)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--noise', type=float, default=.002)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--parent', type=Path)
    p.add_argument('--drop-previous-action', action='store_true',
                   help='Prevent a copy-previous-target shortcut in supervised fitting')
    p.add_argument('--phase-harmonics', type=int, default=0)
    p.add_argument('--unclamped-loss', action='store_true')
    args = p.parse_args()
    if args.steps <= 0 or not 0 <= args.teacher_fraction <= 1 or args.noise < 0 or args.phase_harmonics < 0:
        p.error('Invalid numeric argument')
    if args.mode == 'collect' and args.asset_dir is None:
        p.error('Collection requires --asset-dir')
    if args.mode == 'fit' and not args.datasets:
        p.error('Fitting requires --datasets')
    if args.mode == 'evaluate' and args.actor is None:
        p.error('Evaluation requires --actor')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    globals()[args.mode](args)


if __name__ == '__main__':
    main()
