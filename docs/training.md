# Training and monitoring

The training command uses MuJoCo Warp through the project's vectorized
environment, and RSL-RL PPO. It writes local TensorBoard files, JSON progress,
and checkpoints. It does not require W&B, a cloud account, or external logging.

Run commands from the repository root, with the project's Python environment
activated. The environment is headless; the separate simulator viewer can load
the newest exported actor without interrupting the learner.

## Local validation and a bounded run

First perform a short real training run. The initial GPU kernel compilation can
take longer than subsequent runs and should not be counted as steady-state
training throughput.

```bash
python -m x2_recovery.train --num-envs 128 --max-iterations 10 --output runs/smoke
```

Once physics and policy updates pass validation, start a one-hour budget. The
time limit is checked after each complete rollout and PPO update, so it can
overrun by one iteration. A time-limited run has status `PAUSED` and can resume.

```bash
python -m x2_recovery.train --num-envs 256 --max-iterations 3000 --max-seconds 3600 --output runs/local
```

`--num-envs`, `--device`, `--seed`, `--steps-per-env`, `--save-interval`, and
`--torch-threads` are configurable. `--env-kwargs` accepts a JSON object of
additional environment constructor arguments; record any changes and use the
same environment settings for playback and evaluation.

The default PPO configuration uses 24 steps per environment, hidden layers
256/128/128 with ELU, observation normalization, initial action standard
deviation 0.6, five update epochs, four minibatches, adaptive learning rate
1e-3, target KL 0.01, gamma 0.99, GAE lambda 0.95, and entropy coefficient 0.01.
The actor standard deviation uses a logarithmic parameterization to remain
positive. `config.json` records the actual configuration and package versions.

Compare 128, 256 and 512 environments using fresh output directories. Compare
steady-state `steps_per_second`, memory consumption and actual behaviour;
changing batch size does not guarantee faster learning.

## Observe progress without repeatedly using an AI session

```bash
cat runs/local/status.json
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

TensorBoard is available at <http://127.0.0.1:6006>. Open it in a browser to see
reward, losses, throughput and environment metrics. The JSON status file is
replaced atomically once per iteration. A timer or an ordinary `watch` command
can read it every few minutes without any AI inference:

```bash
watch -n 300 cat runs/local/status.json
```

`progress.jsonl` contains one record per complete training iteration, including
environment steps, wall-clock time, training time, mean reward over the latest
100 completed episodes, throughput, losses and the environment's metrics.
Success is `null` before completed episodes provide a success measurement; it
is not silently reported as zero. Training success uses exploratory actions
and is separate from the final deterministic five-episode evaluation.

The training process continues independently of how often these files are read.
Inspect the simulator and metrics after the initial smoke test, then at useful
intervals such as 10–15 minutes. Additional checking does not speed training.

## Checkpoints, stopping, and resuming

The run saves periodically, at completion, and at a graceful stop. Upstream
RSL-RL saves at iteration zero and then at the configured interval; filenames
here use the number of fully completed updates instead. Thus a save interval
of 50 first produces `model_000001.pt`, then `model_000051.pt`, and so on.
`latest.pt` always points to the newest complete checkpoint.

Press **Ctrl+C once** or send `SIGTERM` to request a stop at the next complete
PPO update. Avoid force-killing the process if a current checkpoint is needed.

```bash
python -m x2_recovery.train --num-envs 256 --max-iterations 6000 --max-seconds 3600 --output runs/local --resume runs/local/latest.pt
```

`--max-iterations` is the cumulative target, including completed iterations. A
resume restores actor and critic parameters, observation normalizers, optimizer
state including the adaptive learning rate, random-number-generator state,
and accumulated timing/step counters. It reads the network and algorithm
configuration from the checkpoint. If the environment implements
`state_dict()`/`load_state_dict()`, its supplied state is restored too;
otherwise simulator episodes restart and an available common step counter is
restored. A resume is not claimed to be a bitwise reproduction of uninterrupted
GPU simulation. Rollout and rolling episode-logging buffers restart.

Checkpoints are PyTorch pickle files: load only the ones produced by this
project or another trusted source. CUDA training is never silently replaced
with CPU training; `--device cpu` must be selected explicitly if supported by
the environment.

## Local playback and deployment artifact

Each checkpoint also has a CPU TorchScript actor, such as
`model_000051_actor.pt`. `actor_latest.pt` points to the newest complete export.
`actor_latest.json` identifies its checkpoint and training step. A viewer should
read this JSON, load its explicit `actor` path, and switch only between episodes
so a demonstration uses one checkpoint throughout.

The actor embeds observation normalization and outputs deterministic action
means. Its input is a float32 tensor of shape `[batch, num_actor_observations]`
and output is `[batch, num_actions]`. The caller must implement the exact
training observation order, scaling, previous-action bookkeeping, action
clipping, control timing and joint mapping. Loading the network alone does not
establish simulator or ROS equivalence.

```python
import torch

policy = torch.jit.load("runs/local/actor_latest.pt", map_location="cpu").eval()
with torch.inference_mode():
    action = policy(observation_tensor)
```

To export a trusted checkpoint separately:

```python
from x2_recovery.train import export_actor

export_actor("runs/local/latest.pt", "runs/local/policy.pt")
```

The exported actor requires PyTorch for inference, but importing RSL-RL,
TensorDict, mjlab or the training runner is unnecessary in the ROS node.

## Environment contract

`X2RecoveryEnv(num_envs=256, device="cuda:0", seed=0)` must reset itself before
the runner is constructed. It provides `num_envs`, `num_actions`, `device`,
`max_episode_length`, `episode_length_buf`, and a JSON-serializable `cfg`.

- `get_observations()` returns a TensorDict containing `actor` and `critic`
  tensors, with batch size `[num_envs]`.
- `step(actions)` returns `(observations, reward, done, extras)`. Reward and
  done have shape `[num_envs]`; completed environments auto-reset.
- `extras["time_outs"]` identifies time-limit truncations for value bootstrap.
- `extras["log"]` supplies scalar or tensor metrics. In particular,
  `recovery/success_rate` should contain success indicators for episodes that
  just finished, measured before auto-reset. Omit it when no episode finished.
- `close()` releases resources when available.

The implementation supports the RSL-RL 5.4/5.5 API. The project dependency lock
selects RSL-RL 5.4.2 with mjlab 1.6.0. The runner preserves the complete model
configuration despite RSL-RL 5.4 consuming class-name keys during construction.
Dependency versions in each run's `config.json` record the actual installed
stack used for that experiment.
