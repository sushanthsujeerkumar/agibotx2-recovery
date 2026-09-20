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

## Targeted stability revision

The original configuration remains available as `--variant baseline` (the
default), including initial Gaussian standard deviation 0.6 and entropy
coefficient 0.01. Its checkpoint and logs are preserved in
`artifacts/runs/local`; they are not overwritten by the revision.

The first experiment learned to reach an upright, foot-supported posture, but
the inspected episode still moved too quickly to pass the two-second stability
check. Its exploration also became excessive: at checkpoint 2501 the per-joint
Gaussian standard deviations ranged from 1.62 to 13.0, and 29–30 of 31 raw
deterministic action means exceeded the executed [-1,1] range in inspected
states. Clamping such outputs makes many different network outputs produce
the same saturated joint target.

RSL-RL's Gaussian entropy is calculated on the unbounded sampled distribution,
before environment action clipping. Its default standard-deviation bounds are
1e-6 to 1e6. Increasing that entropy can therefore increase latent noise without
providing useful additional physical exploration. The PPO likelihoods themselves
remain consistent with the recorded raw samples; this is an exploration-design
problem rather than an incorrect likelihood implementation. PPO normalizes
advantages, so multiplying dense rewards by timestep does not by itself prove
that the entropy gradient dominates.

The **stability variant** starts a fresh policy with initial standard deviation
0.4, effective standard-deviation bounds [0.05,0.6], and entropy coefficient
0.0001. It selects environment reward version 2, which adds the separately
documented stability shaping. The original network topology, PPO update
settings, action mapping and success test remain the same.

```bash
python -m x2_recovery.train --variant stability --num-envs 512 --seed 0 --max-iterations 100000 --max-seconds 5400 --save-interval 250 --output artifacts/runs/local_stability
```

The same 512-environment, 90-minute configuration is available through
[`./TRAIN_STABILITY.sh`](../TRAIN_STABILITY.sh), which also sets CPU thread
limits and a quieter print interval. This stability experiment completed with5/5 original recoveries; its frozen evidence is in artifacts/submission/local_stability. A separate stance refinement is now running. The generic `runs/local`
examples elsewhere in this document describe independent baseline runs.

`--variant stability` passes `reward_version=2` in the environment constructor
kwargs. It is also valid to state this explicitly with
`--env-kwargs '{"reward_version":2}'`. A different reward version is rejected
for this variant. The fresh run avoids inheriting the original actor's very
large saturated means and does not alter its saved checkpoints. This is a
targeted correction, not a controlled single-variable ablation or a guarantee
of successful recovery.

Resume a stability run using its own checkpoint:

```bash
python -m x2_recovery.train --resume artifacts/runs/local_stability/latest.pt --num-envs 512 --max-iterations 100000 --max-seconds 5400 --save-interval 250 --output artifacts/runs/local_stability
```

The launcher equivalent is
`./TRAIN_STABILITY.sh --resume artifacts/runs/local_stability/latest.pt`.
The wall-clock budget applies to each invocation. Monitor the active corrected
run with `watch -n 300 cat artifacts/runs/local_stability/status.json`, and point
the local viewer at `--directory artifacts/runs/local_stability`.

On resume, the saved PPO variant and environment reward version take priority;
an explicitly conflicting reward version is rejected. Standard-deviation bounds
constrain effective sampling noise, not the Gaussian mean. Continue checking
actual movement, action saturation and the final deterministic success test.

Compare 128, 256 and 512 environments using fresh output directories. Compare
steady-state `steps_per_second`, memory consumption and actual behaviour;
changing batch size does not guarantee faster learning.

## Fresh stance refinement from a learned policy

The frozen checkpoint in
`artifacts/submission/stable_crossed_stance_009251` passed **5/5 episodes under
the original two-second stable-recovery criterion**, but its stance has inward-twisted, edge-supported feet with foot-foot self-contact. That criterion does not establish a neutral
stance or robustness. The frozen checkpoint, its evaluation and this limitation
remain preserved; refining stance is a separate experiment.

`--initialize-from` starts a **fresh fine-tuning run**. Unlike `--resume`, it
copies only the learned actor mean network and observation normalizer, plus
the critic network and its normalizer. It retains the new run's action noise,
optimizer, learning-rate schedule, counters, environment reset and seeded RNG
stream. It does not inherit the parent's large iteration count or consume the
parent's remaining training budget. The two options are mutually exclusive.

The critic is retained because the observation/action contract and recovery
reward remain largely shared. Its initial values may be biased for the new
stance reward, so it continues learning; preserving it avoids starting with an
uninformed value model while preserving a useful recovery policy.

The `stance` variant selects reward version 3 and uses initial Gaussian
standard deviation 0.10, effective bounds [0.03,0.20], entropy coefficient
0.0001, **fixed learning rate 0.00005**, and PPO clip 0.1. Fixed learning rate
prevents an adaptive schedule from quickly increasing it during fine-tuning.
Other PPO/network settings remain unchanged. The environment defines the new
stance shaping and training termination; final reports must distinguish the
original recovery result from the additional clean-stance assessment.

After the preceding local run completes, the bounded 45-minute refinement can
be started with:

```bash
python -m x2_recovery.train --variant stance --initialize-from artifacts/submission/stable_crossed_stance_009251/checkpoint.pt --num-envs 512 --seed 1 --max-iterations 100000 --max-seconds 2700 --save-interval 250 --output artifacts/runs/local_stance
```

The deterministic actor output is unchanged immediately after initialization;
the new Gaussian noise is initialized independently. Architecture, observation
group ordering and observation/action dimensions must match the parent or the
command fails before learning. Training still starts from supine resets; no
standing-start curriculum or model assistance is introduced by initialization.

Both `config.json` and every checkpoint record the parent path, SHA-256,
iteration/step count, reward version and the exact state categories copied or
reset. Resume restores that provenance along with reward version 3:

```bash
python -m x2_recovery.train --resume artifacts/runs/local_stance/latest.pt --num-envs 512 --max-iterations 100000 --max-seconds 2700 --save-interval 250 --output artifacts/runs/local_stance
```

The time limit applies to each invocation; a resume is an additional budget,
not part of the original 45 minutes. Only the full trusted training checkpoint
can initialize a new learner; a TorchScript inference export lacks the critic,
configuration and training metadata. The frozen parent's files are not edited.

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
