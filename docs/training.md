# Training and reference provenance

The submitted actor has two parts: a fixed time-indexed motion prior and PPO feedback. It takes 106 physical observations plus episode time divided by 15, and returns 31 normalized joint targets:

```text
action = clip(prior(observation) + 0.01 * tanh(feedback(observation)), -1, 1)
```

The prior uses only the phase input. It linearly interpolates a 750-by-31 table; at each time knot that table is the mean action from eight successful, physically simulated recovery examples. This is a supervised least-squares fit, not RL. The feedback network uses the physical state and phase. Its correction is bounded to ±0.01 in normalized action units, not radians.

## Where the reference came from

The project's earlier standing/crouch work provided a learned standing controller. Physically simulated searches supplied joint-target keyframes for the floor-to-sitting transition and the bridge into that controller. The chosen hybrid reference passed ten fresh supine trials. Its exact keyframes, timings, standing actor and selection evidence are included under `artifacts/submission/final_recovery/reference/`.

`ReferenceController` in `full_recovery.py` reproduces that hybrid reference for data collection. Collection seeds were 17001–17008. The fitted prior passed 10/10 different seeds, 17101–17110, before full-episode PPO. This reference already solves the narrow recovery task; the PPO stage adjusts it rather than learning the whole movement unaided. The reference's metadata describes the state of that earlier preparation stage, so its “training pending” label is historical.

The standing actor is a frozen dependency from an earlier project-trained phase-prior/PPO experiment. Reproducing the final stage uses the included dependency; it does not rerun every earlier search and standing-policy experiment. No externally trained AgiBot controller is used in this final chain. Earlier vendor comparisons remain separately attributed in the historical results.

## PPO settings

| Setting | Value |
|---|---|
| Algorithm/backend | RSL-RL PPO; mjlab/Warp GPU simulation |
| Environments / rollout | 256 / 24 steps each, 6,144 transitions per update |
| Actor and critic | Separate ELU MLPs, 256 / 128 / 128 |
| Observation normalization | Actor frozen at its initialized identity statistics; critic running statistics |
| Learning rate / schedule | 0.0001 / fixed |
| Discount / GAE lambda | 0.99 / 0.95 |
| PPO clip / value coefficient | 0.05 / 1.0, clipped value loss |
| Epochs / minibatches | 5 / 4 |
| Entropy coefficient | 0 |
| Gaussian standard deviation | Starts at 0.1, bounded to [0.02, 0.25] |
| Gradient norm limit | 1.0 |
| Seed | 17201 |
| Checkpoint interval | 25 updates, plus final/interrupt checkpoint |

The Gaussian is over raw feedback outputs; the bounded `tanh` correction is applied inside the environment. At inference the deterministic mean is used. The exported TorchScript actor contains the feedback network, normalization and frozen prior table, so deployment needs only `actor.pt`.

The final local run resumed update 51 (313,344 transitions), completed another 1,497 updates / 9,197,568 transitions in about 30 minutes, and finished at update 1,548 / 9,510,912 cumulative transitions. Cumulative recorded wall time is 1,861.44 seconds, including the short starting run; the earlier reference work is additional. The final logged mean episode return is 80.6365. The [reward plot](../artifacts/submission/final_recovery/training_curve.png) uses the raw progress log and a labelled rolling mean. Return is not a recovery success rate.

The local machine used an Intel i5-10600K, approximately 23 GiB RAM and an NVIDIA RTX 5060 with 8 GB VRAM. Dependencies are pinned in `uv.lock`; training used Python 3.12, MuJoCo 3.11.0, mjlab 1.6.0 and PyTorch 2.11.0. CPU and GPU solvers need not follow identical trajectories.

## Reproduction commands

Run all commands from the repository root after setup. Final-stage reproduction from the saved starting checkpoint:

```bash
./TRAIN_FULL_RECOVERY.sh --headless
```

To regenerate the supervised prior from the included hybrid reference, without overwriting submission evidence:

```bash
.venv/bin/python scripts/prepare_full_recovery.py --output artifacts/runs/reprepared
.venv/bin/python scripts/train_full_recovery.py --prepared artifacts/runs/reprepared \
  --output artifacts/runs/refit_training --seconds 1800 --num-envs 256 --headless --evaluate-final
```

To continue from the final checkpoint:

```bash
.venv/bin/python scripts/train_full_recovery.py \
  --resume artifacts/submission/final_recovery/checkpoint.pt \
  --output artifacts/runs/continued --seconds 1800 --num-envs 256 --headless --evaluate-final
```

Resume restores policy, optimizer, counters and random-number state; environments restart in fresh supine episodes. It does not restore an in-flight physics rollout. The checkpoint embeds its prior, so resume does not depend on the original development directory.

To export a checkpoint and evaluate it:

```bash
.venv/bin/python -c "from x2_recovery.train import export_actor; export_actor('artifacts/submission/final_recovery/checkpoint.pt', 'artifacts/local_evaluation/reexported_actor.pt')"
.venv/bin/python scripts/evaluate_full_recovery.py \
  --actor artifacts/local_evaluation/reexported_actor.pt \
  --episodes 5 --seed 30001 --output artifacts/local_evaluation/reexported_5.json
```

For video recording, install the locked optional media dependencies (the historical `reference` extra includes them):

```bash
bash scripts/setup.sh --reference
MUJOCO_GL=egl .venv/bin/python scripts/demo_full_recovery.py \
  --video artifacts/local_demo/recovery.mp4 --output artifacts/local_demo/recovery.json
```

The recording uses separate renderer data. The supplied clip is an actual policy episode, not an animation or a training-environment replay. Normal evaluation requires no media packages.
