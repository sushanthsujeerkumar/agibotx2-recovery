# Start here — current physics-v2 project

The current reference controller is **an external AgiBot policy plus our trained
PPO ankle correction**, not a stand-up policy learned independently. It passed
5/5 fresh assessment trials and 24/25 total fresh trials. The unchanged external
teacher passed the same 24/25, so no success-rate benefit from the correction is
claimed. Our independent PPO and distilled student remain 0/5 supine.

Read [current results and limitations](LANDING_RESIDUAL_RESULTS.md). Earlier
experiments and the original ZIP are historical. Use the `physics-v2` Git branch.

## On this development PC

The environment, model meshes and local external assets are ready. From this folder:

```bash
./RUN_LANDING_RESIDUAL_DEMO.sh
```

This opens and records a fresh physical replay of seed 11301, then checks it
against saved evidence. Captions identify the external teacher and PPO correction.
The window closes after 15 simulated seconds. Ctrl+C stops a manual run.

## ROS recovery service

Terminal 1, from this folder:

```bash
./RUN_REFERENCE_ROS.sh
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState --once
ros2 topic echo /x2/recovery_status std_msgs/msg/String
```

The first request is accepted before simulator work; another request while busy
is rejected. The service publishes actual simulator joint states and completes
with `SUCCEEDED` only after the full 15-second limit/clean-stance audit. Faults or
timeouts produce `FAILED`. Ctrl+C stops the topic subscriber or ROS launch.
The controller may fail on other seeds; the recorded 25-episode batch includes
one wrist-speed failure.

For headless ROS execution:

```bash
./RUN_REFERENCE_ROS.sh render:=false realtime:=false
```

## Fresh Ubuntu 24.04 setup

Install ROS 2 Jazzy using the instructions linked in [README.md](README.md), then
clone the correct branch (or extract the new release archive):

```bash
git clone --branch physics-v2 https://github.com/sushanthsujeerkumar/hrs-x2-recovery.git
cd hrs-x2-recovery
bash scripts/setup.sh --reference
PYTHONPATH=src .venv/bin/python scripts/fetch_vendor_reference.py --output .cache/vendor_reference
./RUN_LANDING_RESIDUAL_DEMO.sh
```

The setup script installs locked Python dependencies and runs the tests. The
repository and archive both include the prepared model and all 39 meshes.
The separate fetch retrieves the original external policy/configuration locally
and verifies the pinned policy hash. **Those vendor assets are not included in
Git or the archive**, and their reuse/redistribution terms remain unverified.
Skip the fetch when that directory already contains the verified assets. For a
different local asset directory, set `X2_VENDOR_ASSETS_DIR=/absolute/path` before
running either launcher. Fetching uses network access; simulation itself is local.

## Repeat the evaluated comparison

Choose a fresh output directory:

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_landing_residual.py \
  --asset-dir .cache/vendor_reference \
  --checkpoint artifacts/experiments/landing_residual_ppo/update_004.pt \
  --baseline artifacts/experiments/landing_residual_ppo/update_000.pt \
  --seeds 11301 11302 11303 11304 11305 \
  --output artifacts/local_evaluation/new_paired_five
```

This evaluates both controllers on matching seeds, including every 1 ms physics
step. [The full report](LANDING_RESIDUAL_RESULTS.md) includes the training command,
reward specification, checkpoint hashes and the additional 20 evaluation seeds.
To repeat fresh ROS build/service/telemetry/timeout validation:

```bash
bash scripts/validate_reference_ros.sh
```

## Historical independent policy

The earlier independent PPO remains runnable and explicitly unsuccessful under
the stricter full-trajectory recovery checks:

```bash
./RUN_DEMO.sh --assess-stance
./EVALUATE.sh --assess-stance --output artifacts/local_evaluation/original_policy_check
```

`RUN_ROS.sh` retains that historical actor; use `RUN_REFERENCE_ROS.sh` for the
new reference-controller integration. Its checkpoints are not interchangeable
with the 78-input residual or its normalized-correction export.

## What to present

Present the working simulation/ROS system, the actual PPO attempts and saved
reward curves, the attributed external comparison, and the equal 24/25 paired
result. Do not claim that our independent policy learned full recovery, that PPO
improved the teacher's success rate, or that 24/25 proves broad robustness.
The repository preserves the development and failure history. No employer
submission or cloud deployment has been performed.
