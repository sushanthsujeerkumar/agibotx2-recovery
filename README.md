# AgiBot X2 ground recovery

A reproducible X2 Ultra ground-recovery experiment using **MuJoCo, MuJoCo Warp
through mjlab, PPO, and ROS 2 Jazzy**. Every assessed episode starts from a
physically settled supine pose. A shared CPU runtime supplies the local viewer,
deterministic evaluation and ROS-controlled episodes.

**Current result:** model validation, the PPO runner's CPU contract checks and
the real-simulator ROS integration check have passed. GPU/CPU consistency and
512-environment PPO smoke tests passed. The one-hour local learning experiment
is now running; final five-episode policy evaluation is pending. **No
successful learned recovery is claimed.** The scripted controller is explicitly
an untrained baseline, not evidence of a learned policy.

## Setup

The tested host is Ubuntu 24.04, an Intel i5-10600K (6 cores/12 threads), about
23 GiB usable RAM, and an NVIDIA GeForce RTX 5060 with 8 GB VRAM. The verified
driver is 595.84. No cloud compute has been used.

| Component | Installed version |
|---|---|
| Python / ROS | Python 3.12 / ROS 2 Jazzy |
| mjlab | 1.6.0 |
| MuJoCo / MuJoCo Warp | 3.11.0 / 3.11.0 |
| Warp | 1.17.0 |
| PyTorch / torchvision | 2.11.0 / 0.26.0 |
| PyTorch CUDA runtime | 13.0 |
| RSL-RL / TensorDict | 5.4.2 / 0.14.2 |
| Dependency installer used | uv 0.12.17 |

Install [ROS 2 Jazzy for Ubuntu 24.04](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
and its development tools if they are absent. The ROS commands below expect
`/opt/ros/jazzy/setup.bash` and `colcon`. A compatible NVIDIA driver is required
for GPU training. A graphical desktop with OpenGL is needed for a visible viewer.

```bash
git clone https://github.com/sushanthsujeerkumar/hrs-x2-recovery.git
cd hrs-x2-recovery
```

If `uv` is absent, install it using the [official installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/0.12.17/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Install the locked project environment with Ubuntu's Python 3.12 so its ABI
matches ROS Jazzy. CUDA libraries and model dependencies make the first download
substantial; subsequent installs reuse the package cache.

```bash
uv sync --frozen --python /usr/bin/python3
source .venv/bin/activate
python -c 'import torch, mujoco; print("MuJoCo", mujoco.__version__); print("CUDA", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")'
python -m pytest -q tests
```

The prepared model and meshes are included. Rebuilding them is optional:

```bash
python scripts/prepare_model.py --validate
```

This fetches the pinned official source and regenerates the MJCF and provenance
metadata. It does not retrain the policy.

## See the robot and train locally

From the repository root with `.venv` activated, first view one scripted attempt:

```bash
python -m x2_recovery.evaluate --controller scripted --episodes 1 --seed 1001 --render --output artifacts/scripted_preview
```

Start with a short GPU training smoke test, then a bounded experiment:

```bash
python -m x2_recovery.train --num-envs 128 --max-iterations 10 --output artifacts/runs/smoke
python -m x2_recovery.train --num-envs 512 --max-iterations 100000 --max-seconds 3600 --save-interval 250 --output artifacts/runs/local
```

The learner runs headlessly. In a second terminal, activate `.venv` and open the
local simulator to replay newly saved policies:

```bash
python -m x2_recovery.watch --directory artifacts/runs/local --minutes 120
```

The viewer uses the scripted baseline until the first trained actor exists,
then switches checkpoints between episodes. Its terminal identifies the
controller, checkpoint iteration and outcome. It shows checkpoint evaluation;
it is not rendering one of the learner's exploratory environments. Close its
window to stop viewing while training continues.

Press **Ctrl+C once in the training terminal** to save and pause after the
current PPO update. The one-hour budget also pauses at an update boundary.
Resume using a cumulative iteration target:

```bash
python -m x2_recovery.train --num-envs 512 --max-iterations 100000 --max-seconds 3600 --output artifacts/runs/local --resume artifacts/runs/local/latest.pt
```

Checkpoints preserve the actor, critic, observation normalization, optimizer,
RNG state and counters. Simulator episodes restart on resume; bitwise
continuation of an uninterrupted physics trajectory is not claimed.

Progress is written without needing an AI session to poll continuously:

```bash
watch -n 300 cat artifacts/runs/local/status.json
tensorboard --logdir artifacts/runs --host 127.0.0.1 --port 6006
python -m x2_recovery.plot artifacts/runs/local
```

Open [local TensorBoard](http://127.0.0.1:6006). The plot command creates
`training_curve.png`. More options and checkpoint details are in
[training.md](docs/training.md).

## Model, environment and learning method

The model is **AgiBot X2 Ultra v1.3.0**, using the official
`x2_ultra_simple_collision.urdf` at commit
`60c5de582c523cd188f563819e62d34cfdc3d2d0`. It has 31 actuated joints, a free
pelvis and total mass 41.966521 kg. URDF inertias, transforms, joint ranges and
rated effort/velocity values are retained. Primitive collision shapes replace
mesh collisions; PD gains and contact parameters are simulation assumptions.
The source URDF, license, hashes and full modifications are documented in
[model.md](docs/model.md) and [model metadata](assets/x2/model_metadata.json).

The controller runs at 50 Hz over ten 2 ms physics steps. The 106 observations
contain relative joint positions, joint velocities, pelvis-frame gravity and
linear velocity, free-joint angular velocity, pelvis height, three touch
indicators and previous actions. The 31 actions select joint targets across
the complete allowed ranges. A torque-limited PD controller applies them;
simulated velocities are never artificially clipped.

Resets use a one-second dynamically settled supine state with small position
and joint perturbations. Training samples a bank of 16 such states. Episodes
end after 15 simulation seconds, stable recovery, or an invalid physical state.
There are no standing-start episodes, reference-motion rewards or external
assistance. PPO uses 24 steps per environment, five epochs, four minibatches,
256/128/128 ELU networks, observation normalization, adaptive learning rate
1e-3, gamma 0.99, GAE lambda 0.95 and entropy coefficient 0.01.

The dense reward combines normalized pelvis height and upright orientation
(weight 2), upright orientation alone (1), height progress (0.5), both-foot
support (1), and currently stable standing (4). It penalizes normalized torque
(0.03), action changes (0.03), excessive joint speed (0.1) and joint-limit
proximity (1). These terms are multiplied by the 0.02 s control period. A
completed two-second recovery adds 10; an invalid state subtracts 1.
Exact equations, observation ordering, success thresholds and limitations are
in [environment.md](docs/environment.md).

## Five-episode evaluation

Use an exported **TorchScript actor**, not the full optimizer checkpoint:

```bash
python -m x2_recovery.evaluate --controller policy --checkpoint artifacts/runs/local/actor_latest.pt --episodes 5 --seed 1001 --output artifacts/evaluation/policy
```

This evaluates seeds **1001–1005** with deterministic actions and writes a
`summary.json` plus five episode JSON files with sampled trajectories. Add
`--render` to watch, or `--video` to record MP4 files. For headless recording on
an NVIDIA host, select EGL before launching:

```bash
MUJOCO_GL=egl python -m x2_recovery.evaluate --controller policy --checkpoint artifacts/runs/local/actor_latest.pt --episodes 5 --seed 1001 --video --output artifacts/evaluation/policy_video
```

For a separately labelled scripted comparison:

```bash
python -m x2_recovery.evaluate --controller scripted --episodes 5 --seed 1001 --output artifacts/evaluation/scripted
```

Success requires **all conditions continuously for two seconds**: torso tilt
at most 15 degrees, pelvis height at least 0.6048 m, both feet carrying more
than 2 N normal ground force each, all other body-ground support at most 2 N
in total, base linear speed below 0.15 m/s and angular speed below 0.3 rad/s.
Final evaluation inspects actual floor contacts and ignores self-contact.
Merely reaching standing height does not count.

The final policy's success count, episode reasons and measured failure analysis
must come from `artifacts/evaluation/policy/summary.json`; those results are
pending. An absent report is not a zero-success result. Training checkpoints,
curves and evaluation evidence will be preserved for the submitted run; bulky
intermediate runs are ignored by Git. No success percentage is inferred from
reward improvement alone.

Likely experiment limitations are exploration from a fully grounded pose,
dense-reward plateaus in sitting/kneeling, contact-model approximations, and
transfer between GPU and CPU solvers. After measured failures are available,
the next changes should target the observed bottleneck: validate a short
recovery reference, improve stage-specific rewards, expand reset diversity,
and repeat training with independent seeds. None of these improvements is
represented as already implemented.

## ROS 2 demonstration

The **recovery node** queues one simulator episode and returns the service
acceptance response before starting execution. It rejects concurrent requests,
publishes real simulator state, and supervises a separate simulator subprocess
with a wall-clock timeout. The **telemetry node** subscribes to the two topics
and logs the current status plus one actual joint position.

One command builds with `colcon` and launches both nodes:

```bash
bash scripts/ros_launch.sh controller:=scripted render:=true max_sim_duration_s:=15.0 timeout_s:=60.0
```

For the trained actor, replace that launch command with:

```bash
bash scripts/ros_launch.sh controller:=policy checkpoint:="$PWD/artifacts/runs/local/actor_latest.pt" render:=true max_sim_duration_s:=15.0 timeout_s:=60.0
```

In another terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
ros2 topic echo /x2/recovery_status std_msgs/msg/String
```

The first request returns `success=True`; a second request while queued or
running returns `success=False`. Stop the topic echo with Ctrl+C and inspect
one live state message during an episode:

```bash
ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState --once
```

Status values are `IDLE`, `RUNNING`, `SUCCEEDED` and `FAILED`. Joint names,
positions and timestamps come from MuJoCo. Timestamps are simulation time
since the current episode reset, while the watchdog uses monotonic wall time.
No synthetic state or ROS-only demonstration is substituted for simulation.

The equivalent manual build sequence is:

```bash
source /opt/ros/jazzy/setup.bash
export PYTHONPATH="$PWD/src:$PWD/.venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd ros2_ws
colcon build --symlink-install --packages-select x2_recovery_ros
source install/setup.bash
ros2 launch x2_recovery_ros recovery.launch.py controller:=scripted render:=false
```

The ROS default simulation limit is 20 s; the examples above explicitly use
the 15 s evaluation horizon. The default wall timeout is 60 s and includes
worker/model startup. Press Ctrl+C in the launch terminal to stop both nodes.

## Verification evidence

From the repository root:

```bash
python -m pytest -q tests
bash scripts/validate_ros.sh
```

The ROS script builds into a fresh directory, runs tests, starts the real
simulator, checks acceptance and busy rejection, reads joint telemetry, and
forces an unsuccessful attempt to end through the wall-clock timeout. It saves
commands and logs under `ros2_ws/validation/<run-id>/`.

The initial recorded integration run
[`20260919T231348Z-15524`](ros2_ws/validation/20260919T231348Z-15524/integration.json)
passed with 516 actual joint frames, changed joint positions, a busy rejection,
and `FAILED` after the configured 12 s deadline. This validates integration,
not recovery performance. See [ROS validation](docs/ros_validation.md) for the
test scope and detailed evidence, and
[model validation](assets/x2/model_validation.json) for contact and actuator
checks. The PPO runner additionally passed a bounded CPU check covering actual
updates, optimizer/normalizer resume, completed-step counts and TorchScript
inference parity. Main GPU experiment results remain pending as stated above.
