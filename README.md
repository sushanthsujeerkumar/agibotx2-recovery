# AgiBot X2 ground recovery

A reproducible X2 Ultra ground-recovery experiment using **MuJoCo, MuJoCo Warp
through mjlab, PPO, and ROS 2 Jazzy**. Every assessed episode starts from a
physically settled supine pose. A shared CPU runtime supplies the local viewer,
deterministic evaluation and ROS-controlled episodes.

**Final local result:** the selected learned policy achieved **5/5 recoveries** under the documented two-second stability check, in **4.28–8.86 seconds**, and completed a real recovery through ROS 2. The posture remains undesirable: the feet turn inward, stand partly on their edges and brace against each other. It achieved **0/5 on the additional clean-stance check**. A separate 45-minute refinement retained 5/5 first recoveries but also scored 0/5 clean stance and lost stability again in 2/5 episodes during the extended 15-second audit. The more consistent 90-minute policy is retained as the default. No neutral-stance or hardware-readiness claim is made.

Start with [START_HERE.md](START_HERE.md). See [RESULTS.md](RESULTS.md) for experiment comparison and remaining limitations, [requirement evidence](docs/requirements.md) for the assessment checklist, and [selected policy provenance](artifacts/selected_policy.json) for hashes. All three local experiments are preserved. Main training totaled **2 h 39 m 12 s**, excluding setup, compilation, smoke tests and evaluation; no cloud was used.

## Setup

The tested host is Ubuntu 24.04, an Intel i 5-10600 K (6 cores/12 threads), about
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

To reproduce training, start with a short GPU smoke test and the stability
revision (use a fresh output directory if that run already exists):

```bash
python -m x2_recovery.train --num-envs 128 --max-iterations 10 --output artifacts/runs/smoke
python -m x2_recovery.train --variant stability --num-envs 512 --seed 0 --max-iterations 100000 --max-seconds 5400 --save-interval 250 --output artifacts/runs/local_stability
```

[`./TRAIN_STABILITY.sh`](TRAIN_STABILITY.sh) is the ready-to-run launcher for
the same corrected experiment, including CPU thread limits and quieter logging.
Use the launcher or the full training command; an existing run requires resume.

The initial baseline remains reproducible with its original settings and a
separate output directory. The recorded baseline stopped at iteration 2803:

```bash
python -m x2_recovery.train --variant baseline --num-envs 512 --seed 0 --max-iterations 2803 --save-interval 250 --output artifacts/runs/baseline_reproduction
```

This reproduces the configuration and update budget, not bitwise GPU results
or an identical wall-clock duration. The baseline's frozen evidence is in
[`artifacts/submission/local_initial`](artifacts/submission/local_initial/manifest.json).

The completed posture-refinement experiment initialized from the preserved learned policy
without reusing its optimizer, action noise or counters:

```bash
python -m x2_recovery.train --variant stance --initialize-from artifacts/submission/stable_crossed_stance_009251/checkpoint.pt --num-envs 512 --seed 1 --max-iterations 100000 --max-seconds 2700 --save-interval 250 --output artifacts/runs/local_stance
```

`./TRAIN_STANCE.sh` is the convenience launcher for this 512-environment,
45-minute experiment. The stance variant selects reward version 3, initial
noise 0.10 bounded to [0.03,0.20], fixed learning rate 0.00005, PPO clip 0.1 and
entropy coefficient 0.0001. Parent network/normalizer weights and provenance
are retained; the baseline and successful recovery artifacts are preserved.

The learner runs headlessly. In a second terminal, activate `.venv` and open the
local simulator to replay newly saved policies:

```bash
python -m x2_recovery.watch --directory artifacts/runs/local_stability --minutes 120
```

For the stance refinement, use its separate directory and assessment mode:

```bash
python -m x2_recovery.watch --directory artifacts/runs/local_stance --assess-stance --minutes 60
```

This mode continues after the original recovery pass to measure the additional
clean-stance hold; it does not retroactively change the original recovery test.

The viewer uses the scripted baseline until the first trained actor exists,
then switches checkpoints between episodes. Its terminal identifies the
controller, checkpoint iteration and outcome. It shows checkpoint evaluation;
it is not rendering one of the learner's exploratory environments. Close its
window to stop viewing while training continues.

Press **Ctrl+C once in the training terminal** to save and pause after the
current PPO update. The 90-minute budget also pauses at an update boundary.
Resume using a cumulative iteration target:

```bash
python -m x2_recovery.train --num-envs 512 --max-iterations 100000 --max-seconds 5400 --output artifacts/runs/local_stability --resume artifacts/runs/local_stability/latest.pt
```

Checkpoints preserve the actor, critic, observation normalization, optimizer,
RNG state and counters. Simulator episodes restart on resume; bitwise
continuation of an uninterrupted physics trajectory is not claimed. Resume
restores the checkpoint's variant and reward version. `--max-seconds` is the
budget for each invocation, including a resumed invocation.

Progress is written without needing an AI session to poll continuously:

```bash
watch -n 300 cat artifacts/runs/local_stability/status.json
tensorboard --logdir artifacts/runs --host 127.0.0.1 --port 6006
python -m x2_recovery.plot artifacts/runs/local_stability
```

Open [local TensorBoard](http://127.0.0.1:6006). The plot command creates
`training_curve.png`. More options and checkpoint details are in
[training.md](docs/training.md).

## Model, environment and learning method

The model is **AgiBot X2 Ultra v 1.3.0**, using the official
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
1e-3, gamma 0.99 and GAE lambda 0.95. The baseline uses entropy coefficient
0.01 and initial Gaussian standard deviation 0.6. The fresh stability variant
uses entropy coefficient 0.0001, initial standard deviation 0.4 and effective
standard-deviation bounds [0.05,0.6]. This addresses the baseline's observed
growth of raw action noise and saturation at joint targets.

The dense reward combines normalized pelvis height and upright orientation
(weight 2), upright orientation alone (1), height progress (0.5), both-foot
support (1), and currently stable standing (4). It penalizes normalized torque
(0.03), action changes (0.03), excessive joint speed (0.1) and joint-limit
proximity (1). These terms are multiplied by the 0.02 s control period. A
completed two-second recovery adds 10; an invalid state subtracts 1.
Reward version 2 adds a smooth low-base-velocity reward when upright/elevated
and supported by both feet only, plus a mild upright posture penalty. This is
the targeted stability revision; the original reward remains version 1.
Version 3 adds near-standing rewards for signed foot width, heading and flat
soles, plus a targeted hip-yaw penalty. Its training episode ends successfully
only after the additional posture hold, while original recovery is logged
separately. CPU clean-stance evaluation also checks actual foot-to-foot force.
Exact equations, observation ordering, success thresholds and limitations are
in [environment.md](docs/environment.md).

## Five-episode evaluation

Use an exported **TorchScript actor**, not the full optimizer checkpoint.
Recheck the already-frozen initial policy:

```bash
python -m x2_recovery.evaluate --controller policy --checkpoint artifacts/submission/local_initial/actor.pt --episodes 5 --seed 1001 --output artifacts/evaluation/local_initial_recheck
```

Recheck the completed stability run's frozen actor separately:

```bash
python -m x2_recovery.evaluate --controller policy --checkpoint artifacts/submission/local_stability/actor.pt --episodes 5 --seed 1001 --output artifacts/evaluation/local_stability_recheck
```

This evaluates seeds **1001–1005** with deterministic actions and writes a
`summary.json` plus five episode JSON files with sampled trajectories. Add
`--render` to watch, or `--video` to record MP4 files. For headless recording on
an NVIDIA host, select EGL before launching:

```bash
MUJOCO_GL=egl python -m x2_recovery.evaluate --controller policy --checkpoint artifacts/submission/local_stability/actor.pt --episodes 5 --seed 1001 --video --output artifacts/evaluation/local_stability_video
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

### Additional stance assessment

Watch the frozen inward-stance candidate with the extra checks enabled:

```bash
python -m x2_recovery.evaluate --controller policy --checkpoint artifacts/submission/stable_crossed_stance_009251/actor.pt --episodes 1 --seed 1001 --render --assess-stance --output artifacts/evaluation/stance_parent_preview
```

For five episodes after stance refinement:

```bash
python -m x2_recovery.evaluate --controller policy --checkpoint artifacts/runs/local_stance/actor_latest.pt --episodes 5 --seed 1001 --assess-stance --output artifacts/evaluation/local_stance
```

Clean stance requires all original standing conditions plus the following for
two continuous seconds: signed foot-centre width **0.16–0.36 m** in a horizontal
pelvis-heading frame, each foot heading within **25°** of pelvis heading,
absolute hip yaw at most **0.45 rad**, each sole tilt at most **20°**, and total
actual foot-to-foot normal force at most **2 N**. These are added project
criteria, not manufacturer specifications. This assessment continues beyond
an original recovery pass, until clean stance, invalid state or the 15 s limit.
Reports retain original `successes` and add `clean_stance_successes`; a pass
under the former does not imply a pass under the latter.

### Recorded initial result and failure analysis

The frozen `local_initial` policy completed 2,803 PPO updates and 34,443,264
environment steps using 512 environments. The run took 1,450.8 wall seconds
(about 24.2 minutes, excluding initial environment setup). Its final
rolling mean episode return was 50.83. The
[training curve](artifacts/submission/local_initial/training_curve.png),
[configuration](artifacts/submission/local_initial/config.json),
[training status](artifacts/submission/local_initial/status.json),
[checkpoint](artifacts/submission/local_initial/checkpoint.pt) and
[exported actor](artifacts/submission/local_initial/actor.pt) are preserved.

The deterministic evaluation achieved **0/5 recoveries**. Every episode timed
out after 15 simulated seconds, with no invalid-state termination:

| Seed | Outcome | Maximum pelvis height | Final both-foot support |
|---|---|---:|---|
| 1001 | FAILED: timeout | 0.855 m | No |
| 1002 | FAILED: timeout | 0.770 m | Yes |
| 1003 | FAILED: timeout | 0.803 m | Yes |
| 1004 | FAILED: timeout | 0.745 m | No |
| 1005 | FAILED: timeout | 0.869 m | Yes |

All five episodes ended with an upright, elevated torso and no other body-ground
support, but failed both final base-speed checks. The recorded trajectories
first showed simultaneous upright/height checks at approximately 0.62–0.72 s;
the robot then continued hopping, travelling or rotating rather than holding
a stable posture. Reaching height was therefore correctly rejected as success.
See the [five-episode report](artifacts/submission/local_initial/evaluation/summary.json)
and the [first recorded episode](artifacts/submission/local_initial/evaluation/episode_1.mp4).

The original action noise grew to mean standard deviation 11.21 by the end of
training. Inspection of checkpoint 2501 also found 29–30 of 31 deterministic
action means outside the [-1,1] execution range. Unbounded Gaussian entropy can
reward increasingly noisy raw actions even though the simulator clips them to
the same joint targets. The baseline reward also lacks a smooth incentive to
slow the base before reaching the strict binary stability threshold.

### Completed stability result and posture limitation

The fresh `local_stability` run addressed these observations with bounded
exploration noise, lower entropy weight, a smooth velocity-based reward and a
mild posture penalty. It completed 10,455 updates / 128,471,040 environment
steps in 5,400.24 s with 512 environments. The final deterministic result is
**5/5**, using the original unchanged solver-ground checks and two-second hold:

| Seed | Outcome | Recovery time |
|---|---|---:|
| 1001 | SUCCEEDED | 8.56 s |
| 1002 | SUCCEEDED | 8.86 s |
| 1003 | SUCCEEDED | 4.28 s |
| 1004 | SUCCEEDED | 4.72 s |
| 1005 | SUCCEEDED | 6.14 s |

The [final report](artifacts/submission/local_stability/evaluation/summary.json),
[manifest](artifacts/submission/local_stability/manifest.json), actor, checkpoint,
videos and training curve are preserved in `artifacts/submission/local_stability`.
This combined targeted revision is not a single-variable ablation. Total
returns from different reward versions are not directly comparable.

The earlier retained checkpoint 9251 also passed 5/5, in 4.48–6.30 s. A separate
15 s seed-1001 geometry audit found foot-centre width -0.0495 m but positive
ankle/knee widths, inward headings of roughly 71°/59°, sole tilts of 26°/59°,
and about 196 N foot-to-foot bracing force. This is an inward-twisted,
staggered, edge-supported stance, not evidence that the leg joint chains have
crossed or that foot collisions are disabled. The exact measurements and model
checks are in the [geometry audit](artifacts/validation/stance_geometry/README.md).

The completed `local_stance` refinement ran 4,828 updates /59,326,464 new transitions in 2,700.50 s, initialized from checkpoint 9251. It retained 5/5 original recoveries (first passes 5.88–9.88 s), but scored **0/5 clean stance**. Signed sole-centre spacing became positive (2.8–11.3 cm at 15 s) but remained below the 16 cm minimum. Feet still turned inward and used edge support; three final frames had approximately 180 N mutual-foot bracing, while two had no bracing but had become unstable again. The selected stability policy retained stability in all five final frames of the same extended audit. Therefore refinement did not establish an overall improvement and does not replace the default actor. See [refinement results](artifacts/submission/local_stance/evaluation/summary.json).

GPU training uses stance geometry as a bracing proxy; final CPU assessment checks actual foot-to-foot normal force. The original success count records the first continuous 2-second pass. In `--assess-stance` mode, later loss of balance is retained in the terminal checks rather than erasing that earlier pass; the clean-stance count is reported separately.

Remaining limitations include primitive contacts, CPU/GPU solver differences,
a narrow reset distribution and a single training seed per configuration.
The next experiment should first establish clean standing balance and validate a physically feasible recovery reference, then use staged reference guidance or a curriculum in a separately documented experiment. Expand reset diversity and repeat with independent seeds after the posture issue is resolved. More parallel environments alone are not evidence that the posture objective will improve.
No result is inferred from an absent report or from reward improvement alone.

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
bash scripts/ros_launch.sh controller:=policy checkpoint:="$PWD/artifacts/submission/local_stability/actor.pt" render:=true seed:=1001 max_sim_duration_s:=15.0 timeout_s:=60.0
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
inference parity. The frozen final stability actor was also verified through
ROS: one request accepted, another rejected while busy, **428 actual joint
frames** and `RUNNING → SUCCEEDED` at **8.56 simulated seconds** for seed 1001.
Evidence is in
[the successful ROS integration report](ros2_ws/validation/policy_recovery/integration.json).
The selected recovery result is 5/5. The completed stance refinement also reached 5/5 first recoveries but 0/5 clean stance; the selected policy is unchanged. Final clean-source and archive reproduction evidence is in artifacts/validation/final_reproduction/.
