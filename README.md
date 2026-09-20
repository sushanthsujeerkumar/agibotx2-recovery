# AgiBot X2 ground recovery

A MuJoCo recovery controller for the AgiBot X2 Ultra, with PPO training and a ROS 2 service interface.

The submitted policy passed **5/5 fresh assessment episodes**, starting on its back and finishing with two seconds of stable, uncrossed standing. It also passed the earlier ten-episode final-checkpoint evaluation. Both sets check joint position, velocity and commanded effort throughout the trajectory. [Results and limits](RESULTS.md) · [Release checks](docs/verification.md) · [Recovery video](artifacts/submission/final_recovery/recovery.mp4) · [Requirement mapping](docs/requirements.md)

The controller combines a frozen motion prior with learned PPO feedback. The prior comes from recovery trajectories generated in this project; PPO learns small corrections using the robot's state across the full episode. The motion sequence was **not discovered from scratch by PPO**. No external policy or vendor controller is needed to run the submitted model.

## Run it

Tested on Ubuntu 24.04, Python 3.12 and ROS 2 Jazzy. CPU MuJoCo runs evaluation and the ROS controller; an NVIDIA CUDA GPU is needed for the provided training backend. The robot meshes and final model are included.

```bash
# From the repository root. Downloads the locked Python environment.
bash scripts/setup.sh

# Check included model hashes and recorded evidence.
python3 scripts/verify_final_artifacts.py

# Five fresh simulated episodes; writes artifacts/local_evaluation/final_policy_5.json.
./EVALUATE.sh

# One visible 15-second recovery, paced in real time.
./RUN_DEMO.sh
```

The first dependency installation is several GB because the training environment includes PyTorch/CUDA. A desktop display is required for the live viewer. Headless evaluation does not open a window. The submitted video can be viewed without installing the simulator.

## ROS 2

Install ROS 2 Jazzy and colcon separately. For an existing Jazzy installation, the additional build/test tools are available with:

```bash
sudo apt install python3-colcon-common-extensions python3-pytest ripgrep
```

Terminal 1, from the repository root:

```bash
./RUN_RECOVERY.sh render:=true
```

This runs `colcon build`, sources the workspace and launches the recovery and telemetry nodes together. Use `render:=false` on a machine without a display.

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
# Repeat while recovery is running to check busy rejection.
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState --once
ros2 topic echo /x2/recovery_status std_msgs/msg/String
```

The first call returns `success=True` before simulation begins. A second call while queued or running returns `success=False`. The final status is `SUCCEEDED` only after the full 15-second trajectory and standing checks pass; an invalid trajectory or a timeout gives `FAILED`. Ctrl+C stops the launch or topic subscriber.

| Interface | Type | Purpose |
|---|---|---|
| `/x2/start_recovery` | `std_srvs/srv/Trigger` | Accept one recovery attempt, reject requests while busy |
| `/x2/recovery_status` | `std_msgs/msg/String` | `IDLE`, `RUNNING`, `SUCCEEDED`, `FAILED` |
| `/x2/joint_states` | `sensor_msgs/msg/JointState` | 31 joint names, simulated positions and simulation timestamps at 50 Hz |

The recovery node supervises a simulator subprocess so that startup or a stalled simulation cannot block its watchdog. The telemetry node logs status and the first joint's measured position. [Build, launch and timeout evidence](docs/ros_validation.md)

## Train and reproduce

```bash
# Reproduce the final training stage from the included 51-update checkpoint.
# Trains 256 environments for about 30 minutes and displays 16 of them.
./TRAIN_FULL_RECOVERY.sh

# Same training without rendering overhead.
./TRAIN_FULL_RECOVERY.sh --headless

# Start PPO feedback from zero on the included, already fitted motion prior.
.venv/bin/python scripts/train_full_recovery.py --output artifacts/runs/from_prior \
  --seconds 1800 --num-envs 256 --headless --evaluate-final
```

In the batch viewer, press **V** to switch between 16 and all 256 robots. It displays snapshots of the actual training environments. Rendering has its own simulator data and cannot change their dynamics. Training stops at its time budget, saves a checkpoint, and runs a separate deterministic evaluation. Ctrl+C requests a checkpoint and clean shutdown. A timer or iteration count reproduces the procedure, not an identical result across different GPUs or solver versions.

[Training settings and reference provenance](docs/training.md) explain how to regenerate the supervised prior, resume training, export the actor and record video. [Environment specification](docs/environment.md) gives every observation, action, reward term and success threshold. [Model notes](docs/model.md) cover the source URDF, limits and simulation assumptions.

## Design choices

MuJoCo keeps the model, CPU evaluation and ROS runtime small enough to inspect. The mjlab/Warp backend batches the same model on the GPU for training. The 31 joint targets go through a bounded velocity servo and effort-limited motors; the floating base is never repositioned during recovery.

Early direct PPO runs produced crossed-foot standing and joint-limit excursions. Those failures motivated explicit stance checks and a stricter actuator/physics profile. A physically generated recovery reference then provided a useful starting motion for a short PPO feedback run. The final method trades exploration freedom for reliable performance within this narrow task. Prior-only validation already passed 10/10, so the results do not establish that PPO improved success rate.

The main remaining limitation is the reset distribution: small perturbations around one supine pose on one flat floor. Side/prone falls, pushes, varied friction and hardware transfer have not been validated. The next useful experiment is a paired prior-only versus PPO comparison under progressively larger disturbances, followed by a wider reset curriculum.

## Files and evidence

- `src/x2_recovery/`: model contract, simulation, policy export, training and viewer.
- `ros2_ws/src/x2_recovery_ros/`: recovery node, telemetry node and one launch file.
- `artifacts/submission/final_recovery/`: selected actor, resumable checkpoint, reference, raw training log, reward plot, evaluation and ROS evidence. `manifest.json` identifies the deployment files and hashes.
- `tests/`: export parity, physics, stance, viewer isolation and related regression tests.
- Earlier experiments, failed checkpoints and their results remain in Git history. The working tree contains the submitted method only; see [development history](docs/development.md).

The original robot URDF and meshes retain their Mulan PSL v2 license in `assets/x2/`. The Git history records the actual development stages; earlier unsuccessful checkpoints have not been relabelled as successful results.
