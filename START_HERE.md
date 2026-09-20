# Run the submitted project

The retained policy passes the original posture-based check in 5/5 episodes, but the additional clean-stance and whole-trajectory joint-limit audits both score 0/5. The simulator, PPO pipeline, ROS integration and evidence are runnable; the controller remains a partial result with unresolved physical-limit and posture problems. See [RESULTS.md](RESULTS.md). Do not describe the earlier 5/5 posture count as fully validated physical recovery.

## On the development PC

Open a terminal in this project folder. The Python environment and ROS 2 Jazzy are already installed here.

```bash
./RUN_DEMO.sh
```

This opens one learned recovery attempt using the frozen selected actor. It closes after passing the original recovery test or reaching the 15-second limit. Ctrl+C stops it. To watch the longer posture assessment:

```bash
./RUN_DEMO.sh --assess-stance
```

The extra clean-stance timer is deliberately separate from recovery success.

## Five repeatable evaluation episodes

```bash
./EVALUATE.sh
./EVALUATE.sh --assess-stance --output artifacts/local_evaluation/stance
```

Results are JSON under `artifacts/local_evaluation`. The first command reproduces the original metric; the second continues after a recovery pass to check posture. For headless videos add `--video` and set `MUJOCO_GL=egl` on this NVIDIA host.

## ROS 2 demo

Terminal 1, in the project folder:

```bash
./RUN_ROS.sh
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState --once
ros2 topic echo /x2/recovery_status std_msgs/msg/String
```

The first request is accepted, and a second request while running is rejected. Joint states come from the simulator. The selected seed 1001 episode was verified to reach `SUCCEEDED` under the posture criterion; this status does not certify the separately audited joint-limit compliance. Ctrl+C stops the topic echo and launch. Both nodes are started by one launch file. [ROS validation](docs/ros_validation.md) records fresh build, success, busy rejection and forced-timeout evidence.

## On a fresh Ubuntu 24.04 machine

Extract the archive or clone the repository, then install ROS 2 Jazzy and colcon following [README.md](README.md). From the project folder:

```bash
bash scripts/setup.sh
./EVALUATE.sh
./RUN_DEMO.sh
```

Setup creates a local Python 3.12 environment from `uv.lock`. Model meshes, source URDF, license, policy and evidence are included; training is not required to run the selected actor. CUDA is needed for the documented training commands; CPU policy evaluation uses MuJoCo and PyTorch. A desktop with OpenGL is required for visible playback.

## Package contents

- Selected policy: `artifacts/submission/local_stability/actor.pt` (TorchScript inference) and `checkpoint.pt` (full PPO state).
- [Selected provenance and hashes](artifacts/selected_policy.json).
- [Training curve](artifacts/submission/local_stability/training_curve.png) and [five-episode results](artifacts/submission/local_stability/evaluation/summary.json).
- [Detailed results and next experiment](RESULTS.md).
- [Requirement-to-evidence mapping](docs/requirements.md).
- [GitHub repository](https://github.com/sushanthsujeerkumar/hrs-x2-recovery), with meaningful development history. The ZIP is a convenience source snapshot; the GitHub repository retains the history required by the assessment. Repository access is private and must be granted to reviewers before they can inspect it.
