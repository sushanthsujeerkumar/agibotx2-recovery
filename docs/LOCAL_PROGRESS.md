# Local execution progress

Project root: /home/sushanth/Documents/Codex/2026-09-19/l/outputs/hrs-x2-recovery
Private GitHub: https://github.com/sushanthsujeerkumar/hrs-x2-recovery

User authorized full local build, visible simulator and economical timed monitoring.
Cloud is NOT authorized to start in this phase; report local outcome and next plan first.

## Running experiment

- Initial real PPO: artifacts/runs/local (512 environments, seed0, one-hour max-seconds).
- Read process.json for PID and exact start/command, status.json for atomic current status.
- Training started successfully; >1 million environment steps in first minute, ~22k steps/s including PPO, 0 invalid episode rate, no recovery successes yet.
- GPU memory ~1.65GB at early checkpoint with visible CPU viewer; RTX5060 8GB.
- Native visible viewer: x2_recovery.watch, reloads actor_latest.pt between 15-second episodes. It began with scripted baseline and switched to real trained actor. Log is work/viewer.log in parent workspace.
- Main training stops after one hour at a completed PPO update with saved checkpoint/export.
- Checkpoints every250 iterations (~2-3min early). Do not continuously poll; heartbeat check-x2-local-training every15min.

## Completed correctness evidence

- Official URDF-pinned X2 mass/inertias, torque limits, simplified contact shapes, 31 actuators, 1s settled supine resets.
- 3 physics contract tests pass.
- CPU/GPU parity: initial obs error 5.96e-8; max qpos error after0.4s 7.78e-6. artifacts/backend_parity.json.
- Short batched benchmark: 128=15183,256=25266,512=38842 control steps/s (excludes PPO). artifacts/benchmark.json.
- Real PPO smoke: 512env,5iterations,61440steps; saves/reloads/exports CPU actor.
- Independent runner test confirms optimizer+normalizer+RNG restoration and iteration accounting (work/training-runner-check.log).
- ROS19 tests plus actual headless runtime: fresh colcon build, both nodes,0.85ms acceptance,busy rejection,516frames/31joints,FAILED wall timeout. docs/ros_validation.md and ros2_ws/validation/20260919T231348Z-15524.
- Corrected scripted baseline achieves side-roll motion but no complete recovery. Five-episode video evaluation output artifacts/evaluation/scripted_baseline.

## Remaining work

Automatic finalizer scripts/finalize_run.py is already running for local: it sleeps60s with no AI, then copies final checkpoint/actor and runs five-episode video evaluation + plot under artifacts/submission/local_initial. Check its finalization.json before duplicating work. Smoke finalization passed (0/5 expected for five-update smoke policy).

1. Monitor real learning and occasionally evaluate a fixed checkpoint (NOT mutable actor_latest.pt midway across an evaluation). Preserve checkpoint identity.
2. At one-hour stop evaluate final actor 5 seeds1001..1005,15s each, exact feet-only support2s rule; make training reward plot.
3. If clearly stalled, inspect videos/reward and allow at most one targeted local correction; total local training cap3h. Preserve first run; new directory if reward/model changes. Do not blindly extend or rent cloud.
4. Final policy via ROS: launch scripts/ros_launch.sh controller:=policy checkpoint:=ABSOLUTE_ACTOR_PATH; use probe + timeout. Existing script validate_ros.sh defaults scripted.
5. Save final/best checkpoint+actor under artifacts/submission (runs ignored by git), config and progress log, plot and evaluation results. Include selected video (avoid oversized git files); README honest x/5, failure analysis, training walltime.
6. Fresh checkout reproduction validation. Update README/results/docs then meaningful commit+push. Pause heartbeat after final local report and next-step recommendation.

## Tool paths / environment

- .venv/bin/python (Python3.12), uv.lock pinned; scripts/setup.sh reproduces.
- PyTorch2.11.0 CUDA13.0 copied from existing local environment to avoid repeated large downloads; independent new project env; normal uv frozen setup installs same versions elsewhere. Existing project untouched.
- MuJoCo3.11.0, MuJoCoWarp3.11.0, Warp1.17.0, mjlab1.6.0, RSL-RL5.4.2.
- Git credential helper configured per repo using official gh binary under parent work/gh/gh_2.101.0_linux_amd64/bin/gh. Existing keyring authentication used; no credential added.
- All main development milestones already pushed. No need to change user identity/global settings.
- Torch/Warp GPU stream must share Warp-owned stream (default torch stream caused graph-capture error, fixed).
- Actor obs106 and action31; CPU exported actor includes normalizer. Action mapping piecewise full joint range around nominal.
- Ground support during training uses conservative touch sensors (self contact may count); final CPU eval uses solver floor-pair normal forces. Do not confuse training success metric with final success.
