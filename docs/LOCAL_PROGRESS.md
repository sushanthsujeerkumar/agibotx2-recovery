# Local execution progress

Project root: /home/sushanth/Documents/Codex/2026-09-19/l/outputs/hrs-x2-recovery
Private GitHub: https://github.com/sushanthsujeerkumar/hrs-x2-recovery

User authorized full local build, visible simulator and economical timed monitoring.
Cloud is NOT authorized to start in this phase; report local outcome and next plan first.

## Active experiment / intervention

The initial local run was stopped gracefully after 1450.8s (~24.2min), iteration 2803, 34,443,264 steps because learned Gaussian action std grew to 11.2 and nearly all raw deterministic outputs saturated the action clamp. Frozen checkpoint 2501 reached feet/upright/height but kept moving. Final initial policy evaluation is COMPLETE: 0/5 successes, all timeouts, no invalid physics. Evidence: artifacts/submission/local_initial (checkpoint, actor, config, progress, curve, five videos + JSON, hash manifest). This is partial recovery, not stable standing.

The ONE targeted correction is now RUNNING: fresh PPO variant stability with bounded std .05–.6, initial .4, entropy 1e-4; reward version 2 adds gated dense base-velocity stabilization and a mild upright posture penalty. We did not warm-start the saturated baseline outputs.

- Active directory: artifacts/runs/local_stability. Read its status.json and process.json (PID 23009; verify identity, PID can be reused).
- Command: ./TRAIN_STABILITY.sh. 512 environments, seed 0, maximum 5400 seconds (90 minutes), checkpoint every 250 iterations.
- GPU smoke passed: 128 environments, five PPO updates, exported actor CPU rollout; independent runner checks cover updates/export and standard-deviation bounds.
- Native viewer PID 23146 watches local_stability; parent work/viewer-stability.log and process JSON.
- Lightweight finalizer PID 23147 sleeps 60 seconds between status checks without AI usage. It will save artifacts/submission/local_stability; inspect finalization.json before duplicating work. Parent work/finalizer-stability.log and process JSON.
- Heartbeat check-x2-local-training updated to check this run every 15 minutes. No further method revisions without user direction; total local training <=3 hours. Initial + planned revised run is about 114 minutes.
- Initial revised throughput ~22k environment steps/s; std ~0.41. These are health measures, not recovery evidence.

## Initial experiment (preserved)

- Original PPO: artifacts/runs/local (512 environments, seed 0). Gracefully stopped at 24.2 minutes after diagnosing behavior; full original config and evidence preserved.
- Trained throughput ~22–24k steps/s including PPO; zero invalid episode rate. GPU memory ~1.65GB with visible CPU viewer.
- Finalizer completed artifacts/submission/local_initial: 0/5 stable recoveries. All ended upright at standing height, but all failed linear/angular speed thresholds; both feet supported in 3/5 final frames. Do not label this successful recovery.

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

1. Monitor revised real learning economically; occasionally evaluate a fixed checkpoint, never a mutable actor_latest.pt across an evaluation. Preserve checkpoint identity.
2. At revised 90-minute stop, use the automatic finalizer output: five seeds 1001–1005, 15 seconds each, exact feet-only support for two seconds, reward plot and frozen artifacts.
3. Inspect videos and success checks. One targeted correction has already been used. Do not blindly extend or rent cloud.
4. Validate selected final policy through ROS: scripts/ros_launch.sh controller:=policy checkpoint:=ABSOLUTE_ACTOR_PATH, then actual service/topic probe and timeout. Existing policy smoke passed (471 frames, busy rejection, configured timeout), but run the final artifact too. See docs/ros_validation.md and ros2_ws/validation/policy_smoke.
5. Update README/results/docs with final x/5, failure analysis, actual wall time, reproducible commands and selected artifact. Commit/push evidence; runs are ignored, submission evidence is tracked.
6. Fresh source checkout smoke already passed: three physics tests and one exported-policy simulator step using the existing locked Python environment. Final validation should exercise the selected actor from a clean source checkout, without another multi-GB dependency download.
7. Finish ready-to-run deliverable, stop project-owned active simulator/training processes cleanly, pause heartbeat, and report final local result plus the next plan before any cloud action.

## Tool paths / environment

- .venv/bin/python (Python3.12), uv.lock pinned; scripts/setup.sh reproduces.
- PyTorch2.11.0 CUDA13.0 copied from existing local environment to avoid repeated large downloads; independent new project env; normal uv frozen setup installs same versions elsewhere. Existing project untouched.
- MuJoCo3.11.0, MuJoCoWarp3.11.0, Warp1.17.0, mjlab1.6.0, RSL-RL5.4.2.
- Git credential helper configured per repo using official gh binary under parent work/gh/gh_2.101.0_linux_amd64/bin/gh. Existing keyring authentication used; no credential added.
- All main development milestones already pushed. No need to change user identity/global settings.
- Torch/Warp GPU stream must share Warp-owned stream (default torch stream caused graph-capture error, fixed).
- Actor obs106 and action31; CPU exported actor includes normalizer. Action mapping piecewise full joint range around nominal.
- Ground support during training uses conservative touch sensors (self contact may count); final CPU eval uses solver floor-pair normal forces. Do not confuse training success metric with final success.
