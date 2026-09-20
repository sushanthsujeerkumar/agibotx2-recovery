# Local campaign completed

Project: `/home/sushanth/Documents/Codex/2026-09-19/l/outputs/hrs-x2-recovery`
Private repository: https://github.com/sushanthsujeerkumar/hrs-x2-recovery

The user authorized local development, visible simulation and economical monitoring, then a bounded posture refinement. All three main experiments have finished. No additional training, method revision, cloud resource or spending should be started without further user direction. Main training totaled 9,551.56 seconds (2 h 39 m 12 s), excluding setup, compilation, smoke tests and evaluation.

## Results and retained policy

- Initial PPO: 24 m 11 s, 34,443,264 transitions, 0/5 original posture passes.
- Stability PPO: 90 m, 128,471,040 transitions, 5/5 original posture passes; **0/5 clean stance and 0/5 whole-trajectory joint-limit compliance**. Retained for reproducible inspection, not as a fully successful physical controller.
- Stance fine-tuning: 45 m 01 s, 59,326,464 new transitions, 5/5 first posture passes and 0/5 clean stance. Two episodes were unstable again at the end of the extended check. It does not replace the retained policy.
- Retained actor: `artifacts/submission/local_stability/actor.pt`, SHA256 `4d7dba4dc1f31e01ddeee3f12c7b60b93361a5708d66812585fa8e23dddf08f3`.
- The intermediate checkpoint 9251 is preserved as the fine-tuning parent. Its historical folder name describes crossed stance, but geometric evidence shows inward-rotated/tilted sole centres and mutual-foot bracing; ankle and knee origins retain left/right ordering.

## Final correctness finding

The final audit sampled every 2 ms physics step: all five selected-policy trajectories exceed URDF position and speed bounds. Maximum position excursion is 0.14919 rad (8.55 degrees), speed 4.20 times the rating; commanded torque stays within rated effort. The original success check and ROS `SUCCEEDED` only certify the documented posture condition. They do not certify joint-limit compliance. Do not report the 5/5 posture count as fully valid physical recovery.

Read-only tighter-stop/smaller-step diagnostics reduced position error but did not resolve speed excursions. They were not applied to the submitted model or represented as new validated policies. The next recommended work is a separately versioned local actuator and joint-stop revision, explicit trajectory compliance checks, clean standing and feasible reference validation, then retraining. Cloud scaling alone is not the next justified action.

## Completed verification and evidence

- Pinned official URDF and meshes, model provenance/license, 31 actuators, 41.966521 kg, floating base, settled supine starts.
- Five physics/stance unit tests passed; 19 ROS tests passed earlier.
- A fresh archived source checkout reproduced five posture passes and a fresh colcon build, Trigger acceptance, busy rejection, actual 31-joint telemetry and posture-based `SUCCEEDED`. Existing locked Python dependencies were reused; no clean-machine installation claim.
- Earlier literal CLI, timeout and telemetry evidence is retained. Final evidence is in `artifacts/validation/final_reproduction/`.
- All finalizers and training completed. The visible viewer was stopped cleanly; no project training, viewer or ROS process remained at final process inspection.
- README, RESULTS, START_HERE and requirements mapping explicitly identify the physical and posture limitations. Repository and source archive are a runnable partial-result package.

The monitoring heartbeat should remain paused after packaging. Resume only on user direction. Do not automatically begin another experiment because unresolved policy limitations remain.
