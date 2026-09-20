# Active deeper-crouch stage

User approved the next local stage. Existing crouch_ppo actor transfers5/5 deeper starts but noise.005/.015 causes falls. Contact flag normalization was identified as a large input amplification; a separate contact-scale candidate retains5/5 deterministic rises and survives10s at noise.005 but lacks2s uninterrupted clean hold. Root will run one180s256-environment deep_crouch PPO pilot initialized from artifacts/experiments/deep_contact_normalization/checkpoint.pt, explicitly frozen actor normalization, action std.003, bounds.001–.01. Preserve original crouch_ppo until final comparisons. No cloud. Reference5/5 and12deep demos captured; GPU original transfer16/16 deterministic,0/16 at noise.005. See artifacts/validation/deep_crouch_stage for evidence.

# Completed crouch-to-standing stage

The user's next "nxt" request authorized this local stage. Final PPO update362 after300.63s and2,224,128 transitions is frozen at artifacts/experiments/crouch_ppo. It passes5/5 fresh-seed CPU crouch rises and5/5 standing tests through the full10s, but0/5 supine recoveries; all15 trajectories respect monitored limits. Reference and aggregated supervised actor also pass5/5 moderate-crouch tests. Both failed direct clones are preserved. See ../CROUCH_STAGE_RESULTS.md and CROUCH_STAGE.md. CPU/controller/ROS tests:32passed. Timed training and viewers exited0, heartbeat remains paused, no cloud or spending. GPU stochastic training success stays0, but the separate deterministic GPU diagnostic passed16/16 initial episodes with no faults; exploration sensitivity is plausible, not causally proven. The deeper scripted reference passes5/5 at minimum pelvis~.4715m but has not been learned. Do not report curriculum success as supine recovery.

# Physics v2 local phase completed

User's "nxt" authorized the next local correction phase. Branch: physics-v2. No cloud or spending. See ../PHYSICS_V2_RESULTS.md for current results and commands.

Guarded nominal controller and both three-minute PPO balance pilots pass 5/5 clean standing tests for 10 seconds within the monitored joint limits. Both PPO policies achieve 0/5 supine recoveries, while remaining within joint limits in those five timed attempts. No full recovery claim. The original posture count and original source ZIP remain historical.

Completed: 31 unit tests, GPU clean-balance and injected-limit-fault checks, actual ROS acceptance/busy rejection/telemetry/strict FAILED integration, standing/supine evaluations, reference crouch screen, learned-policy video, and viewer shutdown fix verified with exit 0. Two earlier viewer processes crashed during teardown; their training processes and headless evidence were unaffected. All experiment and viewer processes are stopped. Monitoring heartbeat remains paused.

The random-initialized balance pilot eventually passed deterministic standing despite poor noisy training metrics. The nominal-prior pilot also passes; its initialization is explicitly disclosed. A slow scripted moderate crouch-and-rise reference is feasible in a one-seed screen, but the learned balance policy cannot yet rise from that crouch. Next work is staged reference-guided crouch/rise and then kneeling/sitting/supine transitions, validated before larger training. Do not automatically launch another campaign on a heartbeat.

---

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
