> Historical experiment record. The submitted model and current results are described in [../RESULTS.md](../RESULTS.md).

# Landing-residual PPO and ROS stage completed

Six PPO updates collected 32,268 transitions in a 183.14-second local campaign, including validation. Selected update 4 and unchanged teacher both pass 24/25 fresh paired episodes (5/5 first batch), with the same right-wrist fault. No improvement claim. The external teacher supplies recovery; the trained policy adds bounded ankle offsets. Original independent PPO remains 0/5.

Reference-controller ROS integration is verified: fresh colcon build, 27 ROS tests, successful physical recovery with 750 actual joint frames, fast acceptance, busy rejection, and both wall/simulation timeout failures. Full Python/ROS suite: 46 tests. Display and headless evidence match. See [current report](../LANDING_RESIDUAL_RESULTS.md) and [run guide](../START_HERE.md). Further training is stopped for delivery; no cloud or employer submission was started.

---

# Reference-guided student stage completed; recovery remains unresolved

Teacher collection passed 14/16. Four supervised fits (50,000 gradient steps, 80.67 seconds of fitting) produced no student-alone recovery: the final fresh batch scored 0/5, including three speed faults. No new PPO or cloud run was started. Student weights and datasets remain in the ignored local experiment directory; numerical evidence and source are preserved. The prior PPO actor hash is unchanged.

A live replay exposed shared viewer data changing contact observations. Both the student recorder and main runtime now isolate display model/data. The student video matches headless evidence exactly; the main viewer has regression and physical-parity checks. See [the full stage report](../REFERENCE_STUDENT_RESULTS.md). Next: preserve the explicit reference/feedback contract and first address the teacher ankle-speed failures before reference-conditioned PPO.

---

# External get-up compatibility stage completed

The user approved inspection of the official get-up reference. The public ONNX
contains a 151-value single-frame actor input and 344 reference frames. All 29
joint names map; our forward kinematics matches 14 reference body positions
within 0.78 micrometres. Raw reference angles exceed some pinned limits; none of
our robot limits were relaxed. The existing guarded controller, model, gains and
contact geometry remain unchanged.

A separately labelled external vendor actor plus our adapter, with the reference
timeline stretched by 1.2 and continuous feedback, passes 5/5 fresh supine seeds
9401–9405 through the full 15 seconds. Final clean holds are 2.66–3.14 seconds;
there are zero position violations, peak speed is 0.93819 of the rating, and peak
commanded effort is 1.0. Five-second preparation is inside the deadline and no
state is injected after physical supine reset. Actual recording/live viewer for
9401 agrees exactly with numerical evidence and exits 0. Native timing passed
2/5 on seeds 9301–9305, with three wrist-speed faults. Fixed-command termination,
open-loop references and a 1.1 timescale candidate failed. All are preserved.

This is EXTERNAL PRETRAINED policy evidence, not our PPO training. The preserved
own actor still has its prior 0/5 supine result. No new PPO campaign, cloud use or
historical ZIP update occurred. Vendor ONNX, configuration, library and reference
arrays remain outside Git; redistribution/training-data terms are unverified.
The small library was inspected for symbols only, never loaded. See
../VENDOR_REFERENCE_RESULTS.md and artifacts/validation/vendor_reference. The
optional demo launcher requires local vendor assets. Next: appropriately
attributed reference-guided own-policy training, then independent evaluation.
Heartbeat remains paused; do not silently promote vendor success to own PPO or
change ROS defaults.

# Floor-transition feasibility stage

The approved bounded local reference work established a scripted deep-crouch →
hands-and-feet squat → clean-standing motion in 5/5 fresh seeds 9101–9105. Full
16 s trajectories pass 1 kHz position/speed/commanded-effort monitoring; minimum
pelvis height 0.260240–0.260245 m, terminal clean hold 4.46 s, peak speed 0.19727
of rating, effort reaches 1.0. Actual re-simulation video matches the numerical
validation. Run ../RUN_HAND_SUPPORT_DEMO.sh; see ../FLOOR_TRANSITION_RESULTS.md.

No floor-to-squat connection was found. Twelve supine candidates yield only a
repeatable reclined arm-propped endpoint (formal one-second intermediate hold
0/5); twelve symmetric kneeling bridges, six physical IK waypoint attempts and
two gravity-offset attempts all fail to return to standing while staying within
monitored joint limits. Three static poses fit the torque budget. The one
12-waypoint fixed-support path fails its last interpolated interval with 17.59 cm
floor penetration and was rejected before physical replay. Offline kinematic and
static images/results are not physical or learned motion. No new PPO training,
cloud compute, actor replacement or main ZIP update occurred. The retained PPO
actor's most recent full supine assessment remains 0/5.

Next method: a collision-free contact-changing reference, physically validated
before imitation/PPO. Official AgiBot archive metadata/configuration inspection confirmed a get-up
ONNX policy for a different interface/model revision. Its graph/runner semantics
and reuse terms remain to be verified; no vendor policy binary was downloaded or
integrated. See the official_reference_research.md evidence note. Preserve all earlier evidence. The heartbeat remains paused.

# Completed deeper-crouch stage

The user approved continuation. Final deep_crouch PPO update226 is frozen in artifacts/experiments/deep_crouch_ppo after180.16s,1,388,544 transitions. Fresh CPU seeds:5/5deep,5/5moderate,5/5standing; noise.0035/5, .0050/5butall10s, .0150/5withfour earlyfalls. Supine0/5strict, five timeouts. Joint-limit reports are valid only through observed trajectories. Main robustness gain is a contact-normalization correction: normalized flags previously changed by100, now1. Matched .003 oldactor1/5, correctedwarm5/5, finalPPO5/5; no incremental pass-count gain from the short PPO pilot. Generic inputclip5failed and was rejected. Scripted deep reference5/5,12valid demos6000samples captured but unused in refinement.33CPU/controller/ROS tests pass. See ../DEEP_CROUCH_STAGE_RESULTS.md. Keep prior checkpoints and main historical ZIP. No cloud; scheduled heartbeat remains paused. Next missing work is physical floor-to-crouch through sitting/kneeling, not more unstructured worker scaling.

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
