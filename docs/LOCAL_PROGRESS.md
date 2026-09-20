# Local execution progress

Project root: /home/sushanth/Documents/Codex/2026-09-19/l/outputs/hrs-x2-recovery
Private GitHub: https://github.com/sushanthsujeerkumar/hrs-x2-recovery

User authorized full local build, visible simulator and economical timed monitoring. User subsequently approved posture refinement after the second screencast. Cloud is not authorized: report local outcome/next plan first.

## ACTIVE: posture refinement

- Run: artifacts/runs/local_stance, ./TRAIN_STANCE.sh,512env,seed1,max2700s(45min). PID32397; verify process.json and actual command before signaling.
- New reward version3 and variant stance. Initialized learned actor/critic and normalizers from frozen successful checkpoint9251; fresh optimizer/noise/counters/RNG. Parent provenance hash in config/checkpoint. This is fine-tuning, not a scratch policy.
- Added standing-phase signed foot-width/forward-heading/sole-flatness/hipyaw reward. No standing-start curriculum or model changes. Observations/actions unchanged. Training now requires clean stance to terminate; logs original recovery separately. CPU evaluation keeps original criterion and reports an additional clean-stance metric, continuing until clean pass or15s.
- Bounded std0.03–0.20,initial0.10,fixedLR5e-5,clip0.1. CPU parent/export/config tests passed;40update realGPU smoke/resume and episode reset passed. Five physics/stance tests pass. Evidence artifacts/validation/stance_variant.
- Native viewer PID32580 watches local_stance with --assess-stance; overlay shows original and clean holds. Logs/process metadata parent work/viewer-stance.*.
- Automatic finalizer PID32581 sleeps60s without AI; at training end freezes artifacts/submission/local_stance and runs five episodes with --assess-stance +videos and rewardplot. Check its finalization.json before duplicating work. Parent work/finalizer-stance.*.
- Heartbeat check-x2-local-training updated to local_stance every15min. No further method revision or budget extension without user direction. Initial24.2min+stability90min+stance45min=about159min of main local training, under3h cap.

## Preserved results

1. artifacts/submission/local_initial: baseline stopped after1450.8s,2803updates,34443264steps,0/5. It rose but kept hopping/travelling; excessive Gaussian noise11.2 and saturated actions prompted stability correction.
2. artifacts/submission/local_stability: stability run COMPLETE after5400.24s,10455updates,128471040steps,5/5 original stable recoveries. Automated checkpoint/config/progress/curve/videos/manifest preserved. Additional audit artifacts/evaluation/stability_final_stance_audit gives0/5 clean stance. Allfinalstancechecks fail despite genuine stable floor support.
3. artifacts/submission/stable_crossed_stance_009251: earlier frozen9251 candidate,113676288steps,4752s,5/5 original recovery in4.48–6.30s. Parent for stancefine-tuning. Preserve even if later policy regresses. Historical folder name says crossed but geometric audit clarifies below.

## Posture diagnosis

User videos showed inward/crossed-looking legs. Quantitative audit confirms inward-turned, staggered, edge-supported FEET that brace against each other; ankle/knee left/right ordering remains correct, so do not overstate crossed LEGS. Foot-contact-centre signedwidth around-5cm vsnominal+27.43cm. Footheadings~-71/+59deg, sole tilts26/58deg, foot-foot normal force~196N. All31jointaxes/ranges match URDF, footcollisionenabled. No missingcollisionfilter or reversedaxis found. See artifacts/validation/stance_geometry/{README.md,geometry.json}.

Clean stance is separately defined: original stable support plus horizontal pelvis-frame signedfootwidth0.16–0.36m, bothfootheadingerrors<=25deg, |hipyaw|<=.45rad, soles<=20degtilt, footbracing<=2N, held2s. GPUtraining uses geometryproxy for bracing; CPU checks exactfoot-footforce. Original recovery criteria remain unchanged. This extra quality criterion was added after the user's feedback, not retroactively substituted for official recovery results.

## Completed correctness evidence

- Official pinned X2 URDF,41.966521kg,31actuators, torque/velocitycontrol, primitivecontacts, floatingbase, settledsupine resets.
- Five physics/stancegeometrytests pass. InitialGPUCPUobs parity5.96e-8,maxqposdifference7.78e-6 at0.4s.
- Benchmark512env~38842 controlsteps/s excludingPPO; fullPPO~20–24ksteps/s. Localcompute is sufficient for these experiments.
- ROS19tests passed previously; freshcolconbuild+realserviceaccept/busyreject+31jointtelemetry+wall-timeoutpassed.
- Final stable policy now passed actual ROS SUCCEEDED integration: ros2_ws/validation/policy_recovery. Acceptance0.533ms,428frames,8.56s sim/10.837s wall,seed1001,clean shutdown. ActorSHA4d7dba4dc1f31e01ddeee3f12c7b60b93361a5708d66812585fa8e23dddf08f3.
- Fresh sourcecheckout smoke passed earlier using existing lockedvenv, avoiding repeat multiGBdownload.

## Remaining work

1. Monitor active stancefine-tune economically. Occasionally evaluate fixedversionedactor --assess-stance (never mutablelink for reported evidence), preserveidentity. Check originalrecovery retention separately from posture progress.
2. At45min end inspect automaticfinalizer result. Choose best verified artifact based on actual5episode success and quality. Do not replace successfulbaseline with regression or call tilted/bracedstance clean.
3. If finalselectedactor differs from alreadyROSvalidated stabilityactor, validate through ROS. Original ROS successcontract remains unchanged. Integrationprobe available ros2_ws/scripts/check_ros_integration.py.
4. Finalize honestREADME/results/failureanalysis/commands; freshsourcecheckout validation withselectedactor, usinglockedvenv. Save selectedcheckpoint/actor/config/progress/plots/videos/manifest. Commit/push meaningful evidence toexistingprivaterepo. Package ready-to-run deliverable.
5. Stop project-owned activeprocesses afterfinalvalidation; pauseheartbeat. Report actuallocalresult and recommend nextstep beforeanycloud action. Additionalposturelearning is not guaranteed and furthertraining needsuserdirection.

## Tool paths / environment

- .venv/bin/python (Python3.12), uv.lock pinned; scripts/setup.sh reproduces.
- PyTorch2.11.0 CUDA13.0 copied from existing local environment to avoid repeated large downloads; independent new project env; normal uv frozen setup installs same versions elsewhere. Existing project untouched.
- MuJoCo3.11.0, MuJoCoWarp3.11.0, Warp1.17.0, mjlab1.6.0, RSL-RL5.4.2.
- Git credential helper configured per repo using official gh binary under parent work/gh/gh_2.101.0_linux_amd64/bin/gh. Existing keyring authentication used; no credential added.
- All main development milestones already pushed. No need to change user identity/global settings.
- Torch/Warp GPU stream must share Warp-owned stream (default torch stream caused graph-capture error, fixed).
- Actor obs106 and action31; CPU exported actor includes normalizer. Action mapping piecewise full joint range around nominal.
- Ground support during training uses conservative touch sensors (self contact may count); final CPU eval uses solver floor-pair normal forces. Do not confuse training success metric with final success.
