# Local results and selected policy

The selected learned policy is `artifacts/submission/local_stability/actor.pt`, SHA256`4d7dba4dc1f31e01ddeee3f12c7b60b93361a5708d66812585fa8e23dddf08f3`. It passed 5/5 fixed-seed recovery episodes under the documented criteria and a real ROS 2 recovery episode. It is retained because the posture refinement did not deliver a clean stance and was less consistent during the extended 15-second check.

## Experiment comparison

| Experiment | Main training time | New environment steps | Original recovery | Clean stance |
|---|---:|---:|---:|---:|
| Initial PPO |24 m 11 s|34,443,264|0/5|Not assessed separately|
| Stability PPO, selected |90 m 00 s|128,471,040|5/5|0/5|
| Posture fine-tuning |45 m 01 s|59,326,464|5/5|0/5|

Total main training was 9,551.56 s, or 2 h 39 m 12 s. Setup, kernel compilation, small smoke runs, model validation, evaluation and documentation are separate overheads. This is not a claim that the complete project took only 2 h 39 m. All computation was local on the RTX5060/i 5-10600 K host; no cloud was used. Training ran 512 parallel environments at approximately 20–24 k transitions/s including PPO updates. A separate CPU viewer replayed exported deterministic policies.

Reward versions differ, and fine-tuning inherited a trained policy. Therefore these runs are engineering iterations, not a controlled comparison of algorithms or an unbiased proof of faster learning. No Blender reference, motion imitation or second RL algorithm was used. The checkpoint 9251 intermediate candidate also passed 5/5 and remains preserved as the fine-tuning parent.

## Five-episode evaluation

Every episode starts from a settled supine state with small reset perturbations. Seeds 1001–1005 were fixed across comparisons. Original success requires torso tilt<=15 deg, pelvis height>=0.6048 m, both feet supplying>2 N floor-normal force each, all other body-floor forces totaling<=2 N, base linear speed<0.15 m/s and angular speed<0.3 rad/s, continuously for 2 s. The deadline is 15 simulated seconds. CPU evaluation inspects actual solver floor-contact forces.

| Seed | Selected policy recovery time | Fine-tuned policy first recovery time | Fine-tuned clean stance |
|---|---:|---:|---|
|1001|8.56 s|5.88 s|Failed|
|1002|8.86 s|9.88 s|Failed|
|1003|4.28 s|6.16 s|Failed|
|1004|4.72 s|5.96 s|Failed|
|1005|6.14 s|6.00 s|Failed|

There were no invalid-physics terminations in these recorded evaluations. The fine-tuned policy passed the original check at least once in each episode, then continued for the posture assessment. Seeds 1003 and 1004 no longer satisfied all original instantaneous checks at 15 s. This is why a reported first recovery pass can coexist with a failed terminal check. The selected policy still passed the instantaneous checks in all five terminal frames of its extended audit.

The same five seeds were inspected during development and checkpoint selection; they are not an untouched statistical test set. The reset distribution is narrow. These 5/5 counts do not demonstrate broad robustness, diverse falls or hardware transfer.

## Remaining posture failure

The selected policy learns an inward-twisted, staggered stance supported partly on the edges of the feet, with substantial mutual-foot bracing. A geometric audit found the joint axes and ranges match the URDF and foot-to-foot collision is active. Knee and ankle origins retain left/right order; the earlier visual description of “crossed legs” was imprecise. Rotation and tilt reverse the ordering of the sole collision centres.

The additional clean-stance criterion requires the original checks plus sole-centre separation 0.16–0.36 m in a horizontal pelvis frame, each foot heading error<=25 deg, each hip-yaw angle<=0.45 rad in magnitude, sole tilt<=20 deg and mutual-foot normal force<=2 N, all for 2 s. These are additional project quality thresholds, not retrospectively substituted assessment criteria.

Fine-tuning improved terminal signed sole-centre spacing from roughly-5 cm to+2.8–11.3 cm. It did not reach the 16 cm minimum or resolve the heading, sole-tilt and hip-yaw failures. Three terminal frames still had approximately 180 N mutual-foot bracing. No episode achieved any positive clean-stance hold. The retained baseline is thus a successful recovery experiment with a clearly documented posture limitation, not a solved neutral-stance controller.

A weak mean joint-posture penalty originally allowed this solution. Adding stance shaping during low-noise fine-tuning did not escape it within 45 minutes. The initialized policy, the geometry proxy used during GPU training, and competing rewards may contribute to this local optimum; these experiments do not isolate their individual causal effects.

## Next experiment and cloud decision

Stop the current local campaign and retain the complete, reproducible result. If further posture work is required, first validate a clean standing controller and a physically feasible recovery reference on this same model. Then run a separately versioned, staged reference-guided or curriculum experiment, documenting any change to training resets and keeping assessment resets supine. Apply the exact floor and self-contact checks during training where supported. Verify with additional unseen reset seeds and multiple training seeds after the posture objective works.

The current PC already supplies useful GPU throughput. The observed remaining problem is learning the desired contact/posture strategy, so a larger cloud instance is not the next justified step. No cloud price or speedup claim is made, and no paid resource has been started.

## Evidence

- [Initial run](artifacts/submission/local_initial/manifest.json),[selected run](artifacts/submission/local_stability/manifest.json),[fine-tuning](artifacts/submission/local_stance/manifest.json).
- [Selected original evaluation](artifacts/submission/local_stability/evaluation/summary.json),[selected extended posture audit](artifacts/evaluation/stability_final_stance_audit/summary.json),[fine-tuned evaluation](artifacts/submission/local_stance/evaluation/summary.json).
- [Geometry audit](artifacts/validation/stance_geometry/README.md),[ROS evidence](docs/ros_validation.md),[fresh-source reproduction](artifacts/validation/final_reproduction/README.md).
- Training configs, full checkpoints, TorchScript actors, plots, progress logs and five episode videos accompany each preserved run.
