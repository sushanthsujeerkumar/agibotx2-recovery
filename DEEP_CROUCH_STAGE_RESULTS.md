# Deeper crouch and contact-normalization correction

The selected curriculum actor passes **5/5 deeper-crouch rises, 5/5 moderate-crouch rises, and 5/5 standing tests**. It also passes **5/5 deeper rises with small action noise**. Supine recovery remains **0/5**. This is an improved intermediate controller, not a solved floor-recovery assessment.

## What improved

The preceding PPO actor already transferred to the deeper start without additional training. It rose from a pelvis height of approximately **0.4725 m**, versus 0.526 m for the moderate start, in five complete 10-second CPU tests. A separate GPU diagnostic passed all 16 initial environments, representing 11 unique reset seeds. No noisy or scripted trial is included in those deterministic learned-policy counts.

Noise exposed a specific input-scaling problem. Contact flags had constant values in the demonstrations, so their stored standard deviations were zero. The normalizer divides by standard deviation plus 0.01: a foot flag changing from 1 to 0 therefore became −100. In a paired rollout this happened before rapid loss of height. Zero variance was not a division-by-zero error.

A separate candidate changed only the three contact normalization scales, making flag changes ±1 while preserving nominal inputs, actor weights, means and other statistics. At noise standard deviation 0.005, the old actor fell quickly in all five CPU trials; the corrected actor completed all five 10-second trials, although it never held clean standing for two uninterrupted seconds. A generic ±5 input-clipping candidate failed and was rejected.

One **180.16-second PPO refinement** then ran with 256 environments, 226 updates and 1,388,544 transitions. It explicitly froze actor normalization and used smaller exploration noise, initially 0.003. No scripted teacher supplied actions during PPO. The final checkpoint retains the corrected contact scale and has updated actor weights; the audit checks its export against the checkpoint.

In a matched noise-0.003 comparison on seeds 8301–8305, the old actor passed **1/5**, the contact-corrected warm start **5/5**, and final PPO **5/5**. Thus the measured pass-count improvement comes from the contact correction; the short PPO run did not demonstrate an additional pass-count gain. Both corrected artifacts are retained. The final PPO actor is used for the replay after passing all three deterministic start conditions.

## Final fixed-actor evaluation

| Test | Two-second clean hold, full duration, clean finish | Full-duration trials |
|---|---:|---:|
| Deeper crouch, seeds 8001–8005 | **5/5** | 5/5 |
| Moderate crouch, seeds 8101–8105 | **5/5** | 5/5 |
| Standing, seeds 8201–8205 | **5/5** | 5/5 |
| Deeper crouch + noise σ=0.003, seeds 8301–8305 | **5/5** | 5/5 |
| Deeper crouch + noise σ=0.005, same seeds | **0/5** | 5/5 |
| Deeper crouch + noise σ=0.015, same seeds | **0/5** | 1/5 |
| Supine, seeds 1001–1005 | **0/5 recoveries** | Five 15-second timeouts |

Noise is independent Gaussian noise added to all 31 normalized actions at 50 Hz before action clipping. It is a synthetic stress test, not a physical disturbance specification. Four largest-noise trials stop early when the pelvis drops below 0.3 m. Their joint-limit reports cover only the observed prefixes, not an invented complete duration.

Each successful deeper trial stays clean for **8.84–8.88 s** of its 10 s episode. Maximum joint-position overshoot is zero; peak speed is **0.315 times rated**, and commanded effort reaches, but does not exceed, its rating. Position, velocity and commanded effort are monitored at 1 kHz. Posture, exact contacts, sole orientation, left/right foot ordering and foot bracing are checked at 50 Hz. The five successful small-noise trials also satisfy all monitored bounds.

The starts are physically generated, modest perturbations of one crouch. Reset preparation occurs before the timed trial; no joint state is clipped during execution. These results do not establish broad robustness or floor-to-crouch ability.

## Replay and reproduction

```bash
./RUN_DEEP_CROUCH_DEMO.sh
```

This opens a one-minute simulator replay using the frozen final actor, labelled **DEEP_CROUCH ONLY**. The moderate-crouch launcher retains the earlier actor for comparison.

[Watch the final deeper-rise video](artifacts/experiments/deep_crouch_ppo/video/episode.mp4).

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_balance.py --start deep_crouch \
  --checkpoint artifacts/experiments/deep_crouch_ppo/actor.pt --seed 8001 \
  --output artifacts/local_evaluation/deep_crouch.json
./TRAIN_DEEP_CROUCH.sh --output artifacts/runs/deep_crouch_reproduction
```

The second command starts another bounded three-minute run. Training data and reference demonstrations are retained for later curriculum work, but the new 6,000 deep-reference samples were **not used** in this PPO refinement.

## Evidence and remaining work

- [Frozen checkpoint, actor, configuration, provenance and hashes](artifacts/experiments/deep_crouch_ppo/manifest.json); [actual training curve](artifacts/experiments/deep_crouch_ppo/training_curve.png).
- [Deep evaluation](artifacts/experiments/deep_crouch_ppo/deep_evaluation.json), [small-noise passes](artifacts/experiments/deep_crouch_ppo/noise_0003.json), [larger-noise failures](artifacts/experiments/deep_crouch_ppo/noise_0005.json), [supine failures](artifacts/experiments/deep_crouch_ppo/supine_evaluation/summary.json).
- [Paired noise diagnosis](artifacts/validation/deep_crouch_stage/noise_diagnosis/README.md), [normalization correction audit](artifacts/validation/deep_crouch_stage/contact_normalization_review.json), [final policy audit](artifacts/validation/deep_crouch_stage/final_policy_audit.json).
- [Scripted reference and 12 valid demonstrations](artifacts/validation/deep_crouch_stage/reference/README.md), [33 passing CPU/controller/ROS tests](artifacts/validation/deep_crouch_stage/tests.log).

The next missing skill is a physically valid floor-to-crouch transition through sitting or kneeling. The current actor still fails that assessment start. Preserve these passing crouch checkpoints while developing the missing transition. No cloud resources were used; local training is stopped and scheduled monitoring remains paused.
