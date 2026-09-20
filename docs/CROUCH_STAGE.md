# Crouch-to-standing curriculum stage

This is a local continuation on `physics-v2`, authorized by the user's next-step request. It is not a supine-recovery result. Existing submission policies and historical artifacts remain unchanged.

The moderate crouch is reached through controlled simulation from a perturbed standing start. Initial pelvis height is approximately 0.526 m, below the original 0.6048 m standing threshold. No joint state is clipped to enforce limits. Evaluation audits every 1 ms step and requires the existing clean-stance conditions and a two-second hold.

## Reference and supervision

The scripted teacher interpolates from the known crouch joint command to the nominal standing command over three seconds. Its five evaluation seeds 3001–3005 are separate from the 16 demonstration seeds 101–116. All five reference episodes pass, with 8.666 s of clean stance during a 10 s episode. The scripted teacher is not a learned controller.

The demonstration set contains 8,000 real simulator observations and normalized target actions. The actor uses the existing 106-dimensional observation and 31-dimensional action contract. Supervised fitting refits actor normalization using training demonstration seeds only; the normalizer is held fixed for subsequent PPO to avoid changing the meaning of the learned inputs. Supervised checkpoints can initialize a new PPO experiment but cannot resume one.

The initial clone reached held-out action MSE 1.99e-8 but failed all five closed-loop crouch tests. Its action error grew from below .001 at the start to above 1.0 by .3 s on one audited rollout. Observation-noise regularization alone also failed all five. These failures are preserved; fitting recorded actions is not sufficient evidence of a working controller.

A bounded off-reference data aggregation experiment produced a working warm start before PPO. Labels remain the scripted teacher's time-based commands, not an optimal corrective expert. The subsequent PPO stage was evaluated as its own frozen actor, with crouch, standing and supine outcomes reported separately.

## Verification so far

The new crouch GPU reset has the expected 106 observations and a maximum CPU/GPU observation difference of 1.79e-7. A 20-step crouch-hold smoke test produced no invalid state or limit failure. Existing nine physics/trajectory tests pass. Runtime snapshots now preserve `reset_mode`, and the viewer identifies curriculum stages explicitly.

The off-reference dataset adds eight 10-second mixed-controller rollouts (four seeds, 90% or 60% scripted teacher). All eight remain clean and within limits, but these are blended-controller results, not actor-only successes. Supervised validation uses seeds 113–116 and 123–124; training includes augmented seeds 121–122. Actual executed action history is preserved in observations.

## Closed-loop warm-start result

The actor fitted to the aggregated dataset passes 5/5 crouch-to-standing trials on seeds 3001–3005 with no scripted commands during the timed episode. Clean hold lasts 8.80–8.94 s of the 10 s evaluation. All five trajectories respect the monitored position, speed and effort bounds. This supervised policy is preserved separately from the failed clones.

A separate PPO fine-tune ran for 300.63 seconds, 256 environments, seed 4: 362 updates and 2,224,128 transitions. It started from the aggregated actor/normalizer and parent critic, with fresh optimizer, noise and counters. The actor normalizer stayed frozen. Configuration: action std .015, bounds .005–.04, learning rate 1e-5, clip .05, no entropy bonus. The scripted teacher was absent during PPO rollouts and policy evaluation. The final frozen actor passes 5/5 fresh-seed CPU crouch tests and 5/5 standing tests for their full 10-second duration. It achieves 0/5 supine recoveries. All fifteen trajectories pass the independent joint-limit audit. Logged GPU training success remains zero. A separate deterministic GPU diagnostic passed16/16 initial episodes in2.32–2.34s with no limit faults, invalid states or timeouts; it uses approximate training contact criteria and stops at the two-second hold. These deterministic successes must not be generalized to noisy rollouts. See [final results](../CROUCH_STAGE_RESULTS.md).
