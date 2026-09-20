# Deterministic GPU crouch-stage diagnostic

The frozen final PPO actor passed the GPU training environment's clean-stage
termination in **16/16 initial environments**: one at 2.32 s and fifteen at
2.34 s. There were **zero timeouts, invalid episodes or reported limit faults**.
The normal reset-pool sampler selected 11 distinct reset seeds across these 16
environments; this is not a 16-independent-seed result.

The actor ran deterministically, with no action-distribution noise, optimization
or parameter updates. Its SHA-256 is
`188b4b76b936eb32d4fa7e2e86385bf993bdf12cc094338881a64930610d0c91`.
The unchanged environment used `guarded_v2`, `reset_mode=crouch`, reward version
3 and 16 worlds. Initial pelvis heights were 0.52580–0.52592 m.

Results come directly from `env.step` completed-episode metrics aligned with
`nonzero(done)`, including clean success, original-standing attainment,
invalidity, limit failure and timeout flags. Automatic resets cannot overwrite
the first result from a world. In this run, no second episode completed.
Execution stopped as soon as all initial worlds had a result: 117 control steps,
37.44 aggregate environment-seconds and approximately 14.9 s wall time. The hard
budget was 750 steps, or 15 simulated seconds per environment.

This supports agreement between CPU and GPU **deterministic** crouch-stage
outcomes. It suggests sensitivity to the noisy training policy as an explanation
for its zero training successes, but does not establish that explanation
causally. This diagnostic used the GPU environment's touch-sensor and geometric
stance criteria. The separate CPU validation uses exact ground/self-contact
forces and checks stability through the full requested duration. GPU episodes
terminate after the required clean hold, so this diagnostic alone does not
demonstrate 10 or 15 s of continued standing. It is not supine recovery evidence.

[summary.json](summary.json) contains all terminal records and separate first
cohort counts. [config.json](config.json) records actor identity, initial seed
assignments, environment settings and source hashes. The actor/source hashes,
unique first-world result mapping and simulation budget were verified afterward.

Reproduce from the repository root with a fresh output directory:

```bash
.venv/bin/python scripts/validate_crouch_gpu_deterministic.py \
  --actor artifacts/experiments/crouch_ppo/actor.pt \
  --output artifacts/validation/crouch_stage/gpu_deterministic_reproduction
```
