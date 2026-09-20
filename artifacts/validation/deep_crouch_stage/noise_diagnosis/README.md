# Paired action-noise and normalization diagnostic

The unchanged final PPO actor was run twice from exactly the same physically
generated deep-crouch state, seed **7101**. Each timed rollout was capped at 1 s.
One used deterministic actions; the other added Gaussian action noise with
standard deviation **0.005**, using the same RNG convention as the stage evaluator.
No normalization clipping, parameter changes or training were performed.

The deterministic prefix reached pelvis height **0.6489 m at 1 s**. The noisy
prefix crossed the evaluator's 0.3 m stop threshold at **0.46 s**, ending at
**0.2944 m**. Both recorded prefixes respected joint position/speed bounds, with
effort clipped by the existing controller. This short diagnostic does not itself
establish a two-second standing success or the behavior after the noisy cutoff.

The exported actor's normalizer exactly matches the frozen checkpoint. It applies
`(observation - mean) / (stored_std + 0.01)` without clipping the normalized input.
Six dimensions have stored standard deviation below 1e-6, including all three
touch flags. Twelve dimensions have standard deviation below 0.001. The epsilon
keeps the minimum denominator at 0.01 and maximum gain at 100: **zero stored
variance is not a division-by-zero bug**.

Previous-action standard deviations range from **0.001064 to 0.253928**, with
median **0.001983**; their smallest effective denominator is **0.011064**.

| Maximum absolute normalized input, before action | Deterministic prefix | Noisy prefix |
| --- | --- | --- |
| Joint position | 4.23 | 19.74 |
| Joint velocity | 9.48 | 82.87 |
| Base linear velocity | 18.82 | 51.64 |
| Base angular velocity | 12.60 | 92.04 |
| Previous actions | 3.285 | 7.600 |
| Touch flags | 0 | 100 |

In the noisy trace, the right-foot touch flag first changes from 1 to 0 at 0.20 s,
giving a normalized value of −100. At 0.22 s, both feet report no touch and the
raw actor output reaches 2.108, requiring the existing action clipping. The
normalized previous-action magnitude first exceeds 5 at 0.24 s. Previous-action
inputs therefore were not the earliest large outlier in this pair.

This sequence makes normalization of rarely varying inputs a reasonable subject
for a separate controlled experiment. It does **not prove** that normalization
caused the fall or that observation clipping fixes it. The deterministic prefix
also contains normalized inputs larger than 5, so clipping at ±5 alters inputs
used during its successful rise. Leaving normalized contact channels untouched
would retain their ±100 excursions if a foot loses touch.

[summary.json](summary.json) contains the smallest standard deviations, group
statistics, per-control-step traces, matched-prefix differences, limits, and
actor/source hashes. [paired_observations.npz](paired_observations.npz) preserves
raw/normalized observations, executed actions, and matching initial states.
Only two timed prefixes were simulated.

Reproduce from the repository root:

```bash
.venv/bin/python artifacts/validation/deep_crouch_stage/noise_diagnosis/diagnose.py
```
