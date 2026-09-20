# Bounded deeper-crouch reference screen

This is **scripted reference feasibility**, not a learned movement or ground
recovery result. Eight authored poses were screened on seed 4001. The deepest
passing pose was then tested on five separate seeds, 5001–5005, with no further
candidate search.

The selected bilateral commands are hip pitch **−0.9 rad**, knee **1.6 rad**,
ankle pitch **−0.7 rad**, waist pitch **0 rad**, and nominal remaining joints.
It passed all five validation episodes, lowering the pelvis to **0.47150–0.47164 m**
and returning to clean standing. The previously validated moderate crouch reached
approximately 0.526 m. This provides a modest next curriculum step, not a solution
to getting up from the floor.

Each episode begins with the existing physically settled standing reset, followed
by 4 s smooth lowering, 1 s at the crouch command, 4 s smooth return, and 4 s at
nominal standing. No positions or velocities are overwritten after reset. The
original collision model and `guarded_v2` controller/physics remain unchanged.

Across the five selected validation episodes:

- Every joint position, speed and commanded-effort check passed at every 1 ms
  physics step; peak speed was **0.04738×** and effort **0.47361×** rated.
- No commanded target required clipping, including the controller's 0.05 rad
  joint-boundary margin.
- Both feet remained loaded throughout; minimum exact normal forces were
  approximately 145.47 N left and 144.94 N right.
- Exact nonfoot ground support and foot-to-foot bracing forces stayed at zero.
- Final clean standing persisted through the 4 s final hold. The raw sampled
  hold counter includes the return-end boundary sample and reports 4.001 s;
  the physical interval after the return ends is 4 s.

| Knee | Ankle pitch | Hip pitch | Waist pitch | Screen outcome |
| --- | --- | --- | --- | --- |
| 1.6 | −0.70 | −0.90 | 0 | Passed |
| 1.7 | −0.70 | −1.00 | 0 | Fell during lowering |
| 1.8 | −0.70 | −1.10 | 0 | Fell during lowering |
| 1.9 | −0.70 | −1.20 | 0 | Fell during lowering |
| 1.7 | −0.65 | −1.05 | 0 | Fell during lowering |
| 1.8 | −0.72 | −1.08 | 0 | Fell during lowering |
| 1.9 | −0.72 | −1.18 | +0.12 | Fell during lowering |
| 1.9 | −0.72 | −1.18 | −0.12 | Fell during lowering |

All seven failed candidates were stopped when pelvis height fell below 0.3 m;
none had a recorded joint-limit violation first. Their low reported heights are
failure states, not successful deeper crouches. Making a commanded pose legal
does not by itself establish balance.

The actual ankle pitch range is [−0.803, 0.453] rad, and guarded target range is
[−0.753, 0.403] rad. The older exploratory −0.85 rad ankle command exceeded both
the lower joint bound and the controller margin. This screen uses feasible ankle
commands and `hip + knee + ankle = 0`, while retaining all actual limits.

[summary.json](summary.json) contains every result, including exact contacts at
minimum height, end of the bottom hold and the final state. [config.json](config.json)
records all limits, candidates, timing and source hashes. These few perturbed
standing starts do not establish broad disturbance robustness.

Reproduce from the repository root with a fresh output directory:

```bash
.venv/bin/python scripts/screen_deeper_crouch.py \
  --output artifacts/validation/crouch_stage/deeper_reference_reproduction
```

The script performs no learning or cloud operation.
