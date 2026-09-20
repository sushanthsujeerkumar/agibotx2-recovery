# Floor-transition feasibility results

The new result is a **scripted hands-supported squat and return, 5/5 fresh tests**.
It starts from the physically generated deep crouch, lowers to a pelvis height of
0.26024 m with both hands and feet supporting the body, then returns to clean
standing. It is not learned motion and does not start from the floor. The retained
PPO actor and its previously measured **0/5 supine recoveries** are unchanged.

[Watch the 16-second physical re-simulation](artifacts/validation/floor_transition/hand_support_validation/scripted_hand_support.mp4)
or run `./RUN_HAND_SUPPORT_DEMO.sh` from the repository root. The viewer closes
after one episode; closing its window also stops it.

## What was established

Fresh seeds 9101–9105 all completed 16 seconds, ending with 4.46 seconds of clean
standing. Every 1 ms physics step passed the latched joint-position, joint-speed
and commanded-effort checks. Peak joint speed was 0.19727 of the published rating;
commanded effort reached, but did not exceed, its configured limit. The five
minimum pelvis heights span 0.260240–0.260245 m. These narrowly varied reset seeds
do not establish disturbance robustness.

The robot uses unchanged `guarded_v2` physics. Commands pass through the existing
effort-limited controller. No robot state is edited after physical reset. The
16-second checks begin after reset preparation. At the lowest recorded position,
each foot carries about 121–123 N and each wrist about 84–85 N. Final support is
feet only; the stance check requires separated feet, nearly forward headings,
flat soles, small hip yaw and no foot bracing as well as upright, stable standing.

The actual video re-simulation matches the numerical validation on outcome,
height, standing hold, limit report and ground-contact bodies. It uses physical
simulation rather than interpolated robot states. Captions identify its scope.
[Complete five-seed evidence and recording provenance](artifacts/validation/floor_transition/hand_support_validation/README.md).

## Why floor recovery is still unresolved

| Bounded experiment | Measured result | Interpretation |
|---|---|---|
| Lower-and-return screen | 11 valid targets returned; one target rejected before simulation | A lower supported squat is feasible |
| Supine arm-support screen | 12 candidates; best reclined endpoint repeated across five fresh seeds | Partial floor motion only; formal 1 s intermediate hold **0/5** within the 12 s cap |
| Symmetric kneeling commands | 0/12 standing returns; all timed trajectories within monitored limits | Joint interpolation loses support |
| Three full-body IK proposals, each from crouch and supine | 0/6 standing returns; limits respected | Valid isolated poses do not establish reachable transitions |
| Static gravity-offset commands | 0/2 standing returns; limits respected | Accounting for gravity at an endpoint does not fix the transfer |
| One 12-waypoint fixed-support plan | Isolated poses pass; final interpolation penetrates the floor by 17.59 cm | Rejected before physical replay; no continuous path established |

The best supine attempt reaches a reclined, arm-propped endpoint with pelvis
height approximately 0.093 m and torso-up cosine about 0.505. It is neither upright
sitting nor a supported squat. The elbow remains bent despite a near-straight
command, with effort saturation. The formal one-second intermediate criterion
holds only 0.44–0.46 seconds before the cap. See the
[supine evidence](artifacts/validation/floor_transition/supine_support/README.md).

Static contact-force calculations found three plausible seated/kneeling poses
within available actuator torque. Physical attempts still fall backward during
the change of support. A final geometric continuation kept the hands and left
foot fixed, but its last interval passed the moving foot through the floor. That
path was rejected. See the [static force calculation](artifacts/validation/floor_transition/kinematics/statics/README.md)
and [failed path audit](artifacts/validation/floor_transition/kinematics/path/README.md).
Kinematic images and offline force solutions are explicitly labelled; they are
not simulation or recovery successes.

## Next method

Resolve the contact-changing motion before another PPO campaign: obtain or plan
a continuous, collision-free floor-to-squat reference, then validate it through
the same floating-base dynamics and motor limits. A staged swing-foot path or a
different hand/foot contact schedule is needed for the attempted kneeling route.
Only a successful physical reference should become new imitation data or a PPO
curriculum stage. More GPU workers do not fix an invalid reference.

AgiBot documents an official simulation/MC get-up pipeline, which is a potential
source of motion-phase guidance, subject to availability and model compatibility.
It is separate from their example ONNX dance deployment and from our PPO policy.
[Official simulation overview](https://x2-aimdk.agibot.com/en/latest/sim_rl/index.html).
No manufacturer controller has been installed, imported or evaluated here.

## Reproduction and evidence boundaries

Run from this repository with its existing locked `.venv`; use fresh directories
so original measurements are preserved:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
MUJOCO_GL=egl .venv/bin/python scripts/validate_hand_support.py \
  --output artifacts/validation/floor_transition/hand_support_validation_repeat
MUJOCO_GL=egl .venv/bin/python scripts/screen_supine_support.py \
  --output artifacts/validation/floor_transition/supine_support_repeat
.venv/bin/python scripts/validate_ik_bridges.py \
  --output artifacts/validation/floor_transition/ik_bridges_repeat
.venv/bin/python scripts/validate_gravity_offsets.py
```

The IK replay scripts read the preserved proposal JSON files and command their
joint targets; they never inject the proposed floating-base states. Their 28/34 s
diagnostic timings exceed the standard recovery evaluation duration and must not
be reported as standard assessment episodes. The original source snapshots are
preserved where diagnostics were subsequently cleaned up or extended. The live
reference log records a clean viewer run and exit 0. No PPO training or cloud
compute was used in this stage; existing policies and the historical ZIP remain
unchanged.
