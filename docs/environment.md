# Recovery environment specification

Implementation: [`env.py`](../src/x2_recovery/env.py) is the batched GPU
environment; [`common.py`](../src/x2_recovery/common.py) defines the shared
model/controller/success contract; [`runtime.py`](../src/x2_recovery/runtime.py)
is used by CPU evaluation, the viewer and ROS. The training code uses mjlab's
`Simulation` backend directly and RSL-RL's `TensorDict` vector-environment
interface. It does not depend on a second environment manager implementation.

## Physics and reset distribution

The X2 Ultra has a floating base, 31 actuated revolute joints, a flat floor and
gravity of 9.81 m/s². Physics runs at 500 Hz (2 ms), actions at 50 Hz (20 ms).
Each action is held for ten physics steps while PD torque is recomputed on
every physics step. Integration is `implicitfast`; contacts use a pyramidal
friction cone, Newton solver, 30 solver iterations, 10 line-search iterations
and tolerance 1e-6. GPU contact and constraint capacities are 128 and 512.
The complete model assumptions and per-joint limits are in [model.md](model.md).

Every reset starts from the supine model keyframe. Independently jitter base
X/Y by ±0.02 m and joint positions by ±0.005 rad within limits, lift the initial
base by another 0.005 m, and set initial velocities to zero. Hold the perturbed
joint pose under bounded PD control for one simulation second to settle on the
floor. Preserve the physically attained velocities and restart episode time
at zero. A nonfinite reset or generalized speed above 20 is rejected.

The GPU environment precomputes 16 settled states using `seed` through
`seed+15`, then samples this bank on each reset. Evaluation generates each
specified seed directly in the CPU simulator. Default evaluation seeds are
1001–1005. This finite, narrow reset distribution is a deliberate budget
simplification; it does not establish robustness to arbitrary falls, floor
properties, pushes or robot parameter uncertainty.

Neither training nor assessed evaluation uses a standing start. The standing
keyframe sets the reference joint pose and nominal pelvis height only. No
base pose is overwritten during an episode, no external lifting force is
applied, and no joint velocity is clamped to fabricate a recovery.

## Observation space

Actor and critic receive the same **106 float 32 values**, with no privileged
critic information. Joint/action order comes from the model's actuator-to-joint
mapping and is recorded in `assets/x2/model_metadata.json`.

| Indices, zero-based | Count | Quantity |
|---|---:|---|
| 0–30 | 31 | Joint angle minus nominal standing angle, radians |
| 31–61 | 31 | Joint velocity multiplied by 0.1 |
| 62–64 | 3 | Gravity direction `[0,0,-1]` rotated into the pelvis frame |
| 65–67 | 3 | Free-base world linear velocity rotated into the pelvis frame |
| 68–70 | 3 | MuJoCo free-joint angular velocity multiplied by 0.25 |
| 71 | 1 | Pelvis world Z coordinate, metres |
| 72–74 | 3 | Left foot, right foot and aggregate other-body touch above 2 N |
| 75–105 | 31 | Previous clipped action |

Values are clipped to [-10,10]. The GPU wrapper sanitizes nonfinite observation
values, but independently detects invalid physical state and ends that episode;
sanitization does not turn the episode into a success. PPO separately learns
running observation mean/variance. These statistics are embedded in the saved
TorchScript actor, so inference must not normalize observations a second time.

Touch observations in both GPU training and CPU inference use the same model
sensors. They can include self-contact. Precise CPU success evaluation instead
uses actual floor-contact pairs. Keeping these two roles separate preserves
the policy's input definition while making the final outcome test stricter
about where the robot is physically supported.

## Actions and actuator model

The policy emits 31 action values, clipped to [-1,1]. For joint `i`, let
`q0` be its nominal standing angle and `qmin/qmax` its URDF bounds. The target is

```text
q_target_i = q0_i + a_i * (qmax_i - q0_i),  if a_i >= 0
q_target_i = q0_i + a_i * (q0_i - qmin_i),  otherwise
```

This asymmetric map reaches both actual limits; a knee is not restricted to
half its bending range by centering a symmetric scale around standing.
Targets are clipped again to the joint bounds. At every physics substep:

```text
tau_i = clip(Kp_i * (q_target_i - q_i) - Kd_i * qdot_i,
             -effort_i, +effort_i)
```

If speed is at/above a positive rated limit and torque would accelerate it
further, set that torque to zero; use the symmetric rule at the negative
limit. Torque that brakes the motion remains allowed. Unit-gear MuJoCo motors
also enforce the configured effort bounds. Contacts may still back-drive a
joint beyond its rated speed; this is penalized rather than hidden by editing
simulation state. Per-joint PD gains are project assumptions, recorded alongside
the limits in model metadata.

## Reward equation and rationale

Let `dt=0.02 s`, `z` be pelvis height, `z0=0.672 m` nominal standing height,
and `c` the torso's up-axis dot world-up. Define:

```text
U = clip((c + 1) / 2, 0, 1)
H = clip(z / z0, 0, 1.1)
P = clip(z - previous_z, -0.1, 0.1) / dt
F = 1 if both foot touch forces exceed 2 N, else 0
B = 1 if every instantaneous standing check passes, else 0
S = 1 if B has held continuously for 2 seconds, else 0
E = mean((last_substep_torque_i / effort_i)^2)
A = mean((action_i - previous_action_i)^2)
V = mean(max(abs(qdot_i) / rated_speed_i - 1, 0)^2)
J = mean((max(qmin_i + 0.03 - q_i, 0)
          + max(q_i - qmax_i + 0.03, 0))^2)

r = dt * (2*H*U + U + 0.5*P + F*H*U + 4*B
          - 0.03*E - 0.03*A - 0.1*V - J)
    + 10*S - invalid
```

The height-orientation product rewards lifting an upright torso, while the
orientation-only term offers a signal before height improves. Height progress
encourages movement away from the floor. Foot support becomes valuable when
the robot is also elevated/upright. The standing term and terminal bonus target
sustained recovery. Torque, action variation, excessive speed and proximity to
joint limits discourage violent control. Hands, arms and knees may support
intermediate movements; they are not forbidden until final standing.

The dense terms are multiplied by control period so their scale approximates
reward per simulated second. The final torque sample is an inexpensive effort
proxy, not a mechanical-energy measurement. A nonfinite aggregate reward is
replaced with -1 before the additional invalid-state penalty is applied.
This original equation is `reward_version=1` in environment configuration.

These weights are a baseline design, not a demonstrated optimal reward. Sitting,
kneeling or short upward motions can improve dense return without completing
recovery. Therefore training return is always reported separately from actual
five-episode success. No claim that reward alone proves recovery is made.

### Reward version 2: targeted stability correction

The original local experiment learned to rise but continued moving after
reaching an upright, elevated posture. Its frozen policy achieved 0/5 stable
recoveries after 34,443,264 environment steps and 1,450.8 wall seconds. All five
episodes timed out; their terminal linear- and angular-speed checks failed.
The original checkpoint, rewards and evaluation remain preserved under
`artifacts/submission/local_initial`.

Version 2 retains every version-1 reward term and adds the following before
the common invalid-state handling:

```text
C = 1 if aggregate nonfoot touch force <= 2 N, else 0
G = H * U * F * C
L = norm(free-base linear velocity)
W = norm(free-base angular velocity)
T = exp(-1.5 * L^2 - 0.3 * W^2)
Q = mean((q_i - nominal_standing_q_i)^2)

r_version_2 = r_version_1 + dt * (4 * G * T - 0.15 * H * U * Q)
```

The positive `T` term supplies a smooth preference for slowing the base before
it satisfies the strict binary speed checks. Its `G` gate requires elevation,
uprightness, both-foot touch and no other touch support. A small all-joint
posture penalty discourages extreme positions as the robot becomes elevated
and upright. This is a soft preference, not a hard pose constraint; intermediate
recovery poses remain feasible and physical actuator limits still apply.

The fresh `--variant stability` experiment also reduces entropy coefficient
from 0.01 to 0.0001, starts Gaussian standard deviation at 0.4 and limits its
effective value to [0.05,0.6]. The original Gaussian had initial standard
deviation 0.6 and default bounds [1e-6,1 e 6]. Its final mean standard deviation
grew to 11.21, while raw action means were extensively clipped in inspected
states. The corrected policy starts afresh rather than inheriting those
saturated means. The completed `artifacts/runs/local_stability` experiment used
512 environments, 10,455 updates and 128,471,040 environment steps in 5,400.24 s.
Its frozen final actor passed 5/5 original recovery evaluations, in
4.28–8.86 s, and a real ROS episode reached `SUCCEEDED` at 8.56 simulated
seconds for seed 1001. These results do not establish a neutral stance.

This is a combined targeted revision, not a controlled ablation. **The final
success thresholds, floor-contact evaluation, episode duration and two-second
hold requirement are unchanged.** Version 2 does not redefine moving upright
as a successful recovery. Because the reward equation differs, total returns
between the two versions are not directly comparable; compare the fixed-seed
success checks and movement instead.

### Geometry diagnosis and reward version 3

The retained checkpoint 9251 passed 5/5 under the original recovery definition.
A separate 15 s CPU audit found an **inward-twisted, staggered, edge-supported
stance with substantial foot-to-foot bracing**. Its negative foot-centre width
did not mean the joint chains crossed: ankle and knee origins retained the
expected left/right order. Foot rotation and sole tilt shifted the collision
centres. The audit measured approximately 196 N mutual foot force, with active
foot collisions; it did not find reversed joint axes or a missing collider.
See the [full audit](../artifacts/validation/stance_geometry/README.md).

Version 3 preserves model physics, observations, action mapping and supine
resets. It retains version-2 dense rewards and adds near-standing posture
shaping. Define horizontal pelvis forward `f` by normalizing the XY projection
of the pelvis +X axis, and horizontal left `l = [-f_y, f_x]`. For the left and
right foot collision geoms:

```text
w = dot((left_foot_centre - right_foot_centre)_XY, l)
w0 = nominal standing foot-centre width, approximately 0.2743 m
heading_cos_i = dot(normalized(foot_i +X projected onto XY), f)
sole_cos_i = dot(foot_i +Z, world +Z)

phase = clip((H - 0.55) / 0.35, 0, 1)
        * clip((c - 0.5) / 0.5, 0, 1)
width_error = (w - w0) / 0.18
heading_error = mean_over_two_feet(1 - heading_cos_i)
sole_error = mean_over_two_feet(1 - sole_cos_i)
quality = exp(-width_error^2 - 2*heading_error - 2*sole_error)
narrow_or_reversed = clip((0.16 - w) / 0.18, 0, 2)
hip_yaw_error = mean_over_two_hip_yaw_joints(q_i^2)

stance_reward = dt * phase * (6*quality - 2*narrow_or_reversed
                             - heading_error - sole_error
                             - 0.5*hip_yaw_error)
```

Here `H` is normalized height and `c` is the torso up-axis dot world-up from
the original reward equation. The phase is zero for the supine pose and ramps
up near upright standing, leaving the ground-recovery sequence free to use
varied poses. The new posture terms are not gated by foot contact, so lifting
a foot cannot simply switch off those terms. Hip-yaw error is averaged over
the two relevant joints rather than diluted over all 31 joints.

Version 3 adds `stance_reward` to the version-2 dense terms and **replaces** the
terminal bonus with `10 * clean_success`. It does not also award the original
terminal bonus on every subsequent step after ordinary recovery. Common
invalid-state penalties and the 15 s timeout remain unchanged.

The completed stance experiment initialized from the frozen 9251 checkpoint,
copying actor/critic networks and observation normalizers into a new run.
Gaussian noise, optimizer, RNG stream and iteration/step counters are fresh.
It uses 512 environments, a 2,700 s (45-minute) budget, initial noise 0.10
bounded to [0.03,0.20], fixed learning rate 0.00005, PPO clip 0.1 and entropy
coefficient 0.0001. Parent path/SHA and reset/copied state categories are
recorded. It completed 4,828 updates /59,326,464 new transitions in 2,700.50 s:5/5 first recoveries,0/5 clean stance. It did not replace the selected stability policy; see RESULTS.md.

### Additional clean-stance criterion and training termination

The original recovery function in `common.py` is unchanged. The additional
stance criterion requires all original instantaneous standing checks plus:

| Added condition | Threshold |
|---|---|
| Signed foot-centre width in horizontal pelvis frame | 0.16–0.36 m inclusive |
| Each horizontal foot heading relative to pelvis | At most 25 degrees |
| Each hip-yaw joint angle | Absolute value at most 0.45 rad |
| Each sole tilt relative to world-up | At most 20 degrees |
| CPU-only exact mutual-foot support check | Sum of foot-to-foot normal forces at most 2 N |

All conditions must hold simultaneously for two continuous seconds; any missed
condition resets the clean hold counter. These are project-defined evaluation
thresholds, not vendor specifications. Fore-aft staggering is not separately
thresholded in this implementation, so even passing clean stance would not
prove every aspect of a nominal or robust posture.

GPU version-3 training uses the width, heading, hip-yaw and sole checks together
with the original touch-based standing checks. Its posture geometry is a
proxy for avoiding foot bracing; it does **not** directly measure the exact
foot-pair normal force. An original recovery alone no longer ends a version-3
training episode: the policy can continue adjusting stance until the clean
hold, invalid state or timeout. Metrics make the distinction explicit:

- `recovery/original_recovery_rate`: whether the original two-second hold was
  achieved at any time in the episode.
- `stance/clean_success_rate`: whether the added GPU posture hold completed.
- `recovery/success_rate`: in version 3, the same training termination outcome
  as the clean-success metric; in versions 1/2 it is original recovery.

CPU evaluation with `--assess-stance` additionally measures exact foot-to-foot
normal force from solver contacts. It continues after an original recovery
pass and records the first original recovery time, then stops at clean stance,
invalid state or the episode limit. The JSON report retains original
`successes` and reports `clean_stance_successes` separately. Without this flag,
evaluation and ROS retain the original recovery behavior. A GPU clean proxy
pass cannot replace this stricter CPU assessment.

## Original recovery definition and end of an episode

An episode lasts at most 15 simulation seconds (750 control steps). Versions
1/2 stop early for original recovery; version 3 uses the additional training
hold described above. All versions stop for invalid state. Invalid means any nonfinite `qpos/qvel`, any
generalized velocity magnitude above 150, or pelvis height below -0.05 m.
Horizontal posture and nonfoot support are permitted during recovery.

All instantaneous standing conditions must hold **continuously for 100 control
steps (2 seconds)**:

| Check | Threshold |
|---|---|
| Torso upright | `torso_up_dot_world_up >= cos(15 degrees)` |
| Pelvis elevated | `z >= 0.9 * 0.672 = 0.6048 m` |
| Both feet supporting | Left and right normal support forces each > 2 N |
| No other support | Aggregate nonfoot support force <= 2 N |
| Low linear motion | Free-base linear speed < 0.15 m/s |
| Low angular motion | Free-base angular speed < 0.3 rad/s |

A missed condition resets the hold counter. The force tolerance accommodates
small solver forces; it is a numerical definition of support, not a claim of
literally zero contact force. The final evaluator groups MuJoCo contact normal
forces by robot body only when the other geom is the floor. Self-contacts do
not count as ground support. Feet are `left_ankle_roll_link` and
`right_ankle_roll_link`; torso orientation uses `torso_link`.

GPU training applies the same original kinematic thresholds using model touch sensors,
which include self-contact. Its recorded success rate can therefore differ
from final CPU ground-contact evaluation. The latter determines the original
recovery count, while CPU stance assessment supplies its separate count.
CPU and GPU solvers also need not produce identical trajectories
even with matched integration and actuator settings.

The vector environment auto-resets completed slots and reports terminal
metrics before reset. Pure timeouts are marked as truncations for PPO value
bootstrap; success and invalid-state endings are terminal. ROS additionally
has an independent wall-clock watchdog covering startup and stuck simulation,
plus a configurable maximum simulated duration.

## Algorithm and reproducibility limits

PPO uses 24 transitions per environment per update, five learning epochs, four
minibatches, initial learning rate 1e-3 with KL-based adaptation toward 0.01,
discount 0.99, GAE lambda 0.95, clipping 0.2, value-loss
weight 1, clipped value loss and gradient norm limit 1. The baseline entropy
coefficient is 0.01; the stability variant uses 0.0001 as described above.
Actor and critic are
independent ELU MLPs with layers 256/128/128 and learned observation
normalization. The Gaussian actor starts at standard deviation 0.6, with a
logarithmic parameterization; the stability variant starts at 0.4 with bounds
[0.05,0.6]. Evaluation uses the deterministic action mean in both cases.
The stance variant uses the same architecture, rollout and update counts but
fixed learning rate 0.00005, PPO clip 0.1, entropy 0.0001, and initial noise
0.10 bounded to [0.03,0.20].

At 256 environments the rollout contains 6,144 transitions. Actual batch size,
seed, dependency versions, elapsed time and counters are saved with every run.
The initial configuration is a practical baseline for local compute; it is not
the result of a hyperparameter search. Independent-seed performance, robust
fall distributions, hardware transfer and success outside the defined flat
floor task have not been established.

All main local runs use 512 environments, or 12,288 transitions per PPO update.
The baseline stopped at iteration 2803; stability completed its 90-minute budget
at iteration 10455. Stance refinement has a separate authorized 45-minute
budget. GPU numerical variation
means matching seed/configuration/update budget does not promise identical
trajectories or elapsed time. Resuming restores the saved reward version so a
version-2 checkpoint is not silently trained under version-1 rewards.


## Final trajectory-limit audit

Command limits and active soft joint constraints did not keep the learned trajectory within the URDF state bounds. A physics-step audit of the selected policy found position excursions up to 0.14919 rad (8.55 degrees) and speed up to 4.20 times the rating, while commanded torque stayed within effort bounds. All five trajectories fail strict position/speed compliance. The prior posture-based 5/5 result is therefore not a claim of fully valid physical recovery. See [RESULTS.md](../RESULTS.md) and [audit](../artifacts/validation/final_reproduction/physics_step_limit_audit.json). The diagnostic code does not alter physics or clamp state.
