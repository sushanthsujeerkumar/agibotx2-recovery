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

Actor and critic receive the same **106 float32 values**, with no privileged
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
deviation 0.6 and default bounds [1e-6,1e6]. Its final mean standard deviation
grew to 11.21, while raw action means were extensively clipped in inspected
states. The corrected policy starts afresh rather than inheriting those
saturated means. Its output directory is `artifacts/runs/local_stability`,
with 512 environments and a 5,400 s (90-minute) local budget. It is running;
its final outcome is pending.

This is a combined targeted revision, not a controlled ablation. **The final
success thresholds, floor-contact evaluation, episode duration and two-second
hold requirement are unchanged.** Version 2 does not redefine moving upright
as a successful recovery. Because the reward equation differs, total returns
between the two versions are not directly comparable; compare the fixed-seed
success checks and movement instead.

## End of an episode and success measurement

An episode lasts at most 15 simulation seconds (750 control steps). Stop early
for success or invalid state. Invalid means any nonfinite `qpos/qvel`, any
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

GPU training applies the same kinematic thresholds using model touch sensors,
which include self-contact. Its recorded success rate can therefore differ
from final CPU ground-contact evaluation. The latter determines the submitted
success count. CPU and GPU solvers also need not produce identical trajectories
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

At 256 environments the rollout contains 6,144 transitions. Actual batch size,
seed, dependency versions, elapsed time and counters are saved with every run.
The initial configuration is a practical baseline for local compute; it is not
the result of a hyperparameter search. Independent-seed performance, robust
fall distributions, hardware transfer and success outside the defined flat
floor task have not been established.

Both recorded/planned main local runs use 512 environments, or 12,288
transitions per PPO update. The baseline stopped at iteration 2803; the
corrected run is limited by its wall-clock budget. GPU numerical variation
means matching seed/configuration/update budget does not promise identical
trajectories or elapsed time. Resuming restores the saved reward version so a
version-2 checkpoint is not silently trained under version-1 rewards.
