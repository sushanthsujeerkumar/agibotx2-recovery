# Recovery environment

The final configuration is `FullRecoveryEnv` with `physics_profile=guarded_v2`, `reset_mode=supine` and `reward_version=3`. CPU evaluation and ROS use `RecoveryRuntime(controller="full_recovery")`. Other profiles in the source reproduce historical experiments.

## Model and reset

The robot has a floating base and 31 actuated joints on a flat floor. Gravity is 9.81 m/s². Physics runs at **1,000 Hz**, control at **50 Hz** (20 physics steps per action), with `implicitfast` integration, a Newton solver, 50 iterations, 10 line-search iterations and tolerance 1e-6. Collision geometry, inertias and model assumptions are described in [model.md](model.md).

A reset starts from the supine keyframe, without floor intersection. Base X/Y receive independent ±0.02 m perturbations and joint positions ±0.005 rad, clipped within limits. The base is lifted another 0.005 m and the pose is held under bounded control for one simulated second to settle. The physically attained state is retained, then episode time starts at zero. Numerical soft-contact penetration can occur while settling; the model validation records it. No state is overwritten, no lifting force is applied and no velocity is clamped during recovery.

GPU training samples a bank of 16 physically settled resets generated from the training seed. CPU evaluation generates the specified seeds independently. This is a narrow supine distribution, not arbitrary falls. The final policy's evaluation always starts on the floor; supported crouch resets belong only to earlier curriculum experiments.

## Observations

Actor and critic receive the same 107 float32 values. Joint order follows the actuator-to-joint mapping in `assets/x2/model_metadata.json`.

| Indices | Values |
|---|---|
| 0–30 | Joint angle minus nominal angle, radians |
| 31–61 | Joint velocity × 0.1 |
| 62–64 | Gravity direction in the pelvis frame |
| 65–67 | Base linear velocity in the pelvis frame |
| 68–70 | Free-joint angular velocity × 0.25 |
| 71 | Pelvis height, metres |
| 72–74 | Left foot, right foot and aggregate nonfoot touch above 2 N |
| 75–105 | Previous applied normalized action |
| 106 | Elapsed episode time / 15, clipped to [0,1] |

Physical observations are clipped to [-10,10]. Training sanitizes nonfinite values but separately terminates invalid physical states. Touch sensors can include self-contact. Final CPU success uses actual floor-contact pairs instead. Normalization is included in the actor export; inference must not normalize a second time.

## Actions and torque limits

The frozen prior plus PPO feedback returns 31 values in [-1,1]. A positive action scales the distance from nominal to the joint's upper bound; a negative action scales the distance from nominal to its lower bound. The guarded controller then runs this servo at every physics step:

```text
q_target = clip(mapped_action, q_min + 0.05, q_max - 0.05)
v_target = clip((Kp/Kd) * (q_target - q), -0.4*v_rated, 0.4*v_rated)
v_target = clip(v_target,
                -max(0, q - q_min - 0.025)/0.1,
                 max(0, q_max - 0.025 - q)/0.1)
damping = max(Kd, effort_limit/(0.25*v_rated))
torque = clip(damping * (v_target - qdot), -effort_limit, effort_limit)
```

The URDF limits are unchanged. Target margins, velocity shaping and stronger joint constraints are simulation/controller choices, not calibrated hardware settings. Joint constraint margin is 0.04 rad, `solref=[0.002,1]` and `solimp=[0.999,0.999,0.001,0.5,2]`. Contact impulses can still exceed limits, so every physics step is audited. Any excursion beyond 1e-6 rad in position, or a speed/effort ratio above 1+1e-6, fails the entire assessed episode. Later recovery cannot erase a violation.

## Reward, including every weight

The final run uses the following version-3 reward. All dense terms are multiplied by `dt=0.02`. `mean` is across joints unless stated otherwise.

```text
c = torso up-axis dot world-up
H = clip(pelvis_height / 0.672, 0, 1.1)
U = clip((c + 1)/2, 0, 1)
P = clip(height - previous_height, -0.1, 0.1)/dt
F = 1 if both foot touch sensors exceed 2 N, else 0
C = 1 if aggregate nonfoot touch <= 2 N, else 0
B = 1 if all instantaneous standing checks pass, else 0
S = 1 if clean standing has held for 2 seconds with no limit failure, else 0
E = mean((last_substep_torque / effort_limit)^2)
A = mean((action - previous_action)^2)
V = mean(max(abs(qdot)/v_rated - 1, 0)^2)
J = mean((max(q_min + 0.03 - q, 0) + max(q - q_max + 0.03, 0))^2)
L = norm(base linear velocity)
W = norm(base angular velocity)
Q = mean((q - nominal_q)^2)
G = H*U*F*C
T = exp(-1.5*L^2 - 0.3*W^2)

phase_gate = clip((H - 0.55)/0.35, 0, 1) * clip((c - 0.5)/0.5, 0, 1)
width_error = (signed_foot_width - nominal_foot_width)/0.18
heading_error = mean(1 - cos(foot_heading_relative_to_pelvis)) over both feet
sole_error = mean(1 - sole_up_dot_world_up) over both feet
quality = exp(-width_error^2 - 2*heading_error - 2*sole_error)
narrow = clip((0.16 - signed_foot_width)/0.18, 0, 2)
yaw_error = mean(hip_yaw^2) over both hips

r = dt * (2*H*U + U + 0.5*P + F*H*U + 4*B
          - 0.03*E - 0.03*A - 0.1*V - J
          + 4*G*T - 0.15*H*U*Q
          + phase_gate*(6*quality - 2*narrow - heading_error - sole_error - 0.5*yaw_error))
    + 10*S - invalid
```

Nominal foot width is 0.27430018 m. A nonfinite aggregate reward becomes -1 before the invalid-state penalty. `invalid` includes nonfinite physics, generalized speed above 150, pelvis below -0.05 m, or a latched trajectory-limit failure.

Height and orientation give a signal before the robot stands; progress rewards lifting. Foot support and low base speed reward a stable finish. Stance terms discourage the crossed-foot solution seen in early training and only turn on near upright posture, leaving floor transitions free to use hands and knees. Effort, abrupt actions, speed excess and boundary proximity penalize aggressive motion. These weights were chosen during development, not established as optimal. Torque squared is an effort proxy, not measured energy.

## Success and termination

Every instantaneous standing check must pass continuously for at least two seconds:

- Pelvis height at least 0.6048 m (90% of nominal).
- Torso tilt at most 15°; base linear speed below 0.15 m/s and angular speed below 0.3 rad/s.
- Both feet carry more than 2 N normal floor force; all other bodies together carry at most 2 N from the floor.
- Signed lateral foot separation 0.16–0.36 m; each foot heading within 25° of pelvis forward; sole tilt at most 20°; each hip yaw within ±0.45 rad.
- Mutual foot bracing force at most 2 N, checked from actual contacts in CPU evaluation.

Stance and support are sampled at 50 Hz. Joint position, velocity and effort are audited at 1,000 Hz from the initial settled state through the whole attempt. CPU evaluation and the ROS final-policy mode run the complete 15 seconds and require the two-second clean hold at the end. This prevents an early transient success from hiding a later fall.

Training uses touch sensors and geometry proxies for support/bracing, and terminates on success, invalid state or the 750-step limit. Pure timeouts bootstrap the PPO value estimate. Final CPU evaluation is stricter about actual support contacts and full-horizon completion; its success count is the reported result. ROS also has an independent wall-clock watchdog, including simulator startup.
