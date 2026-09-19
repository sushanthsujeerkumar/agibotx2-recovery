# X2 Ultra model provenance and simulation assumptions

The robot is **AgiBot X2 Ultra v1.3.0** from the [official repository](https://github.com/AgibotTech/agibot_x2_urdf), pinned to commit `60c5de582c523cd188f563819e62d34cfdc3d2d0`. The source is distributed under Mulan PSL v2; a copy is included at `assets/x2/LICENSE-MulanPSL-2.0.txt`. The unmodified source URDF is included for inspection.

Run `python scripts/prepare_model.py` to fetch the pinned source and regenerate the model; use `--source /path/to/agibot_x2_urdf` for an existing checkout at that revision. The generator uses Python's standard library. Visual meshes are copied from that exact checkout. `assets/x2/model_metadata.json` records the source and generated model hashes, physical values and control ordering.

## Fidelity and deliberate simplifications

The authoritative physical source is `X2_URDF-v1.3.0/x2_ultra_simple_collision.urdf`. All 31 revolute joints, link transforms, masses, complete inertia tensors, joint bounds and rated effort/velocity limits come from this URDF. Total rigid-body mass is **41.966521 kg**. The pelvis is free in all six degrees of freedom; no external support, position teleportation or gravity compensation is applied during recovery.

The vendor MJCF was inspected but was not copied unchanged: it omits an explicit pelvis inertia and several motor/position limits differ from its URDF. A deterministic converter therefore builds MJCF directly from the URDF. Fixed sensor/camera frames and original visual geometry are retained.

Collision geometry uses explicit boxes, capsules and spheres measured against the vendor meshes. Feet retain the approximately 0.212 m by 0.120 m sole footprint and the original 0.0734 m ankle-to-sole distance. Separate primitives cover pelvis, chest, backpack, head, upper/lower legs, upper/lower arms, wrists and palms. These are simulation approximations, not exact CAD contact geometry. Two-edge kinematic ancestors are excluded from self-collision because neighboring motor housings overlap by construction; other self-collisions remain enabled. The exact exclusions are recorded in metadata.

The scene uses a flat plane, gravity 9.81 m/s², 2 ms physics steps, implicit-fast integration and a Newton contact solver. Joint damping 0.1 N m s/rad, armature 0.02 kg m², friction loss 0.1 N m, floor sliding friction 0.8, and the documented PD gains are **simulation assumptions**, not manufacturer controller calibration. No claimed hardware transfer is implied.

## Actions, limits and state ordering

The 31 torque motors have unit gears and explicit symmetric force/control bounds taken from each URDF effort limit. A position-target controller must compute PD torque, clip to these effort limits and use the velocity limits when shaping its torque envelope. MuJoCo has no hard joint velocity constraint: **do not clip simulated velocity or overwrite joint state**. Contact can briefly drive a joint beyond the motor's rated speed; log this and penalize sustained excess rather than falsifying physics.

`joint_names`, `velocity_limits`, `kp` and `kd` are aligned with `actuator_order` in metadata. `standing_qpos` and `supine_qpos` contain free-base XYZ, quaternion **WXYZ**, then those 31 joints. Joint position/velocity indices should be obtained from `actuator_trnid`, `jnt_qposadr` and `jnt_dofadr`, not assumed in ROS code.

## Reset poses and contacts

The `standing` keyframe is a slightly flexed symmetric stance (hip pitch −0.12 rad, knee +0.24 rad, ankle pitch −0.12 rad). Its pelvis starts at 0.672 m, approximately 1 mm above the sole contact height. This is a diagnostic pose; assessed recovery episodes begin supine.

The `supine` keyframe has pelvis XYZ `[0, 0, 0.195]` m and quaternion `[sqrt(0.5), 0, -sqrt(0.5), 0]`. The robot's forward/face direction points upward. It begins slightly above the floor and must be settled under dynamics with the nominal position controller before the recovery timer starts. The policy should receive the settled state. No default standing starts are used in assessed evaluation.

Every collision-carrying body has a `touch_<body_name>` scalar normal-contact-force sensor and an enclosing site. Metadata provides `foot_sensor_indices`, `nonfoot_sensor_indices`, names and bodies. These sensors include self-contact; requiring zero nonfoot touch is therefore conservative. For precise CPU evaluation, additionally inspect actual geom-floor contact pairs. The correct foot bodies are `left_ankle_roll_link` and `right_ankle_roll_link`; the torso orientation comes from `torso_link`, not the pelvis alone.

## Joint contract

| Joint | Range (rad) | Torque bound (N m) | Rated speed (rad/s) | PD Kp / Kd |
|---|---:|---:|---:|---:|
| `left_hip_pitch_joint` | -2.704 … 2.556 | 120 | 11.936 | 90 / 4 |
| `left_hip_roll_joint` | -0.235 … 2.906 | 120 | 11.936 | 90 / 4 |
| `left_hip_yaw_joint` | -1.8588 … 3.6041 | 120 | 11.936 | 90 / 4 |
| `left_knee_joint` | 0 … 2.4073 | 120 | 11.936 | 90 / 4 |
| `left_ankle_pitch_joint` | -0.803 … 0.453 | 36 | 13.087 | 40 / 2 |
| `left_ankle_roll_joint` | -0.2967 … 0.2967 | 24 | 15.077 | 40 / 2 |
| `right_hip_pitch_joint` | -2.704 … 2.556 | 120 | 11.936 | 90 / 4 |
| `right_hip_roll_joint` | -2.906 … 0.235 | 120 | 11.936 | 90 / 4 |
| `right_hip_yaw_joint` | -3.6041 … 1.8588 | 120 | 11.936 | 90 / 4 |
| `right_knee_joint` | 0 … 2.4073 | 120 | 11.936 | 90 / 4 |
| `right_ankle_pitch_joint` | -0.803 … 0.453 | 36 | 13.088 | 40 / 2 |
| `right_ankle_roll_joint` | -0.2967 … 0.2967 | 24 | 15.077 | 40 / 2 |
| `waist_yaw_joint` | -3.43 … 2.2078 | 120 | 11.936 | 90 / 4 |
| `waist_pitch_joint` | -0.314 … 0.314 | 48 | 13.088 | 60 / 3 |
| `waist_roll_joint` | -0.488 … 0.488 | 48 | 13.088 | 60 / 3 |
| `left_shoulder_pitch_joint` | -3.08 … 2.04 | 36 | 13.088 | 30 / 1.5 |
| `left_shoulder_roll_joint` | -0.061 … 3.0456 | 36 | 13.088 | 30 / 1.5 |
| `left_shoulder_yaw_joint` | -2.556 … 2.556 | 24 | 15.077 | 30 / 1.5 |
| `left_elbow_joint` | -2.3556 … 0 | 24 | 15.077 | 30 / 1.5 |
| `left_wrist_yaw_joint` | -2.556 … 2.556 | 24 | 15.077 | 30 / 1.5 |
| `left_wrist_pitch_joint` | -0.5236 … 0.5236 | 4.8 | 4.188 | 8 / 0.5 |
| `left_wrist_roll_joint` | -1.5097 … 0.724 | 4.8 | 4.188 | 8 / 0.5 |
| `right_shoulder_pitch_joint` | -3.08 … 2.04 | 36 | 13.088 | 30 / 1.5 |
| `right_shoulder_roll_joint` | -3.0456 … 0.061 | 36 | 13.088 | 30 / 1.5 |
| `right_shoulder_yaw_joint` | -2.556 … 2.556 | 24 | 15.077 | 30 / 1.5 |
| `right_elbow_joint` | -2.3556 … 0 | 24 | 15.077 | 30 / 1.5 |
| `right_wrist_yaw_joint` | -2.556 … 2.556 | 24 | 15.077 | 30 / 1.5 |
| `right_wrist_pitch_joint` | -0.5236 … 0.5236 | 4.8 | 4.188 | 8 / 0.5 |
| `right_wrist_roll_joint` | -0.724 … 1.5097 | 4.8 | 4.188 | 8 / 0.5 |
| `head_yaw_joint` | -0.349 … 0.349 | 2.6 | 6.019 | 3 / 0.3 |
| `head_pitch_joint` | -0.3838 … 0.3838 | 0.6 | 6.28 | 3 / 0.3 |

## Completed physics validation

`python scripts/prepare_model.py --source /path/to/pinned/source --validate` compiles and runs the model under MuJoCo 3.11.0. The recorded check (`assets/x2/model_validation.json`) confirms 38 position coordinates, 37 velocity coordinates, 31 actuators, correct actuator ordering and 41.966521 kg total mass. A five-second supine hold remained finite and respected all actuator force limits. It settled to approximately 0.135 m pelvis height with generalized-velocity norm 0.0208; final maximum contact depth was 0.071 mm. Maximum transient impact penetration was 7.08 mm. Initial placement has no contacts or interpenetration. The settled render is `assets/x2/model_supine.png`.

A one-second settling interval already reduced generalized-velocity norm below 0.03 in this deterministic test. Randomized resets should be checked separately. The nominal upright keyframe is geometrically plausible but **does not remain balanced under constant low-gain joint targets**; an active balance policy/controller is required. This limitation must not be confused with successful recovery, and the validation does not claim recovery success.
