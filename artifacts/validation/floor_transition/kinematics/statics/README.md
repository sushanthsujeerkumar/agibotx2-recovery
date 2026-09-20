# Static gravity/contact/actuator check — STATIC ONLY

All three existing IK poses admit static force balance with actuator torques
within both the published effort limits and the torque ranges attainable by the
current `guarded_torque` servo. **This does not establish a physical transition,
stable contact equilibrium, or recovery success.** No integration, IK search, or
policy training was performed.

| Pose | Minimum peak effort fraction | Largest servo target offset | Minimum remaining guarded target margin |
| --- | ---: | ---: | ---: |
| Both hands + bilateral knee regions | 0.15320 | 0.02506 rad | 0.11573 rad |
| Both hands + left foot + right knee region | 0.14528 | 0.02738 rad | 0.03896 rad |
| Seated + feet + hands behind | 0.09948 | 0.01875 rad | 0.03657 rad |

The equation uses all 37 generalized degrees of freedom, including the six
unactuated floating-base degrees:

```text
sum(J_contact_point.T @ force_world) + actuator_torque
    = qfrc_bias - qfrc_passive
```

Each actuator has its actual 1:1 moment mapping. MuJoCo supplies the point
Jacobians, gravity terms, link masses, and zero-velocity passive force. Contact
normal forces are nonnegative. The friction inequalities are
`|fx| + |fy| <= 0.8*fz`, an inscribed diamond conservative relative to a circular
Coulomb cone. There are no contact moments, consistent with `condim=3`.

Four, five, and eight point supports are used respectively. Flat or nearly
coplanar primitive corners provide multiple support points where the geometric
report supplies them; capsule/tilted-box lowest points provide single contacts.
The original submicrometre height errors are projected to the ground plane.
Total weight is 411.69157 N for the actual 41.966521 kg model. Maximum generalized
equilibrium residual is below `3e-14`; exact guarded-servo reconstruction error is
below `2e-13 Nm`.

The LP minimizes the largest actuator effort fraction. Some optimal solutions
use contacts at a friction boundary, and load distribution is not unique. The
solution is therefore a feasibility witness, not a robust target force profile
or a prediction of the compliant MuJoCo contact solver's force distribution.
Joint dry friction assistance is omitted. Contact pressure/structural strength
limits beyond the current model are not represented.

## Servo offsets

At zero velocity, before saturation, the implemented servo has effective
stiffness:

```text
damping = max(kd, effort / (0.25 * rated_velocity))
stiffness = damping * kp / kd
target = actual_joint_position + required_torque / stiffness
```

Each computed target was checked through the exact `guarded_torque` function,
including effort, velocity-target, approach-to-boundary, and target-range clamps.
The full 31-joint target maps are stored in each JSON at
`gravity_offset_targets.target_rad`; the corresponding offsets, forces, torques,
and margins are also saved.

With the command equal to the exact IK joint pose and zero joint velocities, the
servo produces zero torque; a zero-actuator-torque static LP is infeasible for
all three poses. Consequently a real servo must develop posture error or use
target offsets to support gravity. This is normal proportional control behavior,
not evidence of a controller implementation bug. The small computed offsets
also do not establish that gravity sag caused the observed failed transitions.
Contact establishment and dynamic balance remain separate problems.

## Reproduce

From the project directory:

```bash
PYTHONPATH=src .venv/bin/python artifacts/validation/floor_transition/kinematics/statics/check_static_torques.py
```

Each output records the exact input-pose hash. `summary.json` records the script
hash and zero physics steps. The normal published-effort LP, guarded-servo LP,
and zero-actuator-torque LP are retained separately for each pose.
