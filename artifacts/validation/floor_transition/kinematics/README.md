# Offline support-posture geometry — KINEMATIC ONLY

These are static poses calculated with forward kinematics and bounded least
squares. No physics step, policy training, simulated recovery, or runtime state
injection was performed. The PNGs show offline posed geometry, not achieved
robot behavior. Contact force, friction, actuator torque, and dynamic reachability
must be checked separately.

The existing `guarded_v2` model provides the primitive collision shapes, masses,
kinematic transforms, and URDF joint limits. Every proposed target is inside the
guarded interval `[URDF lower + 0.05, URDF upper - 0.05]`. COM is calculated from
the actual link masses. Support polygons use the proposed ground-contact lowest
points on collision primitives, with a 1 mm tolerance for coplanar box corners.

## Three proposals for physical screening

Full 31-joint target maps, base quaternion, clearance by collision shape, and
support polygons are in the matching JSON files. Arms are mirrored; joints not
listed in this table still have explicitly stored values in each JSON.

| File / proposed support | Pelvis height | Base pitch | Leg pitch targets, radians | COM margin |
| --- | ---: | ---: | --- | ---: |
| `hands_knee_region.json`: both hands + both distal thigh/knee regions | 0.35225 m | +0.98035 | Both hip −0.61018, knee 1.43930, ankle +0.02000 | 0.19081 m |
| `half_knee_region.json`: left foot + right distal thigh/knee region + both hands | 0.36355 m | +0.98493 | Left hip −2.36376, knee 2.06798, ankle −0.61417; right hip −0.71541, knee 1.53440, ankle +0.02500 | 0.07665 m |
| `seat_feet_hands.json`: pelvis + both feet + both hands behind body | 0.07740 m | −0.24544 | Both hip −2.36802, knee 2.30024, ankle +0.35016 | 0.10062 m |

All three pass this **geometric-only** screen: proposed-contact error below
1 micrometre, no detected self penetration, no ground penetration beyond
1 micrometre, and positive COM support margin. Floating base coordinates describe
the solved pose; they are not a permitted runtime teleport or controller command.

The knee-region contact is deliberately `*_hip_yaw_link_collision_0`. The model's
thigh capsule extends beyond the knee joint and reaches the ground approximately
9.5 mm before the adjacent shin capsule named `*_knee_link_collision_0`. The
initial attempts requiring the shin capsule itself to touch ground exhausted their
bounded optimization with thigh penetration and are retained as
`hands_knees.json` and `half_kneel.json`. This is a geometry-label distinction, not
an assertion that physical knee support is impossible.

## Contact order suggested by geometry

The earlier `hands_feet.json` proposal provides a bridge at pelvis height 0.29908 m
with both feet and both hands down. Removing one foot from that support polygon
leaves the COM 0.07040 m inside left-foot-plus-hands support, or 0.06903 m inside
right-foot-plus-hands support. Both hands alone give no polygon area in that
configuration. A useful physical test is therefore to maintain both hand contacts
and one foot while transitioning the other leg to the knee region, then transfer
the second leg. These margins do not validate a joint-space interpolation or
guarantee adequate hand friction/torque.

## Reproduce

From the project directory:

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python scripts/plan_support_posture.py --stage refined --render
```

The refined screen has exactly three starts and a 350-evaluation cap per start.
`summary_refined.json` records the source hash, bounds, solver counts, and runtime.
The three resulting images were visually inspected for plausible support/contact
arrangement. No live training/viewer process is created by this script.
