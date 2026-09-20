# Fixed-support transfer attempt — KINEMATIC/STATIC ONLY

**The single bounded attempt did not produce a valid continuous path. Do not
replay these waypoints as a robot motion.** No physical integration, training, or
runtime state injection was performed.

The start is the actual physically attained hands-supported squat saved in
`actual_hand_support.json`, whose hash is recorded in `summary.json`. The offline
calculation preserved the XY locations of two near-coplanar corners per hand and
two left-heel corners. Initial ground compression was 0.2–0.45 mm; the solver
projected these anchors to the plane. The fixed contact schedule kept both hands
and the left foot, unloaded the right foot, and added right knee-region contact
at the final waypoint. No support-slip allowance was introduced.

Exactly 12 waypoints, including the source, were attempted, with at most 100
least-squares evaluations per optimized waypoint. There was one continuation
schedule and no restart or alternative IK search. It took 17.06 seconds.

All 12 isolated waypoints satisfy the geometric screen. Static LPs for the first
10 optimized intermediate poses are feasible on the fixed hand/left-foot
supports, with peak effort fractions approximately 0.61–0.64. The final endpoint
is statically feasible after adding the right knee region, at peak effort
fraction 0.20640. This establishes sampled static configurations only.

The interpolation audit checked seven interior samples in each of the 11
intervals. The first 10 intervals retain contacts within 0.563 mm and pass their
clearance checks. The final interval fails:

- Largest joint change: 1.73269 rad.
- Worst sampled floor penetration: 0.17592 m.
- Worst fixed-contact anchor displacement: 0.004984 m.
- The right knee region abruptly moves from approximately 0.249 m above ground
  to ground contact.

Consequently `complete_geometric_path` is **false**, even though
`complete_sampled_waypoints` is true. The solver found a valid endpoint on the
other side of an invalid intermediate leg configuration. Endpoint feasibility
must not be mistaken for a reachable transition. A future planner would need an
explicit collision-free swing-foot path, potentially with staged hip abduction
and yaw, or a different contact schedule; this run does not demonstrate that such
a path exists.

Full qpos, support errors, floor clearances, static forces, and gravity-offset
joint targets where feasible are retained in `waypoint_*.json` and `summary.json`.
The original nonzero source velocities were recorded but set to zero for static
calculations. Even seven passing interpolation samples would only be a sampled
geometric audit, not a continuous-time collision or dynamic stability guarantee.

To inspect the existing run again without any new IK search:

```bash
PYTHONPATH=src .venv/bin/python scripts/plan_supported_transfer.py --audit-only
```
