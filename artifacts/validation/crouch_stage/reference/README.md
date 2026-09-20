# Scripted crouch-to-standing reference

This is a checked **scripted teacher**, not a learned policy or recovery from the floor.
It establishes that the moderate crouch-to-standing curriculum is physically attainable
under the project's `guarded_v2` simulation assumptions.

The five independent validation seeds, 3001–3005, all completed the 10 s episode,
held clean standing continuously for 8.666 s, finished stable, and stayed within all
joint position, speed and commanded-effort bounds. Peak speed was 0.05288 times
the rated limit and peak commanded effort was 0.36829 times the rated limit.
Pelvis height rose from approximately 0.526 m to 0.669 m. No limit violations or
foot bracing occurred. These seeds differ only by the small perturbations in
`reset_balance`; this is not evidence of broad disturbance robustness.

`reset_crouch` physically lowers the robot from its perturbed standing reset over
3.5 s, following the existing 0.4 s standing settling. That reset also audits
joint limits. At episode time zero, the robot retains the attained positions and
velocities. Its previous action is reset to 31 zeros, matching policy deployment.
The teacher commands bilateral hip pitch −0.7 rad, knee 1.4 rad and ankle pitch
−0.7 rad, with other joints nominal. A 3 s smoothstep interpolation returns those
commands to nominal and then holds through 10 s. Neither joint states nor
contacts are overwritten. All collisions remain enabled.

The position, speed and effort monitor, exact ground-force checks, and clean
stance checks run every 1 ms physics step. The reported monitor count of 10,001
includes the initial state plus 10,000 integration steps. Clean standing requires
the original upright, height, two-foot support and low-speed criteria, together
with the existing separated-feet, foot-heading, hip-yaw, flat-sole and no-bracing
checks. A valid reference must complete the full episode, hold clean standing
for at least 2 s, remain clean at the end and have no latched limit violation.

The distinct training seeds 101–116 all passed. Their 8,000 samples are in
[training_pairs.npz](training_pairs.npz); validation trajectories are never included.
The action target is converted to the existing piecewise full-range normalized
action, then converted back through `ModelInfo.targets` before simulation, so the
saved float32 action is exactly the command used to generate its next observation.
Policy observations use `sensor_forces`, matching deployment; exact solver forces
are used independently for physical validation.

| Array | Shape | Meaning |
| --- | --- | --- |
| `observations` | 8000 × 106 | Float32 pre-action observations |
| `actions` | 8000 × 31 | Float32 normalized position-target commands held for 20 ms |
| `next_observations` | 8000 × 106 | Float32 post-action observations |
| `times_s` | 8000 | Float64 pre-action simulation time |
| `episode_ids` | 8000 | Index into the per-episode arrays |
| `episode_offsets` | 17 | Half-open sample boundaries |
| `seeds` | 16 | Per-episode seeds; sample seed is `seeds[episode_ids]` |
| `initial_qpos` | 16 × 38 | Physically attained reset positions |
| `initial_qvel` | 16 × 37 | Physically attained reset velocities |
| `joint_names` | 31 | Action/actuator order |

Adjacent observation consistency, prior-action alignment, episode boundaries,
seed separation and sample timing were checked in [dataset_checks.json](dataset_checks.json).
Full episode results and schema are in [summary.json](summary.json), with model,
source and configuration provenance in [config.json](config.json). The dataset
SHA-256 is `d6602f73d361d422998287bb92eb8d779998338c51be38d3c7bd2aaef00c3281`.

Reproduce from the repository root:

```bash
.venv/bin/python scripts/validate_crouch_stage.py
```

The script uses CPU MuJoCo and performs no RL training or cloud operation.
