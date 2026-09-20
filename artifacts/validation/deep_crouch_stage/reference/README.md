# Scripted deep-crouch reference

This is a **scripted deep-crouch-to-standing teacher**, not a learned policy or
recovery from the floor. It provides a checked reference and optional demonstration
data for the next local curriculum stage.

The five independent validation seeds, **7001–7005**, all completed the 10 s
episode and ended with **7.95 s of uninterrupted clean standing**. Pelvis height
rose from **0.47245–0.47255 m** to approximately **0.66916 m**. Every joint position,
speed and commanded-effort audit passed. Peak speed was **0.046245×** and peak
commanded effort **0.426775×** rated. There was no target clipping, nonfoot ground
support or foot-to-foot bracing.

`reset_deep_crouch` physically lowers the robot for 4 s and holds for 1 s, following
the existing 0.4 s standing settling. It uses the selected deeper commands: bilateral
hip pitch −0.9 rad, knee 1.6 rad and ankle pitch −0.7 rad, with all other joints
nominal. The reset preserves its attained positions and velocities and audits
joint limits. Its 5.4 s preparation is excluded from the timed episode.

At episode time zero, the previous action becomes 31 zeros, matching deployed
policy observations. A 4 s smoothstep interpolation returns the deep command to
nominal; nominal commands then continue through 10 s. Actions use the existing
piecewise full-range mapping, and the saved float32 action is converted back
through that mapping before simulation. No positions or velocities are clipped,
and the collision model remains unchanged.

Limits, exact ground contacts, foot bracing, and standing/posture checks run at
every 1 ms physics step. The 10,001-state monitor count includes the initial
state plus 10,000 integration steps. A reference passes only if it completes the
full duration, has at least 2 s of clean terminal standing, stays within all
trajectory limits, and requires no guarded-target clipping. Policy observation
contact features use the existing touch sensors; physical validation independently
uses exact solver contact forces.

All **12 demonstration seeds, 301–312**, also passed. Only those demonstrations
appear in [training_pairs.npz](training_pairs.npz): **6,000 samples / 12 episodes**.
The validation episodes are excluded. These starts contain the existing small
standing perturbations and do not establish broad disturbance robustness.

| Array | Shape | Meaning |
| --- | --- | --- |
| `observations` | 6000 × 106 | Float32 pre-action observation |
| `actions` | 6000 × 31 | Float32 normalized position-target action |
| `next_observations` | 6000 × 106 | Float32 post-action observation |
| `times_s` | 6000 | Pre-action simulation time |
| `episode_ids` | 6000 | Index into the per-episode arrays |
| `episode_offsets` | 13 | Half-open sample boundaries |
| `seeds` | 12 | Per-episode seeds; sample seed is `seeds[episode_ids]` |
| `initial_qpos` | 12 × 38 | Physically attained reset positions |
| `initial_qvel` | 12 × 37 | Physically attained reset velocities |
| `joint_names` | 31 | Action/actuator ordering |

Adjacent observations, action histories, timing, seed separation and source hashes
were verified in [dataset_checks.json](dataset_checks.json). Episode results and
the full schema are in [summary.json](summary.json); configuration and source
provenance are in [config.json](config.json). Dataset SHA-256:
`f7bef6cc86b64c3e90f517a0f2581526ff6337aef027081e9b6e608b2ff206dd`.

Reproduce from the repository root, choosing a fresh output directory:

```bash
.venv/bin/python scripts/validate_deep_crouch_reference.py \
  --output artifacts/validation/deep_crouch_stage/reference_reproduction
```

The script performs no learning or cloud operation.
