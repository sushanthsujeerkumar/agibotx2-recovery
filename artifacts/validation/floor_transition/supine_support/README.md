# Supine-to-propped support screen

This bounded scripted screen found a repeatable **reclined, arm-propped endpoint**.
It did not reach upright sitting, the supported squat, standing or ground recovery.

Twelve authored candidates ran once on development seed 9001, for at most 12 s
each. All twelve completed within the recorded joint position, speed and
commanded-effort bounds, without target clipping. They used unchanged
`guarded_v2` physics and `reset_cpu`, with no state edits after the physical reset.
Nonfoot ground support was permitted for this intermediate-motion investigation.

The useful partial result is candidate 10, `wide_arm_press`. Its selected endpoint
is shown in [the recorded video](selected_recording/recorded_motion.mp4) and
[the final frame](selected_recording/frame_0600.png). The visualization replays
saved physics states in a separate rendering data object; it does not rerun or
alter the experimental dynamics.

Five fresh seeds, 9101–9105, reproduced that propped endpoint:

- Pelvis height approximately **0.09304 m**, torso-up cosine **0.5045–0.5048**,
  pelvis-up cosine approximately **0.3982**.
- Final base linear speed approximately **0.0012 m/s** and angular speed
  **0.00167 rad/s**. The development endpoint's maximum joint speed was
  **0.0456 rad/s**, so the limbs were still slowly settling.
- Exact ground normal forces were approximately **155 N at the pelvis**,
  **117 N at each wrist**, and **12 N at each foot**. The torso itself no longer
  carried ground support at that endpoint.
- All five completed 12 s with no position/speed/effort-limit violation and
  no target clipping. Commanded effort reached its configured limit.

The orientation criterion for the provisional intermediate was torso-up ≥0.5,
pelvis-up ≥0.3, total ground normal force >20 N, and low base velocities for 1 s.
The fresh episodes satisfied this combined criterion for only **0.44–0.46 s**
before the 12 s cap. Thus the formal one-second intermediate count is **0/5**,
despite the repeatable and slowly moving propped endpoints. No extra simulation
was run to extend that hold.

The original screen initially imposed a pelvis-height eligibility filter. Review
identified that this was inappropriate for sitting, where the pelvis can remain
on the floor. The saved state/contact histories were reclassified without rerunning
the original twelve episodes. Initial labels are retained in each result, and
`config.json` preserves the original screen source hashes alongside the revised
classification provenance. Low pelvis height remains a measurement, not a reason
to reject a supported sitting candidate.

The final commanded pose uses bilateral hip pitch −1.4, knee 2.1, ankle pitch
−0.7, shoulder pitch 1.6 and elbow −0.12 rad, shoulder roll ±0.6 rad, and waist
pitch +0.24 rad. **Commanded and attained poses differ:** the measured development
endpoint has left shoulder pitch 1.4832 and elbow −1.2259 rad. The elbow has not
reached its authored target under the effort-limited dynamics. Use the recorded
state, not the authored target, when reasoning about a connection to another pose.

[selected_endpoint.json](selected_endpoint.json) stores the attained qpos, qvel,
controls, per-joint values and exact contact measurements. [summary.json](summary.json)
contains all results; each `screen_*.json` includes 50 Hz contact/orientation
history, and `screen_*_states.npz` preserves the corresponding recorded states.
Joint position, speed and effort monitoring ran at every 1 ms physics step;
contact and body-orientation diagnostics ran every 20 ms.

Reproduce the final protocol from the repository root in a fresh output directory:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/screen_supine_support.py \
  --output artifacts/validation/floor_transition/supine_support_reproduction
```

The stage uses no training or cloud computation. None of these observations is a
standing/recovery success claim.
