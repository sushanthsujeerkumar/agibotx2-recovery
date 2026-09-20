# Scripted hands-supported squat return

**This is a scripted intermediate motion, not a learned policy or recovery from the floor.** It starts from the physically generated deep-crouch reset, lowers onto both hands and feet, and returns to standing. No robot state is edited after that reset.

Five fresh validation episodes (seeds 9101–9105) all completed the full 16 seconds and ended with 4.46 seconds of clean standing. All five passed the latched joint-position, joint-speed and commanded-effort checks at every 1 ms physics step. Minimum pelvis heights were 0.260240–0.260245 m; the highest joint-speed ratio was 0.19727 of the published limit. Commanded effort reached its limit (ratio 1.0) without exceeding it. These results show local scripted feasibility in this model, not disturbance robustness or a connection from supine.

The unchanged `screen_supported_lowering.run(info, 2.32, 0.9, seed)` generated each validation result. The low target is bilateral hip pitch −2.52, knee 2.32, ankle pitch −0.7, shoulder pitch −0.9 and elbow −0.12 rad, with all other joints nominal. Smoothstep waypoint times are 0, 4, 6, 10, 13 and 16 s: deep crouch, supported squat, supported squat, deep crouch, nominal standing and nominal standing. Guarded target clipping was zero in the independently recorded re-simulation; the authored waypoints are within the guarded bounds.

At the minimum height of the recorded seed 9101 (4.02 s), exact ground normal loads were approximately 122.86/121.32 N at the left/right ankle-roll bodies and 84.36/85.26 N at the left/right wrist-roll bodies. No other body supported this motion. Final support is feet only, with pelvis height 0.668943 m, torso-up cosine 0.99736, base linear speed 0.00470 m/s and angular speed 0.00905 rad/s.

`scripted_hand_support.mp4` is an actual physical re-simulation of seed 9101: 400 frames at 25 fps, exactly 16 seconds. It does not play interpolated or injected robot states. Its numerical report exactly matches the selected source function on outcome, minimum/final height, terminal hold, complete limit report and contact-body set. `start.png`, `bottom.png` and `end.png` show the actual initial, minimum-height and final states. Captions explicitly identify the motion as scripted and not floor recovery. Visual QA moved captions outside the viewport and enlarged the camera framing; the camera-only re-simulation produced an exactly identical numerical report.

Per-episode JSON files retain the source function's data. Contacts and standing checks are evaluated at 50 Hz, with source trajectory rows retained at 10 Hz; `recorded_episode.json` retains all 800 contact/posture samples at 50 Hz, including every positive ground-contact normal force. The source contact-body display threshold is 1 N. The 16-second limit reports begin after the physical reset preparation. `config.json` records the model and source hashes and the rendering-only revision; `summary.json` records the outcomes and video hash.

Reproduce in a fresh output directory from the project root:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/validate_hand_support.py --output artifacts/validation/floor_transition/hand_support_validation_repeat
```
