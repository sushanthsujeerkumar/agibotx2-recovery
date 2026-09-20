# External get-up policy compatibility results

**An adapted, externally pretrained AgiBot policy completed five of five fresh
supine-to-standing tests in our local model.** Every test ran the full 15 seconds,
ended with at least two uninterrupted seconds of clean standing, and respected
the monitored joint-position, speed and commanded-effort limits at every 1 ms
physics step. This establishes a feasible floor transition in our simulation.

**This is not a policy trained by this project.** Our independently trained PPO
actor is unchanged; its most recent supine assessment remains 0/5. These external
results must not be substituted for our PPO training evidence. The adapter and
reference diagnostic are separately labelled throughout the evidence and video.

[Watch the actual 15-second re-simulation](artifacts/validation/vendor_reference/recording/adapted_vendor_recovery.mp4).
On this PC, `./RUN_VENDOR_REFERENCE_DEMO.sh` opens the simulator, runs one verified
episode and closes it. The vendor assets remain in the local work directory;
they are not included in Git or the historical submission ZIP.

## Five fresh tests

The selected adapter advances the reference timeline at 1/1.2 of its nominal
rate, and divides reference velocity commands by 1.2. It retains feedback through
the end of the episode. The first five seconds prepare the initial joint posture
through physical control from the standard settled supine reset; those five
seconds count toward the 15-second deadline. No reference base pose, joint state
or velocity is injected after reset.

| Seed | Full-duration clean recovery | Final uninterrupted clean hold | Peak speed / rating |
|---|---|---:|---:|
| 9401 | Pass | 3.14 s | 0.9382 |
| 9402 | Pass | 2.80 s | 0.7335 |
| 9403 | Pass | 2.66 s | 0.8011 |
| 9404 | Pass | 3.08 s | 0.7853 |
| 9405 | Pass | 2.70 s | 0.7468 |

All five have zero measured joint-position violations. Commanded torque reaches
but does not exceed the configured effort bounds. Clean standing includes stable
upright posture, sufficient pelvis height, both feet supporting the body, no
other ground support, separated feet, forward headings, flat soles, small hip yaw
and no foot bracing. These are five narrowly perturbed supine resets, not proof
of robustness to arbitrary falls or disturbances. Maximum speed reaches 93.82%
of a joint's rating, so the margin is limited.

[Per-episode evidence and summary](artifacts/validation/vendor_reference/slower_fresh_validation/summary.json).
The visible recording independently re-simulates seed 9401 and matches its
outcome, height, standing hold, full limit report and final stance exactly. It
contains 375 frames at 25 fps. Captioned start, intermediate and final images
are saved alongside it. The viewer closed normally with exit 0.

## What the compatibility check found

The official public MC archive contains `policy_b_lie_up.onnx`, SHA256
`3faf3df8f9616448f9ae580e22c2fa28fcb25c1eb0c9a338a0b0c5b9a520522f`.
The ONNX graph takes `obs[1,151]` and `time_step[1,1]`, and returns 29 actor
outputs plus reference joint and body states. It contains **344 reference frames**.
Its graph accepts one observation frame; the YAML's `frame_stack=5` must not be
used to construct a 755-element input. The reference outputs depend on the
clamped integer time index; the actor itself depends on the observation input.

All 29 controlled joint names map to our model; the two head joints remain at
our nominal values. Evaluating our forward kinematics at the reference poses
reproduces all 14 named reference body positions to within **0.78 micrometres**.
Thus the earlier concern about geometry mismatch was resolved for these named
frames. It does not prove identical collision geometry or dynamics.

The raw reference is not directly compliant with our pinned limits: for example,
right ankle pitch exceeds its range by 0.09043 rad and wrist roll by 0.06130 rad.
The vendor simulator XML has wider wrist-roll ranges than our pinned URDF. We
retain our existing model, ranges, actuator effort/speed ratings, contact model,
gains and `guarded_v2` controller. Requested position targets are bounded by that
existing controller; the resulting adjustment is counted in every report. This
does not clip simulated positions or velocities. These are adapted-controller
results, not a reproduction of the manufacturer's complete MC stack.

The observation is assembled from reference joint positions and velocities,
pelvis-frame gravity, scaled local angular velocity, joint offsets from the
vendor nominal pose, scaled joint velocities, and previous raw network actions.
The target is vendor nominal position plus its metadata action scale times the
network output, then the existing guarded command envelope. The declared schema,
scales and action mapping are recorded in each result. Our small PyTorch graph
evaluation was independently compared with ONNX's reference interpreter on four
fixed randomized inputs; maximum absolute output error was **4.77e-7**, and all
reference outputs matched exactly. This verifies numerical evaluation, not the
unavailable complete vendor runner semantics.

## Preserved failures and tuning

- Pure reference-angle replay at normal and half speed failed to stand. Both
  stayed within the monitored limits. Feedback is necessary for the observed
  successful behavior; the stored motion is not a working open-loop controller.
- The first actor probe stood cleanly for 2.96 seconds, then fell after blending
  to a fixed nominal standing command. Keeping actor feedback active removed
  this failure in the development seed.
- The unchanged-rate adapter passed 2/5 fresh tests (9301–9305). Three trials
  stopped around 7.41 seconds at wrist-speed violations of 1.003–1.026 times the
  rating. Their audits cover only the observed prefixes.
- On development seed 9301, a 1.1 timescale still failed; a 1.2 timescale passed.
  The latter was then frozen and tested on the five new seeds above. The two
  five-seed batches use different seeds, so they do not establish a paired
  estimate of the timing change's benefit.

All failed attempts remain under `artifacts/validation/vendor_reference/`.
The earlier 2/5 result and rejected 1.1 adjustment are not omitted or included
in the new five-seed pass count. No PPO training ran during this stage.

## Reproduce separately from our PPO

Use the locked project environment. Download only the pinned ONNX and small
configuration members from the official public archive into a separate local
directory. The downloader verifies byte ranges, ZIP integrity and the policy
hash; it does not install or execute the manufacturer's compiled MC programs.

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/fetch_vendor_reference.py --output /path/to/vendor-assets
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/vendor_reference_probe.py \
  --asset-dir /path/to/vendor-assets --mode vendor_actor --finish feedback \
  --timescale 1.2 --seed 9401 --output artifacts/local_evaluation/vendor-reference
./RUN_VENDOR_REFERENCE_DEMO.sh /path/to/vendor-assets
```

Use a fresh output directory for each evaluation. The CLI requires explicit
external assets and checks the model hash. Main training, evaluation and ROS
defaults still use this project's own policy paths; this optional diagnostic
does not silently replace them.

## Provenance and next step

The policy originates from AgiBot's
[official MC package](https://x2-aimdk.agibot.com/downloads/mc-x86-v1.0.0-20260522.zip),
described in its [simulation documentation](https://x2-aimdk.agibot.com/en/latest/sim_contents.html).
The graph's reference-output structure is consistent with the public
[BeyondMimic exporter](https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/utils/exporter.py);
that does not establish the vendor model's training provenance or license.
The previous [source-discovery note](artifacts/validation/floor_transition/official_reference_research.md)
records the earlier metadata-only inspection; this stage additionally downloaded
and evaluated the ONNX locally. A small vendor shared library was read only for
exported symbol names and never loaded or executed.

The vendor's policy redistribution and training-data reuse terms remain
unverified. No original ONNX weights, vendor configuration files, shared library
or extracted reference dataset is committed. The report, our adapter source,
numerical results and our simulation recording are preserved separately.

We now have evidence that a contact-changing recovery is physically possible in
the pinned model under our limits. The next training stage can use a clearly
attributed and appropriately reusable motion source to guide our own policy,
then evaluate that new policy independently. Until then, report external-adapter
success and our own PPO's incomplete recovery separately.
