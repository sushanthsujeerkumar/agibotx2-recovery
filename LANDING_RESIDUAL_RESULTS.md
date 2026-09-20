# External teacher with trained PPO landing corrections

The new composite controller passes **5/5 fresh assessment episodes** and
**24/25 across the complete fresh batch**. The unchanged external teacher also
passes **24/25 on exactly the same seeds**, with identical success/failure pairs.
**This experiment does not demonstrate a recovery-success improvement from PPO.**
The external policy supplies recovery; our newly trained policy supplies small
ankle-target corrections. Our independent PPO and imitation student retain their
previous **0/5** full-supine results.

The composite is now connected to the ROS service. A fresh build and real
simulation verified successful recovery, immediate acceptance, busy rejection,
750 joint-state frames, and both wall-clock and simulation-duration timeouts.
The Python/ROS suite passes **46 tests**; the fresh colcon test run passes **27**.

## Diagnosis and bounded experiment

The previous teacher failures on seeds 9503 and 9513 occurred at 10.034 seconds.
At touchdown, the left ankle reversed rapidly and exceeded its speed rating
while the effort-limited controller was already applying opposing torque. On
9503, left-foot normal force reached about 772 N at the sampled failure instant.
This is consistent with a contact-driven excursion; command bounds alone do not
guarantee state-speed compliance. No measured state was clipped or overwritten.

A six-value global timing screen on four development cases produced 2/4 at the
existing 1.20 timescale, 1/4 at 1.18 and 1.24, and 0/4 at 1.22, 1.26 and 1.28.
Slowing the whole motion introduced other wrist faults. The 1.20 timing was kept.
This is an engineering screen, not independent validation.

The new PPO residual adjusts only left/right ankle-pitch targets. Its requested
correction is limited to **±0.04 radians**, multiplied by a smooth window that
is zero outside **9–11 seconds**, rises over 9–9.3 s, and falls over 10.7–11 s.
The external teacher runs throughout. Robot geometry, published ranges, guarded
controller, 1 ms physics step, physical reset and 15-second deadline are unchanged.

| Setting | Value |
|---|---|
| Observation | Shared first 75 physical observations, elapsed/15, two normalized teacher ankle targets: 78 values |
| Action | Two Gaussian raw corrections; tanh bounds them before scaling/windowing |
| Actor / critic | Separate 64–64 ELU networks; zero actor output layer initially |
| PPO | Clip 0.1, learning rate 0.0001, 4 epochs, minibatches of 512, gradient norm cap 1 |
| Discount / GAE | 0.995 / 0.95; terminal value zero at fault or finite task horizon |
| Exploration | Fixed Gaussian standard deviation 0.2; no entropy bonus |
| Training | 6 updates, 8 episodes/update, 4 local CPU simulation workers |
| Collected transitions | 32,268 across the complete campaign |
| Measured campaign time | 183.14 s, including development validation; setup and later fresh testing excluded |
| Selected checkpoint | Update 4, after 21,257 transitions; campaign's update 6 is also saved |

Actor loss uses decisions only inside the landing window. The critic uses the
whole episode. At each 20 ms control step, reward is 0.02 times normalized pelvis
height (clipped 0–1.2), positive torso uprightness, 0.5 for both feet supported,
and 2 for clean stance, minus 0.5 times squared speed excess above 0.75 of rating
and a small 0.001 squared-action penalty inside the window. A trajectory fault
adds −5; a complete successful episode adds +5. See the source for the exact
per-physics-step peak speed calculation and success specification.

Each training update uses four declared development cases (9503, 9513, 9501,
9401) and four new training seeds. The validation set is 11201–11205. Updates
0, 2, 4 and 6 all scored 4/5 there. Update 4 was chosen by mean validation reward
among the tied success counts, before inspecting the fresh batch. Its actor
weights changed by L2 norm 0.20783 from initialization: these were actual PPO
updates, not a renamed pretrained checkpoint. The unchanged teacher has zero
residual and serves as the paired baseline.

## Fresh results and limits

| Seeds | Teacher alone | Teacher + selected PPO residual |
|---|---:|---:|
| 11301–11305 | 5/5 | 5/5 |
| 11401–11420 | 19/20 | 19/20 |
| Complete fresh batch | **24/25** | **24/25** |

Both failed seed 11404 at 9.157 s due to right-wrist speed: approximately 1.102
times its rating. Those failed trajectories are retained and rejected. Successful
residual episodes maintained final clean stance for at least 2.46 s, with zero
position violations, peak speed at most 0.97754 of rating, and bounded effort.
The largest requested correction among those successes was 0.000972 rad.

Every success requires the **full 15 seconds** to remain within monitored joint
position/speed/effort limits and finish with at least two continuous seconds of
upright, stable, two-foot support and clean foot posture, without other-body
support or foot bracing. Passing a posture briefly does not erase a limit fault.
These seeds provide narrow reset variation, not arbitrary-fall robustness.
The known ankle failure is not established as fixed; the validation ankle fault
persisted. Additional cloud compute is not justified by a demonstrated gain here.

## Source, checkpoints and reproduction

The external policy is pinned to SHA256
`3faf3df8f9616448f9ae580e22c2fa28fcb25c1eb0c9a338a0b0c5b9a520522f`
from [AgiBot's MC package](https://x2-aimdk.agibot.com/downloads/mc-x86-v1.0.0-20260522.zip).
Its weights/configuration are not in Git or the submission archive. Vendor reuse
and redistribution terms remain unverified; the URDF license does not establish
the license of those MC assets. External assets are loaded explicitly and are
required at inference. This is not a from-scratch recovery-learning comparison.

All seven of **our** small residual checkpoints (updates 0–6), exports, reward
logs and [reward plot](artifacts/experiments/landing_residual_ppo/reward_curve.png)
are saved in `artifacts/experiments/landing_residual_ppo/`. These contain our
randomly initialized/then PPO-trained networks, not the vendor network or motion
arrays. The selected checkpoint SHA256 is
`74d62dcdd88f00f268bc17bcfb742c5f4aee161709aaaf658210a762c336d033`.
TorchScript exports return **normalized** tanh corrections; their JSON sidecars
specify the 0.04 rad scale and external landing window. They must not be supplied
to the historical 106-input ROS policy interface. Resume is not implemented for
this small bounded experiment; rerun the seeded command to reproduce training.

```bash
uv sync --frozen --extra reference --python /usr/bin/python3
PYTHONPATH=src .venv/bin/python scripts/fetch_vendor_reference.py --output .cache/vendor_reference
PYTHONPATH=src .venv/bin/python scripts/train_landing_residual.py train \
  --asset-dir .cache/vendor_reference --output artifacts/local_evaluation/new_residual_run \
  --updates 6 --workers 4
```

The fetch command requires a fresh output directory; skip it if the pinned assets
are already present. [START_HERE.md](START_HERE.md) contains the demo/ROS commands.
Physics execution is shared between training, comparison, recording and ROS.
The display has its own model/data copies. Library migration, live recording
and ROS bridge checks match saved headless results; no telemetry is synthesized.

## Evidence

- [Paired fresh results](artifacts/validation/landing_residual/paired_fresh/summary.json)
- [Verified fresh replay](artifacts/validation/landing_residual/recording/teacher_plus_ppo_residual.mp4)
- [Real ROS success](artifacts/validation/landing_residual/ros/success.json), [wall timeout](artifacts/validation/landing_residual/ros/wall_timeout.json), [simulation timeout](artifacts/validation/landing_residual/ros/sim_timeout.json)
- [Fresh colcon results](artifacts/validation/landing_residual/ros/test_result.log)
- [Training selection](artifacts/experiments/landing_residual_ppo/selected.json)

The first colcon validation attempt ran zero tests because the simulator venv's
newer setuptools hid the package's test declaration. That failure is preserved.
The corrected validator builds/tests with the Ubuntu ROS environment before
adding simulator dependencies; its fresh run executes all 27 ROS tests.

The current result is a working, attributed reference-controller integration
plus an honestly evaluated PPO correction experiment. The next research work
would target contact modeling/landing robustness and broader reset conditions.
Further training is stopped so the complete assessment can be reviewed.
