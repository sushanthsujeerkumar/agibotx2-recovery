> Historical experiment record. The submitted model and current results are described in [RESULTS.md](RESULTS.md).

# Reference-guided student experiment — incomplete recovery

The local imitation experiment did **not** produce a successful recovery policy.
The final student scored **0/5** on fresh supine episodes. Three episodes exceeded
a monitored speed limit and stopped immediately; two ran the full 15 seconds
without standing. The existing PPO policy and ROS default were not replaced.

The useful software fix from this stage is **viewer isolation**: opening the
simulator can no longer change the controller's physics state or contact sensors.
The corrected student replay matches its headless evaluation exactly.

## Measured results

| Experiment | Training / evaluation scope | Result |
|---|---|---|
| Adapted external teacher, additional seeds 9501–9516 | Full physical supine recovery, same 1.2 reference timescale | **14/16**; two ankle-speed faults |
| Initial student | 5,000 supervised gradient steps; development seeds 9601–9603 | **0/3**; one wrist-speed fault |
| 50% teacher / 50% student action mixture | Six corrective-data rollouts, seeds 9701–9706 | **5/6 assisted**; not student-alone success |
| Student fitted to aggregated corrections | 10,000 additional supervised steps; same three development seeds | **0/3**, all limits valid |
| Student with previous-action inputs masked | Fresh initialization, 15,000 supervised steps; same development seeds | **0/3**, all limits valid |
| Student with finer phase features and unclamped training loss | Fresh initialization, 20,000 supervised steps; fresh seeds 9901–9905 | **0/5**, three speed faults |

All four fits together took **80.67 seconds** of local fitting time, excluding
demonstration collection, evaluation, recording, engineering and verification.
There were **50,000 supervised gradient steps and zero new PPO updates**. The
final actor starts from random weights, but its training targets come from the
external teacher; it is explicitly a teacher-derived imitation model.

The earlier external-controller **5/5** result on seeds 9401–9405 remains valid
for that batch. The new **14/16** result shows that it is not a reliability
guarantee. These are narrow randomized supine resets, not arbitrary fall poses.
The final student was fixed before testing seeds 9901–9905; those episodes were
not added to training. Development seeds were reused between experiments.

## Contract, data and validation

The experimental actor accepts the existing 106 observations plus elapsed
episode time divided by 15, and returns 31 normalized joint targets. Its final
version masks the previous-action slice and embeds 16 sine/cosine phase
frequencies. The network uses 512, 256 and 128 hidden units with ELU activations.
Mean/std normalization is fitted only to training demonstrations, with a 0.1
minimum standard deviation and contact-indicator standard deviation fixed at 1.
The deployment output is bounded even when the supervised loss uses raw outputs.

The 107-input contract is **incompatible with the default 106-input PPO/ROS
actor**. Dedicated experimental runners make that distinction explicit. No vendor
policy, reference array, or teacher query is used during student evaluation.
The student uses physical feedback and its own elapsed-time phase.

The first dataset contains 10,500 samples from the 14 complete, compliant teacher
episodes. Failed teacher episodes were excluded from this initial imitation
dataset and retained as failure evidence. Seed 9513 was requested for validation
but failed collection; the actual initial validation seeds are 9514–9516.
Corrective aggregation also retains labelled student-state prefixes up to the
first limit fault; seed 9706 is held out when fitting the aggregated datasets.
During aggregation, the teacher retains its own previous raw network output;
it acts as a stateful correcting teacher alongside the mixed action controller.

The inverse mapping into our asymmetric 31-joint action space is checked
numerically. A teacher-only reimplementation matches the preserved seed 9401
result exactly. Every evaluated student starts from `reset_cpu`, uses the same
guarded physics/controller, and is monitored at every 1 ms physics step. Success
requires a full compliant 15-second trajectory and a final clean standing hold
of at least two seconds. No physical state is injected after reset.

Small supervised losses did not translate to recovery. Masking previous targets
and a separate exact-preparation diagnostic did not resolve the problem. The
fine-phase variant also changed the training loss, so it is a combined engineering
attempt, not an isolated causal test of either change. These results do not
establish one definitive cause of the imitation failure.

## Viewer reproducibility fix

The first live replay differed from headless evaluation. Changing Torch thread
count and testing cold/warmed actors did not explain it. Headless recording
matched; the live viewer did not. Inspection of the installed MuJoCo 3.11 viewer
showed that `launch_passive` calls `mj_forward` on the supplied data. Sharing that
data lets display setup recalculate values used by contact observations.

Both the experimental recorder and the main runtime now give the viewer a
separate model and data copy. State flows to that display only. Display controls
and mouse interactions cannot change the evaluated controller dynamics. The
isolated 9904 replay matches all nine recorded comparisons, including complete
limit statistics, final stance and simulation time. It still fails recovery.
A regression test deliberately mutates the display model, state and sensors and
checks that the controller trajectory is unchanged.

The full Python/ROS test suite passed **37 tests**. A separate 10-second physical
test of the existing deep-crouch PPO actor matched every recorded state, sensor
and control value with the real viewer open versus headless execution; both
finished with valid limits and clean stance. This is a crouch test, not supine
recovery evidence. All training and viewer processes were stopped afterward.

## Saved evidence and local replay

- [Stage summary](artifacts/validation/reference_student/stage_summary.json)
- [Fresh five-episode results](artifacts/validation/reference_student/final_fresh_eval/summary.json)
- [Fit curves](artifacts/validation/reference_student/training_curves.png) — different validation sets are labelled; these are not reward curves
- [Verified student replay](artifacts/validation/reference_student/recording_isolated/student_recovery.mp4)
- [Replay checks](artifacts/validation/reference_student/recording_isolated/checks.json)
- [Main viewer parity](artifacts/validation/reference_student/main_viewer_parity.json)

On the development PC:

```bash
./RUN_REFERENCE_STUDENT_DEMO.sh
```

This shows failed student seed 9904 with explicit imitation/zero-PPO captions.
It is a diagnostic replay, not the successful external-policy demo.

The original vendor assets remain outside the repository. Derived datasets and
all four student checkpoints/exports are saved locally in the ignored directory
`artifacts/experiments/reference_student_local/`; neither they nor the vendor
weights are included in Git. Vendor policy redistribution and training-data
reuse terms have not been verified. Publicly readable SDK documentation and the
URDF license do not establish those terms for MC policy weights.

To reproduce the bounded experiment after obtaining the separately documented
vendor assets and ONNX dependencies, choose a fresh local output directory:

```bash
bash scripts/run_reference_student_experiment.sh \
  /absolute/path/to/vendor_assets \
  artifacts/experiments/reference_student_local_reproduction
```

Keep reproduction datasets and teacher-derived weights local as well. The script
runs the same four fits and physical evaluations; exact numerical reproduction
depends on the pinned environment and hardware. CPU inference uses two Torch
threads. The final actor SHA256 is
`0ef5958b2e7f37a36ad954dcf2b62292445adb43d2df76d04e8714cedfa1d5bf`.
Our existing PPO actor remains
`cb8dfa17c71b2b2e1bd2ed5e4b1282b47e92308706f0e4d8daa981dc403c9a47`.

## Next local plan

The short imitation attempt is complete and rejected as a recovery controller.
PPO should not start from this failed student merely to accumulate training time.
The next candidate should retain an explicit motion-reference and feedback
contract, then learn small bounded corrections with contact-phase training and
independent supine evaluation. External initialization or reference use must
remain clearly attributed, including in any submission.

First address the teacher's ankle-speed failures and validate that controller on
a larger predeclared local batch. Then benchmark a short reference-conditioned
training run against it. The present PC is fast enough for these experiments;
cloud scaling is not justified by this stage. A successful independently trained
supine recovery policy is still unresolved, and no ready-to-submit success claim
is made.
