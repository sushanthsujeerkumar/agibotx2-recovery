# Physics v2: diagnostic and balance curriculum

This is a new local experiment on branch `physics-v2`. The earlier source ZIP and policies remain frozen. A standing-start balance pass is **not** a supine recovery.

The original policy exploits a contact strategy that exceeds the joint state bounds. A five-variant seed-1001 screen found that earlier, stiffer joint stops prevented position overshoot but did not prevent contact-driven wrist overspeed. At a representative peak, motor braking was already saturated while opposing contact torque exceeded its effort rating. Replaying the old actor with different physics is not a valid solution.

`guarded_v2` preserves the URDF masses, inertias, position ranges and rated effort/speed. It uses 1 ms steps, 50 solver iterations, a 2 ms joint-stop time constant, 0.999 stop impedance and 0.04 rad early limit engagement. An effort-limited velocity servo requests at most 40% of rated speed and slows near position boundaries. These controller/solver choices are assumptions, not measured hardware characteristics. There is no artificial clipping of robot state, extra passive damping, inertia inflation or removed collision.

[MuJoCo's solver documentation](https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters) describes soft constraints and the relationship between stop time constant and timestep. A stop margin activates the constraint early; it does not change the published joint range.

The independent monitor checks actual position, velocity and commanded effort at the attained initial state and every physics step. Violations latch for the whole episode. CPU evaluation reports posture success and validated success separately; `--require-limits` fails on the first violation. ROS now rejects violations or missing monitor evidence before announcing success. Historical ROS evidence remains historical and used the earlier posture-only behavior.

The guarded GPU training path checks limits at every physics step, rejects violating episodes and cannot award them terminal success. The legacy model/controller remains available to reproduce old artifacts. Checkpoint resume preserves physics profile and reset mode; switching either requires a separate experiment.

## Evidence

- `controller_screen.json`: five controller/stop variants on seed 1001; none makes the old actor a compliant recovery.
- `standing_screen.json`: nominal constant targets with the guarded servo sustain standing for 10 seconds; original low-gain PD falls.
- `standing_validation.json`: five additional perturbed standing starts, each maintained clean stance for 10 seconds, with zero position excursions and no speed/effort violations. This validates balance only.
- `gpu_check.json`: 16 GPU environments pass a two-second clean balance hold under zero actions; all 16 deliberately injected overspeed states are rejected. Injected states are test faults, not training/evaluation successes.
- `old_actor_strict/`: strict supine evaluation of the old actor with the guarded profile.
- 31 CPU/controller/ROS tests passed, including a transient violation latch and NumPy/Torch torque parity.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_balance.py
./EVALUATE.sh --physics-profile guarded_v2 --require-limits --assess-stance --output artifacts/local_evaluation/guarded_old_actor
./TRAIN_BALANCE.sh
```

The short balance training pilot starts from perturbed standing states, uses actual PPO, and has a 180-second training budget. It does not replace the submission policy or demonstrate recovery. Full supine evaluation remains separate and always starts supine. Model identification, broader reset validation, a feasible transition curriculum and compliant supine recovery remain unfinished.

## Pilot outcomes

Both 180-second pilots have finished. Final deterministic standing evaluation passes 5/5 for each, while separate supine recovery is 0/5 for each. Both supine test sets comply with the monitored limits but time out without rising. The first pilot did learn balance despite poor stochastic training metrics; its update-101 snapshot did not pass, but its final update-222 actor does. The second pilot discloses a nominal-controller prior. See [complete phase results](../../../PHYSICS_V2_RESULTS.md).

The two longer live viewers crashed during interpreter shutdown. The owned render thread is now joined before GLFW cleanup; the final short live test exits normally. The nominal and learned balance videos are separate, labelled evidence. No viewer or trainer remains active.
