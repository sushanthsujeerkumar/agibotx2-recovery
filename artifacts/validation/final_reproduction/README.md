# Final reproduction and trajectory audit

The frozen selected actor has SHA256 `4d7dba4dc1f31e01ddeee3f12c7b60b93361a5708d66812585fa8e23dddf08f3`.

## Fresh-source reproduction

Source commit `5d0afda232ef50641daef30a307fe05dfc761686` was extracted with `git archive` into a separate directory, including tracked meshes and policy artifacts. The existing locked Python dependency environment was reused through a symlink, with `PYTHONPATH` explicitly pointing to the extracted source. This verifies a fresh source checkout, not a clean-machine dependency installation. Later changes add audit evidence and documentation; they do not change the policy or simulation.

- Five physics/stance tests passed. See [source import and tests](source_and_tests.log).
- Five deterministic evaluation episodes reproduced the original posture-based passes and times. See [evaluation](evaluation/summary.json).
- A fresh colcon build completed successfully. The real ROS launch accepted a Trigger request, rejected a second busy request, published 31-joint state messages and reached the original posture-based `SUCCEEDED` status. Processes shut down cleanly. See [integration report](ros_integration.json), [launch/build log](ros_launch.log) and [probe log](ros_probe.log).
- [Metadata](metadata.json) records source identity, commands and environment reuse.

## Material unresolved failure

The above success status does not certify whole-trajectory physical-limit compliance. [The authoritative audit](physics_step_limit_audit.json) samples every 2 ms physics step during the timed episode: **0/5 trajectories satisfy all URDF position and speed bounds**, despite bounded commanded effort. Maximum position excursion is 0.14919 rad (8.55 degrees); speed reaches 4.20 times the rating. The separate [50 Hz preliminary audit](limit_audit.json) undersamples these peaks.

Reproduce the physics-step audit from the project root:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_limits.py --help
```

The audit observes states without altering the simulation. See its command options for an output path. The separate [joint-stop diagnostic](joint_stop_diagnostic.json) tested three seed-1001 settings without modifying the saved model or policy. Tighter stops reduced position errors but did not resolve speed excursions; no new five-episode compliant result is claimed.

The package is a reproducible partial result. See [RESULTS.md](../../../RESULTS.md) for the additional 0/5 clean-stance outcome and the recommended local physics/controller revision before further training.
