# ROS 2 interface and validation

The `x2_recovery_ros` package contains a recovery supervisor and a telemetry node.
The supervisor's service callback queues an episode and returns; its single-threaded
executor sends the response before the next timer callback starts the simulator.
The busy flag is reserved in the callback, including the short queued interval.

Each accepted request creates one subprocess using the shared
`x2_recovery.runtime.RecoveryRuntime`. The worker reads actual joint positions,
joint names and simulation time after every control step. The parent publishes
those values unchanged as `sensor_msgs/msg/JointState`; timestamps are simulator
time since reset, not wall-clock timestamps. A new episode resets this time.
The two ROS nodes do not enable `use_sim_time`, so service/watchdog timers remain
independent of simulation time. No `/clock` topic is required by this package.

The worker forwards the runtime's stable-standing success check. Invalid states,
exceptions, worker death, a configured simulation-duration limit, or the separate
wall-clock watchdog produce `FAILED`. The wall watchdog includes model and policy
loading, remains active if a simulator call hangs, and terminates the worker.
All episodes use the real simulator; there is no synthetic joint-state fallback.

## Build and run

From the repository root, first install the shared simulator project according to
the main README. ROS 2 Jazzy uses Python 3.12. Use the same Python version for the
project virtual environment and expose its packages to ROS before launching.

The wrapper performs the build, sources ROS/the workspace, and adds the simulator
environment to `PYTHONPATH` automatically:

```bash
scripts/ros_launch.sh controller:=scripted render:=true
```

Equivalent manual commands:

```bash
source /opt/ros/jazzy/setup.bash
export PYTHONPATH="$PWD/src:$PWD/.venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd ros2_ws
colcon build --symlink-install --packages-select x2_recovery_ros
source install/setup.bash
ros2 launch x2_recovery_ros recovery.launch.py controller:=scripted render:=true
```

The command starts both required nodes. Headless runs use `render:=false`.
For policy execution add `controller:=policy checkpoint:=/absolute/path/to/checkpoint`.
`timeout_s` defaults to 60 wall seconds and `max_sim_duration_s` to 20 simulation
seconds. `realtime:=true` paces simulation for visible playback; false runs as fast
as the single simulator allows. The default seed is 0.

In another terminal, source ROS and the workspace as above, then run:

```bash
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
ros2 topic echo /x2/recovery_status std_msgs/msg/String
ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState --once
```

The first response must be `success=true`, the second `success=false` while busy.
Status is retained for late subscribers and also published every second. The
telemetry node logs status and the first actual joint's position every second.
It clears its cached joint when a new episode starts.

For an unsuccessful timeout check use a short wall timeout and a longer simulation
limit (allow enough startup time on the test machine):

```bash
ros2 launch x2_recovery_ros recovery.launch.py controller:=scripted render:=false timeout_s:=8.0 max_sim_duration_s:=60.0
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
```

## Automated checks

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon test --packages-select x2_recovery_ros --event-handlers console_direct+
colcon test-result --verbose
```

Unit tests explicitly substitute a small runtime in test code only, verifying
acceptance-before-execution, rejection while queued, deadlines, errors, invalid
frames, timestamp conversion, and closing the simulator. These unit tests do not
constitute simulator-integration evidence. Real simulator evidence is recorded
separately below. Reproduce the full fresh-build and live integration check with:

```bash
scripts/validate_ros.sh
```

The script uses a separate ROS domain (71 by default), a headless MuJoCo worker,
12 wall seconds per attempt and a 60-second simulation safeguard. It creates a
new build/install tree and evidence directory each run. Set `X2_ROS_DOMAIN_ID`,
`X2_ROS_TIMEOUT_SECONDS`, or `X2_ROS_EVIDENCE_DIR` to override these choices. It
does not interact with a separately opened viewer.

## Recorded checks, 2026-09-20

- A fresh build into new `validation/final_fresh_build` and
  `validation/final_fresh_install` directories passed: one package, 1.35 seconds.
- One `ros2 launch x2_recovery_ros recovery.launch.py render:=false` command
  started both nodes. Recovery reported the service ready, telemetry received
  `IDLE`, and both nodes shut down cleanly on SIGINT. The captured output is
  `ros2_ws/validation/final_fresh_launch.log`.
- `colcon test` and `colcon test-result --verbose`: 19 tests, zero errors,
  zero failures, zero skipped. The tests run against ROS 2 Jazzy/Python 3.12.3.

### Real simulator integration: passed

The complete `scripts/validate_ros.sh` run passed on 2026-09-20 local time
(2026-09-19 23:13 UTC). Evidence is in
`ros2_ws/validation/20260919T231348Z-15524/`.

| Check | Observed result | Evidence |
| --- | --- | --- |
| Fresh build | One package built successfully in 1.61 s | `build.log` |
| Unit tests | 19 passed, zero errors/failures/skips | `test_results.log`, `tests.log` |
| One launch command | Both required nodes started | `launch.log` |
| Acceptance before execution | `success=true` in 0.00085 s; first telemetry arrived afterwards | `integration.json` |
| Concurrent request | Immediate second request returned `success=false` | `integration.json` |
| Actual simulator telemetry | 516 messages, 31 named joints, finite changing positions | `integration.json` |
| Simulation timestamps | Strictly increasing from 0.02 to 10.32 seconds | `integration.json` |
| Configured wall timeout | `RUNNING` then `FAILED` at 12.019 wall seconds | `integration.json`, `launch.log` |
| Literal CLI start call | `Trigger_Response(success=True, ...)` | `cli_accepted.log` |
| Literal CLI busy call | `Trigger_Response(success=False, ...)` | `cli_busy.log` |
| Literal CLI live joint echo | Named positions and simulator timestamp received | `cli_joint_states.log` |
| Literal CLI final status echo | `data: FAILED` | `cli_final_status.log` |
| Clean shutdown | Both nodes exited cleanly; no episode process remained | `launch.log` |

The scripted baseline moved the real robot but did not achieve recovery during
these deliberately short timeout checks. `FAILED` is the expected result here,
not a claimed successful recovery. This validates ROS/simulator integration; the
five-episode learned-policy evaluation is a separate experiment.

Representative telemetry from the second real episode:

```text
status=RUNNING; left_hip_pitch_joint=-1.41117 rad; sim_time=2.880 s
status=RUNNING; left_hip_pitch_joint=-0.48975 rad; sim_time=6.880 s
Recovery completed: wall-clock timeout (includes simulator startup)
status=FAILED; left_hip_pitch_joint=-0.17509 rad; sim_time=10.600 s
```

The real-simulator probe is available for the final integration pass:

```bash
python3 ros2_ws/scripts/check_ros_integration.py --expect-final FAILED --deadline 30 --output ros2_ws/validation/integration.json
```

Run it from the repository root after sourcing ROS and the built workspace, with
the two nodes already launched. It creates one episode, checks acceptance and busy
rejection, records actual joint messages with advancing timestamps, and checks the
final status. Use a longer `--deadline` for a longer configured timeout.

### Exported-policy ROS smoke: passed

A separate check loaded the frozen exported PPO smoke actor
`artifacts/runs/smoke/model_000005_actor.pt` (five training iterations) through the
same ROS worker using `controller:=policy`. This is a checkpoint-loading and
execution smoke test, **not the final recovery evaluation**. It ran headlessly on
ROS domain 72 while GPU training and the visible viewer continued separately.

```bash
ROS_DOMAIN_ID=72 scripts/ros_launch.sh controller:=policy checkpoint:="$PWD/artifacts/runs/smoke/model_000005_actor.pt" render:=false realtime:=true timeout_s:=12.0 max_sim_duration_s:=60.0
```

In another sourced terminal, with the same ROS domain:

```bash
ROS_DOMAIN_ID=72 python3 ros2_ws/scripts/check_ros_integration.py --expect-final FAILED --deadline 30 --output ros2_ws/validation/policy_smoke/integration.json
```

- Acceptance returned in **0.000916 seconds**; the immediate second request was
  rejected as busy.
- **471 actual simulator joint messages** covered all 31 joints, with finite
  changing positions and strictly increasing timestamps from 0.02 to 9.42 seconds.
- The policy episode transitioned `RUNNING → FAILED` after **12.017 wall seconds**;
  the supervisor explicitly logged `wall-clock timeout (includes simulator startup)`.
- Both ROS nodes shut down cleanly after the check.

Evidence and checkpoint SHA-256 provenance are in
`ros2_ws/validation/policy_smoke/{integration.json,metadata.json,probe.log,launch.log}`.
The short smoke actor did not achieve recovery in this timeout test. No recovery
success is claimed from this ROS integration result.
