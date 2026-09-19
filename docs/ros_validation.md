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
constitute simulator-integration evidence. The recorded real-simulator build,
launch and CLI outcomes will be appended below after the shared runtime is ready.

## Recorded checks, 2026-09-20

- A fresh build into new `validation/final_fresh_build` and
  `validation/final_fresh_install` directories passed: one package, 1.35 seconds.
- One `ros2 launch x2_recovery_ros recovery.launch.py render:=false` command
  started both nodes. Recovery reported the service ready, telemetry received
  `IDLE`, and both nodes shut down cleanly on SIGINT. The captured output is
  `ros2_ws/validation/final_fresh_launch.log`.
- `colcon test` and `colcon test-result --verbose`: 19 tests, zero errors,
  zero failures, zero skipped. The tests run against ROS 2 Jazzy/Python 3.12.3.
- Episode execution, real simulator telemetry, busy rejection via CLI, and
  timeout integration are pending completion of the shared simulator runtime.

The real-simulator probe is available for the final integration pass:

```bash
python3 ros2_ws/scripts/check_ros_integration.py --expect-final FAILED --deadline 30 --output ros2_ws/validation/integration.json
```

Run it from the repository root after sourcing ROS and the built workspace, with
the two nodes already launched. It creates one episode, checks acceptance and busy
rejection, records actual joint messages with advancing timestamps, and checks the
final status. Use a longer `--deadline` for a longer configured timeout.
