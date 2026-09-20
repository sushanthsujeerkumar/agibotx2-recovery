# ROS 2 integration and validation

The package `x2_recovery_ros` contains two nodes launched together by `recovery.launch.py`:

```text
Trigger request -> recovery node -> simulator subprocess
                         |              |
                         |        real joint samples
                         v              v
                /x2/recovery_status  /x2/joint_states
                         \              /
                          telemetry node
```

The service reserves the attempt and queues work. A subsequent timer callback starts the simulator, after the acceptance response has been sent by the single-threaded executor. Another request while queued or running is rejected. The worker loads the exported actor and publishes actual simulation joint states. Status uses reliable, transient-local QoS so late subscribers can read the latest state. Joint messages have simulation timestamps, restarting from zero for each episode.

## Manual build and launch

After Python setup, from the repository root:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --packages-select x2_recovery_ros
source install/setup.bash
cd ..
export PYTHONPATH="$PWD/src:$($PWD/.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))'):${PYTHONPATH:-}"
ros2 launch x2_recovery_ros recovery.launch.py controller:=full_recovery \
  checkpoint:="$PWD/artifacts/submission/final_recovery/actor.pt" \
  physics_profile:=guarded_v2 seed:=30001 render:=false \
  realtime:=true max_sim_duration_s:=15.0 timeout_s:=60.0
```

`./RUN_RECOVERY.sh` performs those steps. The generic launch also supports a scripted baseline and plain 106-input actors; use the explicit `full_recovery` mode and `guarded_v2` profile for this submission.

## Recorded checks

The complete validation script is `scripts/validate_final_ros.sh`. It creates new build/install directories, builds the package, runs its tests, launches both nodes, executes the real service/topic CLI calls, and checks successful recovery, busy rejection and both deadlines. It stops only the launch process group it created.

```bash
scripts/validate_final_ros.sh
```

The submission run passed with **28 ROS package tests, zero failures or errors**. Its evidence is in [`artifacts/submission/final_recovery/ros/`](../artifacts/submission/final_recovery/ros/).

| Check | Recorded outcome |
|---|---|
| Fresh colcon build/test | Passed; `build.log`, `test_result.log` |
| Service acceptance | `success=True`, response about 0.00258 s; `success.json` |
| Immediate second request | `success=False`, busy; `success.json` and `cli_busy.log` |
| Full submitted-policy recovery | `RUNNING` then `SUCCEEDED`; `success.json` |
| Joint telemetry | 31 names, changing positions, increasing simulation timestamps; `success.json`, `cli_joints.log` |
| Telemetry node | Status and first joint logged; `launch.log` |
| Literal CLI recovery | Accepted, busy rejected, then `SUCCEEDED`; `cli_*.log` |
| Wall-clock timeout | `timeout_s=5.0`, result `FAILED`; `wall_timeout.json` |
| Simulation timeout | `max_sim_duration_s=0.1`, result `FAILED`; `sim_timeout.json` |

Example timeout commands, with the nodes running and idle:

```bash
ros2 param set /x2_recovery timeout_s 5.0
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
# After FAILED, restore the wall deadline and test simulation duration.
ros2 param set /x2_recovery timeout_s 60.0
ros2 param set /x2_recovery max_sim_duration_s 0.1
ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}'
```

The wall watchdog includes process/model startup; the simulation deadline is episode time. Parameters are captured when an attempt is accepted. The final policy still requires a full 15-second evaluation when a larger simulated duration is configured; a smaller value causes timeout, not early success.
