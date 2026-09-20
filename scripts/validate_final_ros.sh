#!/usr/bin/env bash
# Fresh ROS build + final full-recovery policy success, busy and timeout checks.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
evidence="${X2_FINAL_ROS_EVIDENCE:-$repo_dir/artifacts/local_evaluation/final_ros_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ -e "$evidence" ]]; then
    echo "Choose a fresh evidence directory: $evidence" >&2
    exit 2
fi
mkdir -p "$evidence"
set +u
source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID="${X2_ROS_DOMAIN_ID:-93}"
export ROS_LOCALHOST_ONLY=1
export ROS_LOG_DIR="$evidence/ros_logs"
export PYTHONUNBUFFERED=1
venv_site="$("$repo_dir/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
cd "$repo_dir/ros2_ws"
colcon --log-base "$evidence/colcon_log" build --build-base "$evidence/build" \
    --install-base "$evidence/install" --packages-select x2_recovery_ros > "$evidence/build.log" 2>&1
colcon --log-base "$evidence/test_log" test --build-base "$evidence/build" \
    --install-base "$evidence/install" --packages-select x2_recovery_ros \
    --event-handlers console_direct+ > "$evidence/test.log" 2>&1
colcon test-result --test-result-base "$evidence/build" --verbose > "$evidence/test_result.log" 2>&1
set +u
source "$evidence/install/setup.bash"
set -u
# Keep colcon's Ubuntu setuptools/pytest discovery separate from the simulator
# venv; expose MuJoCo/Torch only after the fresh package build and test stages.
export PYTHONPATH="$repo_dir/src:$venv_site:${PYTHONPATH:-}"
launch_pid=''
cleanup() {
    if [[ -n "$launch_pid" ]] && kill -0 "$launch_pid" 2>/dev/null; then
        kill -INT -- "-$launch_pid" 2>/dev/null || true
        for _ in {1..50}; do
            kill -0 "$launch_pid" 2>/dev/null || break
            sleep .1
        done
        if kill -0 "$launch_pid" 2>/dev/null; then
            kill -TERM -- "-$launch_pid" 2>/dev/null || true
        fi
        wait "$launch_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
setsid ros2 launch x2_recovery_ros recovery.launch.py controller:=full_recovery \
    checkpoint:="$repo_dir/artifacts/submission/final_recovery/actor.pt" \
    physics_profile:=guarded_v2 seed:=30001 render:=false \
    realtime:=true timeout_s:=60.0 max_sim_duration_s:=15.0 > "$evidence/launch.log" 2>&1 &
launch_pid=$!
/usr/bin/python3 "$repo_dir/ros2_ws/scripts/check_ros_integration.py" --expect-final SUCCEEDED \
    --deadline 60 --output "$evidence/success.json" > "$evidence/success_probe.log" 2>&1
timeout 10s ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}' > "$evidence/cli_accepted.log" 2>&1
rg -q 'success=True' "$evidence/cli_accepted.log"
timeout 10s ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}' > "$evidence/cli_busy.log" 2>&1
rg -q 'success=False' "$evidence/cli_busy.log"
timeout 10s ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState --once > "$evidence/cli_joints.log"
sleep 20
timeout 5s ros2 topic echo /x2/recovery_status std_msgs/msg/String --once > "$evidence/cli_success.log"
rg -q 'SUCCEEDED' "$evidence/cli_success.log"
ros2 param set /x2_recovery timeout_s 5.0 > "$evidence/set_wall_timeout.log"
/usr/bin/python3 "$repo_dir/ros2_ws/scripts/check_ros_integration.py" --expect-final FAILED \
    --deadline 30 --output "$evidence/wall_timeout.json" > "$evidence/wall_timeout_probe.log" 2>&1
rg -q 'wall-clock timeout' "$evidence/launch.log"
ros2 param set /x2_recovery timeout_s 60.0 > "$evidence/reset_wall_timeout.log"
ros2 param set /x2_recovery max_sim_duration_s 0.1 > "$evidence/set_sim_timeout.log"
/usr/bin/python3 "$repo_dir/ros2_ws/scripts/check_ros_integration.py" --expect-final FAILED \
    --deadline 30 --output "$evidence/sim_timeout.json" > "$evidence/sim_timeout_probe.log" 2>&1
rg -q 'simulation duration timeout' "$evidence/launch.log"
echo "PASS: fresh build, real recovery, CLI telemetry, busy rejection, wall and simulation timeouts."
