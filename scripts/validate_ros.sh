#!/usr/bin/env bash
# Fresh colcon build + tests + real simulator service, telemetry and timeout checks.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${X2_VENV_DIR:-$repo_dir/.venv}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
evidence_dir="${X2_ROS_EVIDENCE_DIR:-$repo_dir/ros2_ws/validation/$run_id}"
timeout_s="${X2_ROS_TIMEOUT_SECONDS:-12.0}"
mkdir -p "$evidence_dir"
set +u
source /opt/ros/jazzy/setup.bash
set -u
# Isolate this headless validation from any visible development simulator or ROS nodes.
export ROS_DOMAIN_ID="${X2_ROS_DOMAIN_ID:-71}"
export ROS_LOCALHOST_ONLY=1
export ROS_LOG_DIR="$evidence_dir/ros_logs"
export PYTHONUNBUFFERED=1
cd "$repo_dir/ros2_ws"
colcon --log-base "$evidence_dir/log" build --build-base "$evidence_dir/build" \
  --install-base "$evidence_dir/install" --packages-select x2_recovery_ros \
  2>&1 | tee "$evidence_dir/build.log"
colcon --log-base "$evidence_dir/test_log" test --build-base "$evidence_dir/build" \
  --install-base "$evidence_dir/install" --packages-select x2_recovery_ros \
  --event-handlers console_direct+ 2>&1 | tee "$evidence_dir/tests.log"
colcon test-result --test-result-base "$evidence_dir/build" --verbose \
  2>&1 | tee "$evidence_dir/test_results.log"
set +u
source "$evidence_dir/install/setup.bash"
set -u
venv_site="$("$venv_dir/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export PYTHONPATH="$repo_dir/src:$venv_site:${PYTHONPATH:-}"
/usr/bin/python3 -c 'import rclpy, mujoco, x2_recovery.runtime'
launch_pid=''
cleanup() {
  if [[ -n "$launch_pid" ]] && kill -0 "$launch_pid" 2>/dev/null; then
    kill -INT -- "-$launch_pid" 2>/dev/null || true
    for _ in {1..50}; do
      kill -0 "$launch_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$launch_pid" 2>/dev/null; then
      kill -TERM -- "-$launch_pid" 2>/dev/null || true
    fi
    wait "$launch_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
setsid ros2 launch x2_recovery_ros recovery.launch.py controller:=scripted \
  render:=false realtime:=true "timeout_s:=$timeout_s" max_sim_duration_s:=60.0 \
  > "$evidence_dir/launch.log" 2>&1 &
launch_pid=$!
/usr/bin/python3 "$repo_dir/ros2_ws/scripts/check_ros_integration.py" \
  --expect-final FAILED --deadline 60 --output "$evidence_dir/integration.json" \
  2>&1 | tee "$evidence_dir/probe.log"
timeout 10s ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}' \
  2>&1 | tee "$evidence_dir/cli_accepted.log"
rg -q 'success=True' "$evidence_dir/cli_accepted.log"
timeout 10s ros2 service call /x2/start_recovery std_srvs/srv/Trigger '{}' \
  2>&1 | tee "$evidence_dir/cli_busy.log"
rg -q 'success=False' "$evidence_dir/cli_busy.log"
timeout 10s ros2 topic echo /x2/joint_states sensor_msgs/msg/JointState --once \
  > "$evidence_dir/cli_joint_states.log"
sleep "$timeout_s"
timeout 5s ros2 topic echo /x2/recovery_status std_msgs/msg/String --once \
  > "$evidence_dir/cli_final_status.log"
rg -q 'FAILED' "$evidence_dir/cli_final_status.log"
rg -q 'wall-clock timeout' "$evidence_dir/launch.log"
printf 'PASS: fresh ROS build, unit tests, real simulator telemetry, acceptance, busy rejection and wall timeout. Evidence: %s\n' "$evidence_dir"
