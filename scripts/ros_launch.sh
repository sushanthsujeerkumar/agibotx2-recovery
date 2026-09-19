#!/usr/bin/env bash
# Build the two-node ROS package and launch it against the shared simulator.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${X2_VENV_DIR:-$repo_dir/.venv}"
if [[ ! -x "$venv_dir/bin/python" ]]; then
  echo "Missing virtual environment: $venv_dir. Run the setup instructions first." >&2
  exit 1
fi
set +u
source /opt/ros/jazzy/setup.bash
set -u
cd "$repo_dir/ros2_ws"
colcon build --symlink-install --packages-select x2_recovery_ros
set +u
source install/setup.bash
set -u
venv_site="$("$venv_dir/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export PYTHONPATH="$repo_dir/src:$venv_site:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
/usr/bin/python3 -c 'import rclpy, mujoco, x2_recovery.runtime'
exec ros2 launch x2_recovery_ros recovery.launch.py "$@"
