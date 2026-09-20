#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec bash scripts/ros_launch.sh controller:=policy checkpoint:="$PWD/artifacts/submission/local_stability/actor.pt" render:=true seed:=1001 timeout_s:=60.0 max_sim_duration_s:=15.0 "$@"
