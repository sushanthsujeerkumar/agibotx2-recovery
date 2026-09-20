#!/usr/bin/env bash
# Build and launch the submitted policy and telemetry nodes.
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")" && pwd)"
exec "$repo_dir/scripts/ros_launch.sh" controller:=full_recovery \
  checkpoint:="$repo_dir/artifacts/submission/final_recovery/actor.pt" \
  physics_profile:=guarded_v2 seed:=30001 max_sim_duration_s:=15.0 timeout_s:=60.0 "$@"
