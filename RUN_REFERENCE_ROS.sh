#!/usr/bin/env bash
# Explicitly external teacher + our trained PPO ankle correction.
set -euo pipefail
cd "$(dirname "$0")"
assets="${X2_VENDOR_ASSETS_DIR:-$PWD/.cache/vendor_reference}"
exec bash scripts/ros_launch.sh controller:=reference_residual \
    checkpoint:="$PWD/artifacts/experiments/landing_residual_ppo/update_004.pt" \
    vendor_assets:="$assets" physics_profile:=guarded_v2 render:=true realtime:=true \
    seed:=11301 timeout_s:=60.0 max_sim_duration_s:=15.0 "$@"
