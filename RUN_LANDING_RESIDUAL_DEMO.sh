#!/usr/bin/env bash
# Physical replay, labelled external teacher + trained PPO ankle correction.
set -euo pipefail
cd "$(dirname "$0")"
assets="${X2_VENDOR_ASSETS_DIR:-$PWD/.cache/vendor_reference}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=glfw
exec .venv/bin/python scripts/record_landing_residual.py --asset-dir "$assets" \
    --checkpoint artifacts/experiments/landing_residual_ppo/update_004.pt --seed 11301 \
    --expected artifacts/validation/landing_residual/paired_fresh/residual/11301.json \
    --output "artifacts/local_demo/landing_residual_$(date +%Y%m%d_%H%M%S)_$$" --live
