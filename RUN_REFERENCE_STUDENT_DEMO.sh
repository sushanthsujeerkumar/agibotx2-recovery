#!/usr/bin/env bash
# Replay the failed experimental student with explicit on-screen provenance.
set -euo pipefail
cd "$(dirname "$0")"
actor=artifacts/experiments/reference_student_local/bc_phase_features/actor.pt
if [ ! -f "$actor" ]; then
    echo "Local experimental student is absent. See REFERENCE_STUDENT_RESULTS.md." >&2
    exit 2
fi
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=glfw
out="artifacts/local_demo/reference_student_$(date +%Y%m%d_%H%M%S)_$$"
exec .venv/bin/python scripts/record_reference_student.py --actor "$actor" --seed 9904 \
    --expected artifacts/validation/reference_student/final_fresh_eval/9904.json --output "$out" --live
