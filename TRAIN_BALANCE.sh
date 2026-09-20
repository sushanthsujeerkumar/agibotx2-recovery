#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec .venv/bin/python -m x2_recovery.train --variant stance --num-envs 256 --seed 2 --torch-threads 2 --max-iterations 1000 --max-seconds 180 --save-interval 50 --print-interval 25 --env-kwargs '{"physics_profile":"guarded_v2","reset_mode":"balance"}' --output artifacts/runs/balance_v2 "$@"
