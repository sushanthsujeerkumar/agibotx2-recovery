#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec .venv/bin/python -u -m x2_recovery.train --variant stability --num-envs 512 --seed 0 --max-iterations 100000 --max-seconds 5400 --save-interval 250 --print-interval 100 --output artifacts/runs/local_stability "$@"
