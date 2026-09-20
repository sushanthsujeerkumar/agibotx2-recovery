#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec .venv/bin/python -u -m x2_recovery.train --variant stance --initialize-from artifacts/submission/stable_crossed_stance_009251/checkpoint.pt --num-envs 512 --seed 1 --max-iterations 100000 --max-seconds 2700 --save-interval 250 --print-interval 100 --output artifacts/runs/local_stance "$@"
