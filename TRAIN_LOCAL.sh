#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec .venv/bin/python -u -m x2_recovery.train --num-envs 512 --max-iterations 100000 --max-seconds 3600 --output artifacts/runs/local "$@"
