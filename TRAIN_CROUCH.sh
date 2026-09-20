#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec .venv/bin/python -m x2_recovery.train --variant crouch --num-envs 256 --seed 4 --torch-threads 2 --max-iterations 1000 --max-seconds 300 --save-interval 50 --print-interval 25 --freeze-actor-normalization --initialize-from artifacts/experiments/crouch_teacher_augmented/checkpoint.pt --output artifacts/runs/crouch_v3 "$@"
