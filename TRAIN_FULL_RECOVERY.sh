#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing virtual environment: $PWD/.venv. Run bash scripts/setup.sh first." >&2
  exit 1
fi
exec .venv/bin/python scripts/train_full_recovery.py \
  --num-envs 256 --visible 16 --seconds 1800 \
  --resume artifacts/submission/final_recovery/initial_checkpoint.pt --evaluate-final \
  --output "artifacts/runs/full_recovery_$(date +%Y%m%d_%H%M%S)" "$@"
