#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec .venv/bin/python scripts/train_full_recovery.py \
  --num-envs 256 --visible 16 --seconds 1800 \
  --resume artifacts/submission/final_recovery/initial_checkpoint.pt --evaluate-final \
  --output "artifacts/runs/full_recovery_$(date +%Y%m%d_%H%M%S)" "$@"
