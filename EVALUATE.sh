#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing virtual environment: $PWD/.venv. Run bash scripts/setup.sh first." >&2
  exit 1
fi
exec .venv/bin/python scripts/evaluate_full_recovery.py \
  --actor artifacts/submission/final_recovery/actor.pt --episodes 5 --seed 30001 \
  --output artifacts/local_evaluation/final_policy_5.json "$@"
