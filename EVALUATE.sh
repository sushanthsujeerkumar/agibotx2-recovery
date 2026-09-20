#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec .venv/bin/python scripts/evaluate_full_recovery.py \
  --actor artifacts/submission/final_recovery/actor.pt --episodes 5 --seed 30001 \
  --output artifacts/local_evaluation/final_policy_5.json "$@"
