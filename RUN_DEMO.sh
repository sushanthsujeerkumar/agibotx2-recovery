#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec .venv/bin/python -m x2_recovery.evaluate --controller policy --checkpoint "$PWD/artifacts/submission/local_stability/actor.pt" --episodes 1 --seed 1001 --render --output artifacts/local_demo "$@"
