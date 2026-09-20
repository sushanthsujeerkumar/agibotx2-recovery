#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec .venv/bin/python -m x2_recovery.watch --directory artifacts/experiments/deep_crouch_ppo --minutes 1 --physics-profile guarded_v2 --reset-mode deep_crouch --assess-stance "$@"
