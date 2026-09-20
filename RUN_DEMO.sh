#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing virtual environment: $PWD/.venv. Run bash scripts/setup.sh first." >&2
  exit 1
fi
exec .venv/bin/python scripts/demo_full_recovery.py --render --keep-open "$@"
