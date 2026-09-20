#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec .venv/bin/python scripts/demo_full_recovery.py --render --keep-open "$@"
