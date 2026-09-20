#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/RUN_RECOVERY.sh" render:=true "$@"
