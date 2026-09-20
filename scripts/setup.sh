#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
extra_args=()
if [[ "${1:-}" == '--reference' ]]; then
  extra_args=(--extra reference)
elif [[ "$#" -gt 0 ]]; then
  echo "Usage: $0 [--reference]" >&2
  exit 2
fi
if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -x "$PROJECT_DIR/.tools/uv" ]]; then
  UV_BIN="$PROJECT_DIR/.tools/uv"
else
  mkdir -p "$PROJECT_DIR/.tools"
  curl -LsSf https://astral.sh/uv/0.12.17/install.sh -o "$PROJECT_DIR/.tools/install-uv.sh"
  UV_INSTALL_DIR="$PROJECT_DIR/.tools" UV_NO_MODIFY_PATH=1 sh "$PROJECT_DIR/.tools/install-uv.sh"
  UV_BIN="$PROJECT_DIR/.tools/uv"
fi
UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=3 "$UV_BIN" sync --frozen --python /usr/bin/python3 "${extra_args[@]}"
.venv/bin/python -m pytest tests -q
printf '\nSetup complete. ROS 2 Jazzy must be installed separately for ROS integration.\n'
