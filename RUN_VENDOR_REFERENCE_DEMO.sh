#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ASSET_DIRECTORY="${1:-${X2_VENDOR_ASSETS_DIR:-../../work/official_getup_compatibility}}"
if [[ ! -f "$ASSET_DIRECTORY/policy_b_lie_up.onnx" ]]; then
  echo 'External AgiBot assets are required. See VENDOR_REFERENCE_RESULTS.md for the download command.' >&2
  exit 1
fi
mkdir -p artifacts/local_demo
REPLAY_DIRECTORY="$(mktemp -d artifacts/local_demo/vendor-reference-XXXXXX)"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec .venv/bin/python scripts/record_vendor_reference.py \
  --asset-dir "$ASSET_DIRECTORY" --seed 9401 --timescale 1.2 --live \
  --expected artifacts/validation/vendor_reference/slower_fresh_validation/9401.json \
  --output "$REPLAY_DIRECTORY"
