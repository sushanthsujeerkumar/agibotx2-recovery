#!/usr/bin/env bash
# Reproduce the bounded supervised experiment; no cloud and no PPO updates.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 EXTERNAL_VENDOR_ASSET_DIRECTORY FRESH_OUTPUT_DIRECTORY" >&2
    exit 2
fi
assets=$(realpath "$1")
out=$(realpath -m "$2")
if [ -e "$out" ]; then
    echo "Choose a fresh output directory: $out" >&2
    exit 2
fi
mkdir -p "$out"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
py=.venv/bin/python
script=scripts/train_reference_student.py
"$py" "$script" collect --asset-dir "$assets" --seeds {9501..9516} --output "$out/demonstrations"
"$py" "$script" fit --datasets "$out/demonstrations/dataset.npz" --steps 5000 --output "$out/bc_round0"
"$py" "$script" evaluate --actor "$out/bc_round0/actor.pt" --seeds 9601 9602 9603 --output "$out/bc_round0_eval"
"$py" "$script" collect --asset-dir "$assets" --actor "$out/bc_round0/actor.pt" --teacher-fraction .5 --seeds {9701..9706} --output "$out/dagger1"
"$py" "$script" fit --datasets "$out/demonstrations/dataset.npz" "$out/dagger1/dataset.npz" --parent "$out/bc_round0/checkpoint.pt" --validation-seeds 9514 9515 9516 9706 --learning-rate .0001 --steps 10000 --output "$out/bc_round1"
"$py" "$script" evaluate --actor "$out/bc_round1/actor.pt" --seeds 9601 9602 9603 --output "$out/bc_round1_eval"
"$py" "$script" fit --datasets "$out/demonstrations/dataset.npz" "$out/dagger1/dataset.npz" --validation-seeds 9514 9515 9516 9706 --drop-previous-action --steps 15000 --output "$out/bc_no_history"
"$py" "$script" evaluate --actor "$out/bc_no_history/actor.pt" --seeds 9601 9602 9603 --output "$out/bc_no_history_eval"
"$py" "$script" fit --datasets "$out/demonstrations/dataset.npz" "$out/dagger1/dataset.npz" --validation-seeds 9514 9515 9516 9706 --drop-previous-action --phase-harmonics 16 --unclamped-loss --learning-rate .0001 --steps 20000 --output "$out/bc_phase_features"
"$py" "$script" evaluate --actor "$out/bc_phase_features/actor.pt" --seeds {9901..9905} --output "$out/final_fresh_eval"
