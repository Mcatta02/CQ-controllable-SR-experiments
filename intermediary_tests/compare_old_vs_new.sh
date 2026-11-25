#!/usr/bin/env bash
# Runs the SAME timing script against the pre-optimization baseline
# (pre_optimized_baseline/) and the current optimized code (models/),
# swapping files in and out, and prints a before/after comparison.
#
# Usage: ./compare_old_vs_new.sh
# Run from the repo root.

set -euo pipefail

MODELS_DIR="models"
BASELINE_DIR="pre_optimized_baseline"
BACKUP_DIR="$(mktemp -d)"
FILES=(attentions.py convs.py rnn.py transformers.py)

echo "Backing up current (optimized) files to $BACKUP_DIR ..."
for f in "${FILES[@]}"; do
    cp "$MODELS_DIR/$f" "$BACKUP_DIR/$f"
done

restore() {
    echo "Restoring optimized files..."
    for f in "${FILES[@]}"; do
        cp "$BACKUP_DIR/$f" "$MODELS_DIR/$f"
    done
}
trap restore EXIT

echo
echo "=== Swapping in pre-optimization baseline ==="
for f in "${FILES[@]}"; do
    cp "$BASELINE_DIR/$f" "$MODELS_DIR/$f"
done
CUDA_VISIBLE_DEVICES="${GPU:-3}" python compare_timing.py --label OLD

echo
echo "=== Restoring optimized code ==="
restore
trap - EXIT   # already restored, skip the exit-trap restore too

CUDA_VISIBLE_DEVICES="${GPU:-3}" python compare_timing.py --label NEW

echo
echo "Done. See old_results.json and new_results.json for the raw numbers."
