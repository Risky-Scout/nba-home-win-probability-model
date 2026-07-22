#!/usr/bin/env bash
# Open the next interview stage files in Cursor / VS Code.
# Usage (from repo root on GitHub main):  ./present.sh 0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# If script lives in print pack, allow calling with REPO=...
REPO="${REPO:-$PWD}"
if [[ ! -f "$REPO/README.md" || ! -d "$REPO/nba_wp" ]]; then
  REPO="/Users/josephshackelford/nba-home-win-probability-model"
fi
cd "$REPO"
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
if [[ "$branch" != "main" ]]; then
  echo "WARNING: on branch '$branch' — interview pack expects GitHub main." >&2
  echo "Run: git fetch origin && git checkout main && git reset --hard origin/main" >&2
fi

open_files() {
  if command -v cursor >/dev/null 2>&1; then
    cursor -r "$@"
  elif command -v code >/dev/null 2>&1; then
    code -r "$@"
  else
    open "$@"
  fi
}

stage="${1:-}"
case "$stage" in
  0)
    echo "STAGE 0 — Frame"
    open_files README.md SUMMARY.md artifacts/final_metrics.json
    ;;
  1)
    echo "STAGE 1 — Data audit"
    open_files artifacts/data_audit.json nba_wp/data.py
    ;;
  2)
    echo "STAGE 2 — Features / leakage"
    open_files nba_wp/features.py tests/test_feature_timing.py
    ;;
  3)
    echo "STAGE 3 — Model + stacker"
    open_files nba_wp/model.py artifacts/selected_spec.json artifacts/coefficient_table.csv
    ;;
  4)
    echo "STAGE 4 — Selection"
    open_files scripts/select_model.py nba_wp/selection.py \
      artifacts/selection_proof.json artifacts/march_architecture_results.csv
    ;;
  5)
    echo "STAGE 5 — Ablation"
    open_files artifacts/feature_group_ablation.csv figures/ablation_log_loss.png
    ;;
  6)
    echo "STAGE 6 — Importance / correlation"
    open_files artifacts/permutation_importance.csv \
      figures/permutation_importance.png figures/feature_correlation_matrix.png
    ;;
  7)
    echo "STAGE 7 — Calibration"
    open_files figures/march_calibration.png figures/april_calibration.png \
      artifacts/april_calibration_bins.csv
    ;;
  8)
    echo "STAGE 8 — April predictions"
    open_files outputs/april_predictions.csv outputs/april_predictions_frozen_snapshot.csv
    ;;
  9)
    echo "STAGE 9 — Limitations + verify"
    open_files docs/LIMITATIONS_AND_ROADMAP.md docs/REVIEWER_GUIDE.md
    python validate_submission.py --root . --data data/nba-win-probability-data.csv --recompute
    ;;
  all)
    echo "Opening full interview tab set"
    open_files README.md SUMMARY.md \
      artifacts/data_audit.json artifacts/selection_proof.json artifacts/selected_spec.json \
      artifacts/final_metrics.json artifacts/feature_group_ablation.csv \
      figures/ablation_log_loss.png figures/april_calibration.png \
      outputs/april_predictions.csv nba_wp/features.py nba_wp/model.py nba_wp/selection.py
    ;;
  *)
    echo "Usage: ./present.sh {0..9|all}"
    echo "Stages: 0 frame | 1 data | 2 features | 3 model | 4 selection |"
    echo "        5 ablation | 6 importance | 7 calibration | 8 april | 9 verify"
    exit 1
    ;;
esac
