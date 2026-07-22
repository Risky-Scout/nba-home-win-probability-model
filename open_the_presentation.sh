#!/usr/bin/env bash
# Tabs from 12_THE_PRESENTATION.pdf — pre-open list, in order
set -euo pipefail
REPO="/Users/josephshackelford/nba-interview"
cd "$REPO"
code "$REPO" \
  SUMMARY.md \
  outputs/april_predictions.csv \
  artifacts/data_audit.json \
  nba_wp/data.py \
  nba_wp/features.py \
  artifacts/feature_group_ablation.csv \
  configs/architecture_candidates.json \
  nba_wp/model.py \
  artifacts/selected_spec.json \
  artifacts/march_architecture_results.csv \
  artifacts/selection_proof.json \
  artifacts/final_metrics.json \
  figures/april_calibration.png \
  docs/LIMITATIONS_AND_ROADMAP.md \
  artifacts/feature_correlations.csv \
  artifacts/feature_dictionary.csv \
  artifacts/feature_examples.csv \
  artifacts/paired_bootstrap_vs_elo.json \
  tests/test_feature_timing.py
echo "Opened 12_THE_PRESENTATION tab set."
echo "Terminal next:"
echo "  cd $REPO && source .venv/bin/activate"
echo "  python validate_submission.py --root . --data data/nba-win-probability-data.csv --recompute"
