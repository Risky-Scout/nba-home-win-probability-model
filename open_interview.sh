#!/usr/bin/env bash
set -euo pipefail
REPO="/Users/josephshackelford/nba-interview"
cd "$REPO"
code "$REPO" \
  README.md SUMMARY.md \
  artifacts/data_audit.json nba_wp/data.py \
  nba_wp/features.py nba_wp/model.py nba_wp/selection.py scripts/select_model.py \
  artifacts/selection_proof.json artifacts/selected_spec.json artifacts/coefficient_table.csv \
  artifacts/march_architecture_results.csv artifacts/final_metrics.json \
  artifacts/feature_group_ablation.csv figures/ablation_log_loss.png \
  artifacts/permutation_importance.csv figures/permutation_importance.png \
  figures/feature_correlation_matrix.png \
  figures/march_calibration.png figures/april_calibration.png \
  outputs/april_predictions.csv outputs/april_predictions_frozen_snapshot.csv \
  docs/LIMITATIONS_AND_ROADMAP.md docs/REVIEWER_GUIDE.md \
  validate_submission.py present.sh
open "/Users/josephshackelford/Desktop/Technical Portfolio/Bet365 Interview prep/GITHUB_MAIN_PRINT_PACK/01_PRESENTATION_SCRIPT_STAGES.md"
echo "Opened. In VS Code terminal run:"
echo "  source .venv/bin/activate && ./present.sh 0"
