# Claim-to-artifact map

This document is the reviewer's index. Every strong statement in the README
has a corresponding code path, generated artifact, or automated test.

| Claim | Source code | Generated evidence | Automated check |
|---|---|---|---|
| The file has 1,230 games | `nba_wp/data.py::audit_games` | `artifacts/current/data_audit.json` | `tests/test_real_data_contract.py::test_real_data_audit` |
| The actual schema has 16 columns | `nba_wp/data.py::EXPECTED_COLUMNS` | `artifacts/current/data_audit.json` | load-time schema exception |
| Pregame wins/losses reconcile | `nba_wp/data.py::_record_reconciliation` | `artifacts/current/data_audit.json` | `test_real_data_audit` |
| Current-game box scores do not enter current features | `nba_wp/features.py::build_features` | `outputs/engineered_features.csv` | `test_current_game_postgame_values_do_not_change_current_features` |
| Same-date games are batched | `nba_wp/features.py::build_features` | feature `performance_cutoff` | `test_same_day_games_are_batched` |
| Frozen snapshot stops target-month performance updates | `build_features(..., freeze_date=...)` | frozen prediction CSVs | `test_frozen_snapshot_stops_performance_updates` |
| April cannot enter model selection | `scripts/select_model.py`; `nba_wp/selection.py` | `artifacts/current/pre_march_selection_proof.json` | validator selection assertions |
| The selected configuration came from the grid | `nba_wp/model.py::search_calibration` | `march_architecture_results.csv`; `march_tuning_top_candidates.csv`; `selected_spec.json` | full reproduction |
| March probabilities use coefficients fitted through February | `scripts/score_final.py`; `nba_wp/reporting.py::score_and_write` | `outputs/march_predictions.csv` | validator |
| April coefficients are refitted through March | `score_and_write` | `artifacts/trained_model.joblib`; April output | full recomputation |
| Metrics equal the game-level probabilities | `nba_wp/model.py::evaluate` | `artifacts/current/final_metrics.json` | `tests/test_artifacts.py`; `validate_submission.py` |
| The final model improves the constant baseline | `ablation_table` | `feature_group_ablation.csv`; bootstrap JSON | artifact test plus direct recomputation |
| Bradley-Terry is the largest selected contribution | `permutation_importance` | `permutation_importance.csv` | deterministic seed and rerun |
| Saved output is reproducible | `score_and_write` | prediction CSVs and manifest | `validate_submission.py --recompute` |

## Manual spot check

For any row in `outputs/april_predictions_frozen_snapshot.csv`:

1. identify the game and prediction;
2. inspect the same `game_id` in `outputs/engineered_features.csv`;
3. confirm the listed `performance_cutoff`;
4. inspect component probabilities;
5. recompute the final probability from `artifacts/current/selected_spec_pre_march.json`;
6. recompute its log-loss and Brier contribution.

The prediction files contain every quantity required for that calculation.
