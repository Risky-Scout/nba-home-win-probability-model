# Claim-to-artifact map

This document is the reviewer's index. Every strong statement in the README
has a corresponding code path, generated artifact, or automated test.

| Claim | Source code | Generated evidence | Automated check |
|---|---|---|---|
| The file has 1,230 games | `nba_wp/data.py::audit_games` | `artifacts/data_audit.json` | `tests/test_real_data_contract.py::test_real_data_audit` |
| The actual schema has 16 columns | `nba_wp/data.py::EXPECTED_COLUMNS` | `artifacts/data_audit.json` | load-time schema exception |
| Pregame wins/losses reconcile | `nba_wp/data.py::_record_reconciliation` | `artifacts/data_audit.json` | `test_real_data_audit` |
| Current-game box scores do not enter current features | `nba_wp/features.py::build_features` | `outputs/engineered_features.csv` | `test_current_game_postgame_values_do_not_change_current_features` |
| Same-date games are batched | `nba_wp/features.py::build_features` | feature `performance_cutoff` | `test_same_day_games_are_batched` |
| Frozen snapshot stops target-month performance updates | `build_features(..., freeze_date=...)` | frozen prediction CSVs | `test_frozen_snapshot_stops_performance_updates` |
| April cannot enter model selection | `scripts/select_model.py`; `nba_wp/selection.py` | `artifacts/selection_proof.json` | validator selection assertions |
| The stacker coefficients were fitted, not typed | `nba_wp/model.py::fit_logit_stacker` | `march_architecture_results.csv`; `selected_spec.json` | full reproduction |
| Elo MOV uses winner − loser rating difference | `nba_wp/features.py::build_features` | `outputs/engineered_features.csv` | `tests/test_elo_mov_winner_diff.py` |
| Deploy stacker never sharpens (temperature T >= 1) | `nba_wp/model.py::fit_logit_stacker` | `selected_spec.json` (floored + unconstrained coeffs) | `tests/test_stacker_temperature_floor.py` |
| Primary April is frozen (April outcomes cannot change April prices) | `nba_wp/reporting.py::score_and_write` | `outputs/april_predictions.csv` | `tests/test_primary_april_frozen.py` |
| March probabilities use coefficients fitted through February | `scripts/score_final.py`; `nba_wp/reporting.py::score_and_write` | `outputs/march_predictions.csv` | validator |
| April coefficients are refitted through March | `score_and_write` | `artifacts/trained_model.joblib`; April output | full recomputation |
| Metrics equal the game-level probabilities | `nba_wp/model.py::evaluate` | `artifacts/final_metrics.json` | `tests/test_artifacts.py`; `validate_submission.py` |
| The final model improves the constant baseline | `ablation_table` | `feature_group_ablation.csv`; bootstrap JSON | artifact test plus direct recomputation |
| Bradley-Terry is the largest selected contribution | `permutation_importance` | `permutation_importance.csv` | deterministic seed and rerun |
| Saved output is reproducible | `score_and_write` | prediction CSVs and manifest | `validate_submission.py --recompute` |

## Manual spot check

For any row in `outputs/april_predictions.csv`:

1. identify the game and prediction;
2. inspect the same `game_id` in `outputs/engineered_features.csv`;
3. confirm the listed `performance_cutoff`;
4. inspect component probabilities;
5. recompute the final probability from `artifacts/selected_spec.json`;
6. recompute its log-loss and Brier contribution.

The prediction files contain every quantity required for that calculation.
