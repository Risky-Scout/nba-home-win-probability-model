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
| April cannot enter model selection | `scripts/select_model.py`; `nba_wp/selection.py` | `artifacts/selection_proof.json` | `tests/test_information_policies.py`; validator |
| The deployed model is Elo-only (champion decision) | `nba_wp/selection.py` | `selected_spec.json` (`model_family`/`champion` = `elo_only`) | `tests/test_champion_promotion.py` |
| Primary April prices recompute from the deployed Elo coefficients | `nba_wp/reporting.py::score_and_write` | `selected_spec.json` `elo_model`; `outputs/april_predictions.csv` | `tests/test_champion_promotion.py` |
| Elo architecture is selected by aggregate frozen-policy OOS log loss with a one-SE stability rule (lowest K in band) | `nba_wp/selection.py` | `march_architecture_results.csv`; `selected_spec.json` (`architecture_selection`) | `tests/test_architecture_selection.py`; `tests/test_champion_promotion.py` |
| MOV offset (2.2) and cold-start warmup (off) are data-driven | `nba_wp/selection.py` | `selected_spec.json` (`mov_offset_selection`, `cold_start_selection`) | `tests/test_champion_promotion.py` |
| Elo MOV uses winner − loser rating difference | `nba_wp/features.py::build_features` | `outputs/engineered_features.csv` | `tests/test_elo_mov_winner_diff.py` |
| Rejected challenger blend is a genuine convex stack (0<=w<=1, sum 1, T>=1) | `nba_wp/model.py::fit_logit_stacker` | `selected_spec.json` `challenger` block | `tests/test_stacker_temperature_floor.py::test_stacker_weights_are_convex_when_stacker_is_used` |
| Nested audit rejects the blend (worse OOS log loss and Brier) | `scripts/nested_validation.py` | `artifacts/nested_frozen_block_summary.json`; `artifacts/nested_daily_sequential_summary.json` | `tests/test_champion_promotion.py` |
| Nested audit also rejects calibrated-Elo (over-corrects) and schedule-Elo challengers | `scripts/nested_validation.py`; `nba_wp/model.py` | nested `*_summary.json` (`calibration_challenger`, `schedule_challenger`) | `tests/test_calibration_challenger.py` |
| `raw_unit_coefficient` must be paired with `raw_intercept` (not the standardized intercept) | `nba_wp/model.py` | `selected_spec.json` (`elo_model.raw_intercept`); `artifacts/workbook_reconciliation.json` | `tests/test_workbook_reconstruction.py` |
| Period boundaries are derived from the data, not hard-coded | `nba_wp/periods.py`; `nba_wp/data.py`; `scripts/select_model.py` | `artifacts/selection_proof.json` (`selection_input_max_date`) | `validate_submission.py` (dynamic date checks) |
| Frozen outer block ignores all in-block outcomes; daily uses only prior dates | `scripts/nested_validation.py` | nested `*_summary.json` (`frozen_block_leakage_guarantee_verified`) | `tests/test_information_policies.py`; `tests/test_leakage_mutations.py` |
| Calibration report contains every candidate | `scripts/nested_validation.py` | nested `*_summary.json`; reliability figures | `tests/test_champion_promotion.py` |
| Primary April is frozen (April outcomes cannot change April prices) | `nba_wp/reporting.py::score_and_write` | `outputs/april_predictions.csv` | `tests/test_primary_april_frozen.py` |
| March probabilities use coefficients fitted through February | `scripts/score_final.py`; `nba_wp/reporting.py::score_and_write` | `outputs/march_predictions.csv` | validator |
| Deployed Elo-only coefficients are refit through March 31 | `score_and_write` | `artifacts/trained_model.joblib`; April output | full recomputation |
| Metrics equal the game-level probabilities | `nba_wp/model.py::evaluate` | `artifacts/final_metrics.json` | `tests/test_artifacts.py`; `validate_submission.py` |
| The Elo-only champion improves the constant baseline | `ablation_table` | `feature_group_ablation.csv`; bootstrap JSON | artifact test plus direct recomputation |
| Saved output is reproducible | `score_and_write` | prediction CSVs and manifest | `validate_submission.py --recompute` |

## Manual spot check

For any row in `outputs/april_predictions.csv`:

1. identify the game and prediction;
2. inspect the same `game_id` in `outputs/engineered_features.csv`;
3. confirm the listed `performance_cutoff`;
4. read the `elo_diff` feature for the game;
5. recompute the final probability from the `elo_model` coefficients in
   `artifacts/selected_spec.json`: `p = sigmoid(intercept + coef · z(elo_diff))`;
6. recompute its log-loss and Brier contribution.

The prediction files contain every quantity required for that calculation.
