# Model report

All numbers in this file are rendered programmatically from
`reports/metrics.json` and `artifacts/current/*.json` by
`scripts/generate_model_report.py`. Do not edit numbers by hand.

## Executive summary

- Task: probability that the home team wins each April 2026 game, using
  October–March information only.
- Selected model: **direct L2 logistic regression** on three leakage-controlled
  features (`elo_diff`, `bt_logit`, `trend_diff`), architecture
  `k10_hl20`, regularization C = 0.1.
- Selection used **only pre-March expanding folds** (January and February
  validation). Proof: `0` March rows and
  `0` April rows entered selection.
- Locked March test (scored once): log loss 0.5103,
  182/239 correct.
- Primary April forecast (state frozen 2026-03-31): log loss
  0.4695, Brier 0.1511, AUC 0.8623,
  74/96 correct.
- The selected model decisively beats the constant-rate and record-difference
  baselines. Its difference from a single-feature **Elo-only logistic is
  statistically unresolved** (95% CI for Δ per-game log loss includes zero on
  both March and April). We keep the declared 72-candidate selection outcome
  and record Elo-only as the pre-registered simpler challenger for future data.

## Data audit

- 1230 games, 30 teams,
  2025-10-21 → 2026-04-12; every team plays 82 games:
  True.
- Home-win rate: 0.5545.
- Missing values: 0; duplicate game ids:
  0; tied scores: 0.
- Pregame wins/losses reconcile exactly against replayed results:
  0 mismatches.

## Target and leakage controls

Target: `home_win = 1` iff `home_points > away_points`. Postgame box-score
columns never enter same-game features; team state updates **after** features
are written; same-date games are batched. Guards raise on March-or-later rows
in selection (`nba_wp/selection.py`), with tests in
`tests/test_feature_timing.py` and `tests/test_temporal_protocol.py`.

## Temporal protocol

| Stage | Period | Role |
|---|---|---|
| Development | Oct–Feb | Feature and model development |
| Fold 1 | train < Jan → validate Jan | Selection |
| Fold 2 | train < Feb → validate Feb | Selection |
| Locked test | March | Scored once after freeze |
| Forecast | April | Frozen 2026-03-31 refit on Oct–Mar |

A third fold (train Oct–Nov → validate Dec) was evaluated as a **protocol
sensitivity**: with only ~250 training games it pushes selection toward heavier
shrinkage (C = 0.01) that scores worse when refit on five months of data. The
declared two-fold protocol stands; the sensitivity is disclosed here rather
than silently absorbed.

## Selection process

Declared budget: 3 Elo K × 3 trend half-lives × 7 C values = 63 direct
logistics + 9 architecture-matched blend challengers = **72
candidates**. Primary metric: mean validation log loss; ties break by Brier,
then AUC, accuracy, model type, architecture name.

Winner: `k10_hl20`, C = 0.1,
mean validation log loss 0.6319.

## Selected model coefficients (fit through March for April scoring)

| Feature | Standardized β | Raw-unit β |
|---|---|---|
| elo_diff | 0.7321 | 2.1595 |
| bt_logit | 0.1560 | 0.2931 |
| trend_diff | 0.0617 | 0.0173 |
| (intercept) | 0.2385 | — |

## Locked March test (scored once)

| Model | Log loss | Brier | AUC | Accuracy | Δ per-game LL (selected − model) 95% CI |
|---|---|---|---|---|---|
| constant_home_rate | 0.6806 | 0.2437 | 0.5000 | 0.6025 | -0.1703 [-0.2146, -0.1244] |
| record_difference_logistic | 0.5447 | 0.1800 | 0.8232 | 0.7741 | -0.0344 [-0.0504, -0.0186] |
| elo_only_logistic | 0.5079 | 0.1664 | 0.8232 | 0.7741 | 0.0024 [-0.0019, 0.0063] |
| selected_model | 0.5103 | 0.1672 | 0.8260 | 0.7615 | — |

Reliability (selected model):

| Bin | n | Mean prediction | Observed rate |
|---|---|---|---|
| 0.0–0.1 | 4 | 0.0882 | 0.0000 |
| 0.1–0.2 | 19 | 0.1510 | 0.0526 |
| 0.2–0.3 | 23 | 0.2594 | 0.0870 |
| 0.3–0.4 | 23 | 0.3577 | 0.4783 |
| 0.4–0.5 | 29 | 0.4488 | 0.5517 |
| 0.5–0.6 | 30 | 0.5500 | 0.7667 |
| 0.6–0.7 | 36 | 0.6566 | 0.6667 |
| 0.7–0.8 | 40 | 0.7529 | 0.9000 |
| 0.8–0.9 | 27 | 0.8371 | 0.8889 |
| 0.9–1.0 | 8 | 0.9186 | 0.8750 |

## Frozen April forecast (primary)

| Model | Log loss | Brier | AUC | Accuracy | Δ per-game LL (selected − model) 95% CI |
|---|---|---|---|---|---|
| constant_home_rate | 0.6792 | 0.2430 | 0.5000 | 0.5938 | -0.2097 [-0.2609, -0.1676] |
| record_difference_logistic | 0.5114 | 0.1661 | 0.8464 | 0.7812 | -0.0420 [-0.0650, -0.0222] |
| elo_only_logistic | 0.4672 | 0.1507 | 0.8691 | 0.7812 | 0.0023 [-0.0022, 0.0066] |
| selected_model | 0.4695 | 0.1511 | 0.8623 | 0.7708 | — |

Reliability (selected model):

| Bin | n | Mean prediction | Observed rate |
|---|---|---|---|
| 0.0–0.1 | 1 | 0.0825 | 0.0000 |
| 0.1–0.2 | 10 | 0.1559 | 0.1000 |
| 0.2–0.3 | 8 | 0.2409 | 0.1250 |
| 0.3–0.4 | 13 | 0.3375 | 0.3077 |
| 0.4–0.5 | 11 | 0.4493 | 0.6364 |
| 0.5–0.6 | 8 | 0.5449 | 0.3750 |
| 0.6–0.7 | 13 | 0.6552 | 0.9231 |
| 0.7–0.8 | 10 | 0.7561 | 0.9000 |
| 0.8–0.9 | 14 | 0.8570 | 0.8571 |
| 0.9–1.0 | 8 | 0.9273 | 1.0000 |

Rolling-daily April (separate descriptive scenario,
`predictions/april_predictions_rolling_scenario.csv`): log loss
0.4705, 76/96 correct. It is **not** the
assignment result.

## Ensemble / challenger review

The v1 log-odds blend (Elo + BT/trend component logistics, Platt-calibrated)
survives as a challenger. On the locked March test its predictions correlate
0.9893 with the selected model; paired
Δ per-game log loss (selected − blend) = -0.0104
[-0.0191, -0.0014].
The selected direct logistic is simpler and no worse, so the blend remains a
documented challenger, not the selected model.

## Calibration

| Period | Intercept α | Slope γ | ECE | Min p | Max p |
|---|---|---|---|---|---|
| Locked March | 0.2666 | 1.4079 | 0.1106 | 0.0671 | 0.9557 |
| Frozen April | 0.2563 | 1.4367 | 0.1133 | 0.0825 | 0.9652 |

Slope > 1 suggests probabilities are somewhat under-dispersed (too close to
0.5), but with 96 April games the uncertainty is large. This is a diagnostic,
not a solved property. April is never used to recalibrate.

## Uncertainty (date-block bootstrap on frozen April)

Method: paired_date_block_bootstrap, 1000 replicates,
seed 2026; intervals condition on the locked specification.

| Metric | Mean | 5% | 95% |
|---|---|---|---|
| Log loss | 0.4684 | 0.4263 | 0.5052 |
| Brier | 0.1505 | 0.1324 | 0.1663 |
| AUC | 0.8651 | 0.8244 | 0.9091 |
| Accuracy | 0.7722 | 0.7272 | 0.8214 |

## Feature-set sensitivity (pre-March folds only)

| Feature set / C | Mean validation log loss |
|---|---|
| elo_C0.1 | 0.63073 |
| elo_bt_C0.1 | 0.63087 |
| elo_C0.3 | 0.63130 |
| elo_C0.03 | 0.63137 |
| elo_bt_C0.3 | 0.63142 |
| elo_bt_C0.03 | 0.63155 |

The Elo-only single feature achieves essentially the same pre-March validation
log loss as the three-feature model. Because the declared search did not
include feature-set pruning, and any post-hoc switch would now be informed by
locked-test results, the three-feature winner stands and Elo-only is recorded
as the pre-registered simpler challenger for future seasons.

## Feature drift

Monthly means/std/quantiles and max standardized distance vs the Oct–Feb
training distribution: `reports/feature_drift_monthly.csv`. The engineered
features are differences of bounded team states (not mechanically growing
cumulative sums), and April values stay within the training envelope.

## Computational performance

End-to-end scoring (feature rebuild, final fit, locked-test + frozen scoring,
diagnostics, figures): 14.26 s.
Model selection over 72 candidates:
nan s.

## Limitations

1. April was viewed during the wider project before this protocol; it is a
   retrospective scoring period, not a pristine holdout.
2. No bookmaker prices → no market-edge, CLV, or profitability claims.
3. One season, 96 forecast games → wide intervals; Elo-only equivalence
   unresolved.
4. No injuries, lineups, travel, or rest-model in the supplied data beyond
   schedule-derived features.

## Recommended production extensions

Time-stamped market odds ingestion and de-vigging; lineup/injury feeds;
rolling refits with monitoring and rollback; per-team calibration monitoring;
multi-season backtesting before any pricing use.
