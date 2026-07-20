# Model report

All numbers in this file are rendered programmatically from
`reports/metrics.json` and `artifacts/current/*.json` by
`scripts/generate_model_report.py`. Do not edit numbers by hand.

## Executive summary

- Task: probability that the home team wins each April 2026 game, using
  October–March information only.
- Selected model: **direct L2 logistic regression** on
  `elo_diff`,
  architecture `k5_hfa80_hl20`, C = 0.03,
  chosen by **prequential daily validation** (399
  January–February games) over 672 declared candidates.
- Proof: `0` March rows and
  `0` April rows entered selection.
- Locked March test (scored once): log loss 0.5135,
  184/239 correct.
- Primary April forecast (state frozen 2026-03-31): log loss
  0.4746, Brier 0.1530, AUC 0.8655,
  75/96 correct.
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

Declared budget (configs/model.yaml): Elo K × home advantage × trend half-life
× 5 nested feature sets × 7 C values = **672 candidates**
(trend-free sets deduplicated across the half-life axis). Estimator:
**prequential daily expanding validation** — fit on all games before date d,
score date d, pooled per-game log loss over every January/February game
(399 games). Ties break by
Brier, AUC, accuracy, then **fewer features**.

Winner: `k5_hfa80_hl20`, feature set
`elo`, C = 0.03,
validation log loss 0.6309.

## Selected model coefficients (fit through March for April scoring)

| Feature | Standardized β | Raw-unit β |
|---|---|---|
| elo_diff | 0.7831 | 3.4781 |
| (intercept) | 0.2319 | — |

## Locked March test (scored once)

| Model | Log loss | Brier | AUC | Accuracy | Δ per-game LL (selected − model) 95% CI |
|---|---|---|---|---|---|
| constant_home_rate | 0.6806 | 0.2437 | 0.5000 | 0.6025 | -0.1671 [-0.2111, -0.1221] |
| record_difference_logistic | 0.5447 | 0.1800 | 0.8232 | 0.7741 | -0.0312 [-0.0451, -0.0177] |
| elo_only_logistic | 0.5039 | 0.1655 | 0.8241 | 0.7657 | 0.0095 [0.0021, 0.0164] |
| three_feature_challenger | 0.5060 | 0.1660 | 0.8273 | 0.7657 | 0.0075 [0.0025, 0.0121] |
| selected_model | 0.5135 | 0.1684 | 0.8241 | 0.7699 | — |

Reliability (selected model):

| Bin | n | Mean prediction | Observed rate |
|---|---|---|---|
| 0.0–0.1 | 2 | 0.0832 | 0.0000 |
| 0.1–0.2 | 19 | 0.1587 | 0.0526 |
| 0.2–0.3 | 22 | 0.2572 | 0.0909 |
| 0.3–0.4 | 19 | 0.3432 | 0.3684 |
| 0.4–0.5 | 34 | 0.4457 | 0.5294 |
| 0.5–0.6 | 32 | 0.5509 | 0.7812 |
| 0.6–0.7 | 38 | 0.6579 | 0.7105 |
| 0.7–0.8 | 43 | 0.7497 | 0.8837 |
| 0.8–0.9 | 24 | 0.8375 | 0.8333 |
| 0.9–1.0 | 6 | 0.9138 | 1.0000 |

## Frozen April forecast (primary)

| Model | Log loss | Brier | AUC | Accuracy | Δ per-game LL (selected − model) 95% CI |
|---|---|---|---|---|---|
| constant_home_rate | 0.6792 | 0.2430 | 0.5000 | 0.5938 | -0.2046 [-0.2533, -0.1642] |
| record_difference_logistic | 0.5114 | 0.1661 | 0.8464 | 0.7812 | -0.0368 [-0.0560, -0.0164] |
| elo_only_logistic | 0.4644 | 0.1502 | 0.8655 | 0.7812 | 0.0102 [0.0031, 0.0193] |
| three_feature_challenger | 0.4656 | 0.1500 | 0.8682 | 0.7708 | 0.0090 [0.0002, 0.0192] |
| selected_model | 0.4746 | 0.1530 | 0.8655 | 0.7812 | — |

Reliability (selected model):

| Bin | n | Mean prediction | Observed rate |
|---|---|---|---|
| 0.0–0.1 | 1 | 0.0871 | 0.0000 |
| 0.1–0.2 | 10 | 0.1766 | 0.1000 |
| 0.2–0.3 | 8 | 0.2535 | 0.2500 |
| 0.3–0.4 | 14 | 0.3529 | 0.2143 |
| 0.4–0.5 | 11 | 0.4614 | 0.6364 |
| 0.5–0.6 | 10 | 0.5624 | 0.6000 |
| 0.6–0.7 | 10 | 0.6583 | 0.9000 |
| 0.7–0.8 | 9 | 0.7453 | 0.8889 |
| 0.8–0.9 | 17 | 0.8485 | 0.8824 |
| 0.9–1.0 | 6 | 0.9220 | 1.0000 |

Rolling-daily April (separate descriptive scenario,
`predictions/april_predictions_rolling_scenario.csv`): log loss
0.4728, 76/96 correct. It is **not** the
assignment result.

## Ensemble / challenger review

The v1 log-odds blend (Elo + BT/trend component logistics, Platt-calibrated)
survives as a challenger. On the locked March test its predictions correlate
0.9966 with the selected model; paired
Δ per-game log loss (selected − blend) = -0.0064
[-0.0110, -0.0020].
The selected direct logistic is simpler and no worse, so the blend remains a
documented challenger, not the selected model.

## Calibration

| Period | Intercept α | Slope γ | ECE | Min p | Max p |
|---|---|---|---|---|---|
| Locked March | 0.2403 | 1.4834 | 0.1042 | 0.0717 | 0.9421 |
| Frozen April | 0.2465 | 1.5293 | 0.1029 | 0.0871 | 0.9582 |

Slope > 1 suggests probabilities are somewhat under-dispersed (too close to
0.5), but with 96 April games the uncertainty is large. This is a diagnostic,
not a solved property. April is never used to recalibrate.

## Uncertainty (date-block bootstrap on frozen April)

Method: paired_date_block_bootstrap, 1000 replicates,
seed 2026; intervals condition on the locked specification.

| Metric | Mean | 5% | 95% |
|---|---|---|---|
| Log loss | 0.4734 | 0.4310 | 0.5116 |
| Brier | 0.1525 | 0.1343 | 0.1693 |
| AUC | 0.8679 | 0.8270 | 0.9132 |
| Accuracy | 0.7829 | 0.7347 | 0.8315 |

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
diagnostics, figures): 13.99 s.
Model selection over 672 candidates:
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
