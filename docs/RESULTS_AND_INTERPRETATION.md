# Results and interpretation

The deployed champion is **Elo-only**. All primary numbers below are the
Elo-only champion. The logistic-stacked blend is an implemented-but-**rejected
challenger** and is reported only for comparison.

## Primary April holdout (Elo-only champion, frozen)

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| April frozen (primary) | 0.464369 | 0.149770 | 0.866847 | 78.1250% |

This is the headline deliverable: the deployed Elo-only logistic map, with the
April performance state frozen at March 31, and no April result allowed to
update any April price. Accuracy is 78.1250% on a 96-game sample. The frozen
April price distribution is well-behaved: `home_win_probability` ranges from
approximately 0.069 to 0.970, with 9 of 96 games at or above 0.90. Mean
forecast is 0.549 against an observed home rate of 0.594, so the champion is
mildly **under**-forecasting home wins — a safer failure mode than
overconfidence.

For reference, the rejected blend on the *same* frozen April window scores
log loss 0.468725, Brier 0.150465, AUC 0.864148, accuracy 81.25% — worse than
Elo-only on both proper scores (`outputs/challenger_blend_april_predictions.csv`).

## March (Elo-only, base fit through February, one-step-ahead)

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| March (Elo-only) | 0.506590 | 0.165880 | 0.823392 | 76.9874% |

March is an in-sample diagnostic and a cross-check, **not** the selection
criterion. The deployed Elo architecture is chosen by aggregate frozen-policy
rolling out-of-sample log loss with a one-standard-error stability rule (see
`architecture_selection` in `selected_spec.json`); the single March one-step
split happens to agree. These March numbers are therefore in-sample and should
not be read as an independent test. The honest out-of-sample evidence is the
nested rolling-origin audit below.

## April sequential backtest (live-update simulation, Elo-only)

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| April sequential | 0.464648 | 0.149914 | 0.866397 | 78.1250% |

The sequential backtest lets earlier April dates update team state for later
April dates (base models refit through each prior day). It answers a different
(live-update) question than the frozen primary, so it is reported as a
diagnostic only (`outputs/april_predictions_sequential_backtest.csv`). It
essentially matches the frozen primary here.

## Nested rolling-origin audit — the honest out-of-sample evidence

`scripts/nested_validation.py` runs two clearly separated information policies,
never mixed into one metric: a **frozen-block** policy (per weekly origin, the
performance state is frozen at the day before the block, and models are fit
strictly before the block; verified that mutating any in-block outcome cannot
change any in-block price) and a **daily-sequential** policy (each date's price
uses only strictly-prior dates). Every procedure independently selects its own
architecture by its own inner out-of-fold score; the convex stacker is trained
on inner out-of-fold component predictions. 11 weekly outer folds, 501
out-of-sample games.

Pooled nested metrics (frozen-block / daily-sequential):

| Model | Log loss | Brier | AUC |
|---|---:|---:|---:|
| Constant home rate | 0.688 / 0.688 | 0.247 / 0.247 | — |
| **Elo-only (champion)** | **0.532 / 0.532** | **0.177 / 0.177** | 0.813 / 0.811 |
| Rank-only (BT + trend) | 0.550 / 0.549 | 0.184 / 0.184 | — |
| Blend (rejected) | 0.548 / 0.547 | 0.183 / 0.182 | — |

Block-bootstrap, blend − Elo-only:

- Frozen-block: ΔlogLoss +0.0166 (95% CI [+0.0101, +0.0232]),
  ΔBrier +0.0061 (95% CI [+0.0035, +0.0089]).
- Daily-sequential: ΔlogLoss +0.0149 (95% CI [+0.0076, +0.0220]),
  ΔBrier +0.0055 (95% CI [+0.0025, +0.0086]).

Positive deltas mean the blend is worse. **0 of 4,000 week-block bootstrap
replicates favored the blend** on either metric, under either policy. With only
~11 weekly blocks this is strong *directional* evidence, not production-grade
certainty from a large independent sample. Under both policies the
champion–challenger decision is **keep_elo_only** (promote the blend only if it
beats Elo-only on both log loss and Brier with block-bootstrap upper CI below
zero — not met).

See `artifacts/nested_frozen_block_summary.json`,
`artifacts/nested_daily_sequential_summary.json`, the per-fold
`*_folds.csv`, and per-game `*_predictions.csv`.

## Calibration

Calibration is computed for every candidate (constant, Elo-only, rank-only,
blend) with intercept α, slope β, week-block bootstrap CIs, ECE with
uncertainty, reliability tables, mean-forecast-vs-observed, and tail counts.

- Elo-only, frozen-block: α = −0.05 (95% CI [−0.32, +0.22]),
  β = 1.37 (95% CI [1.22, 1.57]), ECE ≈ 0.059, mean forecast 0.554 vs
  observed 0.557.
- Elo-only, daily-sequential: α = −0.04, β = 1.32 (95% CI [1.19, 1.49]),
  ECE ≈ 0.063.
- Blend: β ≈ 1.75–1.80, ECE ≈ 0.092–0.115 — more compressed toward 0.5 and
  worse calibrated.

β > 1 means Elo-only is mildly **under**confident (probabilities could be
sharpened by roughly 1.3×), a safer failure mode than overconfidence.
Reliability diagrams are in `figures/nested_frozen_block_reliability.png` and
`figures/nested_daily_sequential_reliability.png`.

## Ablation interpretation

The March ablation (`artifacts/feature_group_ablation.csv`) shows:

- a constant training prior is inadequate;
- record differential contains substantial team-strength information;
- Elo materially improves proper scoring;
- Bradley-Terry plus trend provides a strong component ranking;
- **the logistic-stacked blend does not beat Elo-only out-of-sample**, so the
  simpler Elo-only model is deployed and the blend is rejected;
- adding rest and noisy box-score style signals in one rich linear challenger
  does not earn promotion.

## Paired bootstrap

Paired game-level bootstrap log-loss differences are stored in:

- `artifacts/paired_bootstrap_champion_vs_blend.json` (Elo-only vs blend)
- `artifacts/paired_bootstrap_vs_elo.json`
- `artifacts/paired_bootstrap_vs_rank.json`
- `artifacts/paired_bootstrap_vs_constant.json`

March comparisons remain affected by model selection and are reported as
diagnostics; the decisive out-of-sample comparison is the nested audit above.
