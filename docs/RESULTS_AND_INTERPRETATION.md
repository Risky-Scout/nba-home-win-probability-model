# Results and interpretation

## Primary April holdout (frozen pre-April)

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| April frozen (primary) | 0.484432 | 0.155823 | 0.862798 | 81.2500% |

This is the headline deliverable: base models trained through February, the
deploy stacker (temperature floor T >= 1), and no April result allowed to update
any April performance state. Accuracy is 81.25% (78 of 96 games) on a 96-game
sample. The frozen April price distribution is well-behaved: maximum
`home_win_probability` approx 0.948, 4 of 96 games at or above 0.90, and none
at or above 0.95.

## March (selection / stacker-training period)

| Stacker | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| Deploy (T >= 1) | 0.508373 | 0.164854 | 0.830044 | 76.9874% |
| Unconstrained (selection surface) | 0.488026 | 0.157018 | 0.830044 | 78.6611% |

March is the selection set: the architecture minimizing **unconstrained** March
log loss is chosen, and the blend coefficients are fitted on March component
logits by penalized maximum likelihood. March numbers are therefore in-sample
for the stacker and should not be read as an independent test. The deployed
model applies the temperature floor (a + b = 1), which trades a little in-sample
March log loss for honest, non-sharpened probabilities out of sample.

## April sequential backtest (live-update simulation)

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| April sequential | 0.474545 | 0.152112 | 0.852901 | 82.2917% |

The sequential backtest lets earlier April dates update team state for later
April dates. It scores slightly better than the frozen version but answers a
different (live-update) question, so it is reported as a diagnostic only.

## Ablation interpretation

The March ablation shows:

- a constant training prior is inadequate;
- record differential contains substantial team-strength information;
- Elo materially improves proper scoring;
- Bradley-Terry plus trend provides the strongest component ranking;
- the logistic-stacked blend improves probability quality over either
  component;
- adding rest and noisy box-score style signals in one rich linear challenger
  does not earn promotion.

See `artifacts/feature_group_ablation.csv`.

## Permutation interpretation

See `artifacts/permutation_importance.csv` for the regenerated mean March
log-loss increases after shuffling each feature. Bradley-Terry remains the
dominant ranking signal, Elo meaningful, and trend a smaller correction.

## Paired bootstrap

On March, paired game-level bootstrap log-loss differences for
`final minus component` are stored in:

- `artifacts/paired_bootstrap_vs_elo.json`
- `artifacts/paired_bootstrap_vs_rank.json`
- `artifacts/paired_bootstrap_vs_constant.json`

Both component comparisons remain affected by March model selection and are
reported as diagnostics, not decisive generalization results.

## Calibration

Open:

- `figures/march_calibration.png`
- `figures/april_calibration.png`
- `artifacts/march_calibration_bins.csv`
- `artifacts/april_calibration_bins.csv`

The stacker coefficients are fitted on March, so calibration should be treated
as a monitored layer, not a permanent universal constant.
