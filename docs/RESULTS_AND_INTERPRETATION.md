# Results and interpretation

## Operational one-step-ahead

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| March | 0.488029 | 0.157148 | 0.830336 | 78.6611% |
| April | 0.458717 | 0.144995 | 0.853351 | 82.2917% |

### March

March is the selection set: the architecture minimizing March log loss is
chosen, and the blend coefficients are fitted on March component logits by
penalized maximum likelihood. March numbers are therefore in-sample for the
stacker and should not be read as an independent test.

### April

April proper scores are favorable relative to March, and accuracy is 82.29%
(79 of 96 games) on a 96-game sample. Small AUC or accuracy movements at this
sample size are not statistically decisive.

## Frozen snapshot sensitivity

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| March | 0.496766 | 0.162053 | 0.823026 | 75.7322% |
| April | 0.454728 | 0.144498 | 0.863698 | 80.2083% |

The frozen April version has stronger proper scores and ranking than the
operational version, but lower 0.5-threshold accuracy.

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
