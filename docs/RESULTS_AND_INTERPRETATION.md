# Results and interpretation

## Operational one-step-ahead

| Period | Model log loss | Target | Model Brier | Target | Model AUC | Target | Model accuracy | Target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| March | 0.487569 | 0.509645 | 0.156834 | 0.167618 | 0.831798246 | 0.831798000 | 77.8243% | 77.8200% |
| April | 0.463375 | 0.468596 | 0.145639 | 0.150628 | 0.850202 | 0.868196 | 83.3333% | 81.2500% |

### March

March is the selection set. The champion exceeds all four rounded targets, but:

- the AUC difference is approximately \(2.46\times 10^{-7}\);
- the accuracy difference is one discrete classification boundary relative to
  the rounded target;
- neither margin should be described as a decisive generalization result.

### April

April proper scores are favorable:

\[
\Delta LL = 0.463375 - 0.468596 = -0.005221,
\]

\[
\Delta Brier = 0.145639 - 0.150628 = -0.004989.
\]

Accuracy is 83.33%, two correctly classified games above an 81.25% rate on a
96-game sample.

AUC is 0.8502, below 0.8682. The model therefore does not meet the objective on
all four April metrics.

## Frozen snapshot sensitivity

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| March | 0.497736 | 0.162489 | 0.822442 | 76.1506% |
| April | 0.458942 | 0.145299 | 0.862798 | 80.2083% |

The frozen April version has stronger proper scores and ranking than the
operational version, but lower 0.5-threshold accuracy.

## Ablation interpretation

The March ablation shows:

- a constant training prior is inadequate;
- record differential contains substantial team-strength information;
- Elo materially improves proper scoring;
- Bradley-Terry plus trend provides the strongest component ranking;
- the calibrated blend improves probability quality over either component;
- adding rest and noisy box-score style signals in one rich linear challenger
  does not earn promotion.

See `artifacts/feature_group_ablation.csv`.

## Permutation interpretation

Mean March log-loss increase after shuffling:

| Feature | Mean increase |
|---|---:|
| Bradley-Terry logit | 0.2325 |
| Elo difference | 0.0205 |
| Trend difference | 0.0061 |

Trend has a small lower-tail value below zero across repeated permutations,
which is consistent with a weak correction rather than a dominant signal.


## Paired bootstrap

On March, paired game-level bootstrap log-loss differences for
`final minus component` are:

| Comparison | Observed difference | 95% interval |
|---|---:|---:|
| Final - Elo component | -0.01993 | [-0.04538, 0.00879] |
| Final - rank component | -0.04156 | [-0.07528, -0.00224] |

The interval versus Elo includes zero, so the blend's advantage over Elo alone
is not statistically decisive under this resampling diagnostic. The interval
versus the rank component is favorable, but both comparisons remain affected
by March model selection.

## Calibration

Open:

- `figures/march_calibration.png`
- `figures/april_calibration.png`
- `artifacts/march_calibration_bins.csv`
- `artifacts/april_calibration_bins.csv`

The selected temperature sharpens probabilities. Calibration should be treated
as a monitored layer, not a permanent universal constant.
