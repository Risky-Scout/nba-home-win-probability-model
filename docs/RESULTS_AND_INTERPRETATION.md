# Results and interpretation (audit remediation)

After running `make reproduce`, interpret artifacts as follows.

## Primary April result (assignment)

`outputs/april_predictions_frozen_snapshot.csv` and
`final_metrics.json → primary_april_result`.

This is the assignment-aligned frozen March 31 information set. All April games
are scored from the March 31 performance-state snapshot without consuming April
outcomes.

Representative values after remediation (direct logistic, pre-March selected):

| Metric | Frozen April (primary) |
|---|---|
| Games | 96 |
| Log loss | ≈ 0.469 |
| Brier | ≈ 0.151 |
| AUC | ≈ 0.862 |
| Accuracy | 74 / 96 (≈ 77.1%) |

## Sequential April (sensitivity only)

`final_metrics.json → sequential_daily.april` and
`outputs/april_predictions.csv`.

Earlier April results update ratings; later April games use those updates.
Operationally useful, but **not** the headline assignment result.

## Locked March test

`final_metrics.json → locked_march_test`.

March was **not** used to select the specification on this branch. Report exact
correct-game counts. Do not describe tiny AUC/accuracy gaps versus retrospective
reference floats as meaningful wins.

Example wording:

> The model produced lower March log loss and Brier score than the retrospective
> reference values. March AUC and accuracy were effectively ties at the reported
> precision when differences are only rounding artifacts. Exact correct-game
> count: see `locked_march_test.correct_games` / 239.

## Calibration

See `artifacts/calibration_diagnostics.json` and
`artifacts/extreme_probability_audit.csv`.

Interpretation:

- \(\alpha \approx 0\): calibration-in-the-large
- \(\gamma \approx 1\): appropriate sharpness
- \(\gamma < 1\): overconfident
- \(\gamma > 1\): too conservative

Do not recalibrate using April.

## Uncertainty

See `artifacts/date_block_bootstrap_summary.json`.

Paired date-block intervals condition on the locked specification. Differences
versus Elo and rank-component baselines are included when those columns are
present.

## Market language

Use model-estimated fair probability / zero-margin fair odds only:

\[
\text{Fair decimal odds} = 1 / p
\]

This conversion does not include overround, market consensus, liability, limits,
injuries, news, or trader adjustments. No alpha or profitability claims.
