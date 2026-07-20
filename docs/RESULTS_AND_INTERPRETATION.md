# Results and interpretation (audit remediation)

After running `make reproduce`, interpret artifacts under `artifacts/current/`.

## Primary April result (assignment)

`outputs/april_predictions_frozen_snapshot.csv` and
`final_metrics.json → primary_april_result`.

| Metric | Frozen April (primary) |
|---|---|
| Games | 96 |
| Correct | 74 |
| Accuracy | 77.08% |
| Log loss | 0.469453 |
| Brier | 0.151060 |
| AUC | 0.862348 |

## Sequential April (sensitivity only)

`final_metrics.json → sequential_daily.april`.

Produces 76/96 correct but slightly worse log loss and Brier than the frozen
primary. Do **not** headline sequential April as the assignment result.

## Locked March test

`final_metrics.json → locked_march_test`.

Exact correct-game count: **182 / 239** (76.15%).

March was not used to select the specification. Do not describe tiny
AUC/accuracy gaps versus retrospective reference floats as meaningful wins.

## Model comparison (prominent)

Date-block bootstrap differences versus Elo on frozen April do **not** provide
convincing evidence that the three-feature champion beats simpler Elo.
Proper-score differences are small and include zero; the log-loss point
estimate can be slightly unfavorable to the champion.

Correct claim:

> The direct model won the declared pre-March validation process, but its
> incremental April value over Elo remains statistically unresolved.

## Calibration (diagnostic, not solved)

Frozen-April approximate diagnostics:

| Statistic | Value |
|---|---|
| ECE | ≈ 0.113 |
| Intercept \(\alpha\) | ≈ 0.256 |
| Slope \(\gamma\) | ≈ 1.437 |

A slope above one suggests probabilities may be insufficiently dispersed
(too close to 0.50). Treat this as a finding, not confirmation that calibration
is solved. Do not recalibrate using April.

## Uncertainty caveat

`date_block_bootstrap_summary.json` intervals condition on the locked
specification. They estimate evaluation uncertainty on the observed April
schedule, not combined candidate-search + selection + fitting uncertainty.

## Market language

Use model-estimated fair probability / zero-margin fair odds only:

\[
\text{Fair decimal odds} = 1 / p
\]

No alpha or profitability claims.
