# One-pager: say this if you get only five minutes

## Decision

Calibrated log-odds blend of margin Elo and Bradley-Terry + recent trend.
Primary objective: home-win **probability** quality (log loss / Brier).

\[
z=0.19\,\mathrm{logit}(p_E)+0.81\,\mathrm{logit}(p_R),\quad
p=\sigma(z/0.59+0.33)
\]

(Temperature divides full \(z\).)

## Pipeline in one breath

Audit data → build leakage-safe chronological features → fit Elo and
BT/trend components through February → select architecture + calibration on
March → freeze spec → refit through March → score April → export diagnostics.

## Leakage rule

For each date: read state → write all feature rows → then update. Same-day
batched. No current-game box score in its own prediction. Selection input hard
truncated at March 31.

## Selected spec

`hfa_75`: Elo K=10, HFA=75, BT C=0.15, trend 45d / 10g; blend w=0.19,
temperature=0.59, shift=0.33.

## Results to remember

| | Log loss | Brier | AUC | Acc |
|---|---:|---:|---:|---:|
| March | 0.488 | 0.157 | 0.832 | 77.8% |
| April | 0.463 | 0.146 | 0.850 | 83.3% |

March clears rounded targets. April wins proper scores + accuracy, **misses
AUC**. No post-hoc retune.

## Proof files

`selection_proof.json` · `feature_group_ablation.csv` · `final_metrics.json` ·
`april_predictions.csv` · `validate_submission.py`

## Closing line

> Reproducible evidence chain, April excluded from selection code, strong
> probability scores, AUC miss reported honestly.
