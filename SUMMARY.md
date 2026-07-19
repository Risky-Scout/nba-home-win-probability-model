# Two-minute summary

## Decision

Use a calibrated log-odds blend of:

- a margin-of-victory Elo component;
- a Bradley-Terry team-strength component;
- a recent point-margin trend correction.

The model remains fast, probabilistic, and coefficient-level interpretable.

## Leakage control

A game row is processed in this order:

1. read both teams' states;
2. calculate and store the pregame feature row;
3. finish all games on that date;
4. observe outcomes and update state.

Current-game points, turnovers, fouls, and rebounds cannot enter their own
prediction. Model selection receives only rows dated no later than March 31.

## Selection

Five declared architectures are evaluated. For each architecture, a
three-parameter calibration grid is searched on March. Candidates must exceed
all four March numerical targets. The selected specification is generated in
`artifacts/selected_spec.json`.

## Operational results

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| March | 0.487569 | 0.156834 | 0.831798246 | 77.8243% |
| April | 0.463375 | 0.145639 | 0.850202 | 83.3333% |

March exceeds all four rounded targets. April exceeds log loss, Brier, and
accuracy but misses the AUC target.

## Interview position

The strongest claim is not "every target was beaten." It is:

> The complete evidence chain is reproducible, April is excluded from
> selection code, probability quality is strong, and the AUC miss is reported
> without post-hoc retuning.

Open next:

1. `artifacts/selection_proof.json`
2. `artifacts/feature_group_ablation.csv`
3. `outputs/april_predictions.csv`
4. `docs/INTERVIEW_WALKTHROUGH.md`
