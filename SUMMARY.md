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

Five declared architectures are evaluated. For each architecture, the blend is
**fitted by penalized maximum likelihood (logistic stacking)** on March
component logits — no grid search. The selection rule is: minimize March log
loss. The selected specification is generated in
`artifacts/selected_spec.json`.

## Operational results

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| March | 0.488029 | 0.157148 | 0.830336 | 78.6611% |
| April | 0.458717 | 0.144995 | 0.853351 | 82.2917% |

## Interview position

> The complete evidence chain is reproducible, April is excluded from
> selection code, the blend coefficients are fitted — not searched — and
> probability quality is strong and reported without post-hoc retuning.

Open next:

1. `artifacts/selection_proof.json`
2. `artifacts/feature_group_ablation.csv`
3. `outputs/april_predictions.csv`
4. `docs/REVIEWER_GUIDE.md`
