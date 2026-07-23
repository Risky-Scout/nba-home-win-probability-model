# Two-minute summary

## Decision

Use a calibrated log-odds blend of:

- a margin-of-victory Elo component;
- a Bradley-Terry team-strength component;
- a recent point-margin trend correction.

Deployed blend is a **convex logit mix** (temperature floor \(T \ge 1\)) so
correlated Elo/rank signals cannot freely sharpen into 99%+ prices.

## Leakage control

A game row is processed in this order:

1. read both teams' states;
2. calculate and store the pregame feature row;
3. finish all games on that date;
4. observe outcomes and update state.

Current-game points, turnovers, fouls, and rebounds cannot enter their own
prediction. Model selection receives only rows dated no later than March 31.

**Elo MOV** uses winner Elo − loser Elo (not home − away for every game).

## Selection and March honesty

Five declared architectures are scored by unconstrained March log loss.
The stacker is also fit on March component logits. **March is therefore
in-sample for the blend and for architecture choice** — reported March
metrics are selection-period scores, not a pristine holdout.

Deploy coefficients enforce \(T \ge 1\) (no sharpening).

## Primary April deliverable

`outputs/april_predictions.csv` is the **frozen pre-April** file:

- no April outcomes update team performance state;
- base-model generator matches the March stacker fit (through February).

`outputs/april_predictions_sequential_backtest.csv` is an optional live-update
simulation only.

April has been inspected during development; treat it as a **retrospective
demonstration set**, not an untouched final exam.

## Operational results

| Period | Role | Log loss | Brier | AUC | Accuracy |
|---|---|---:|---:|---:|---:|
| March (deploy \(T\ge1\)) | Selection / stacker train | 0.508373 | 0.164854 | 0.830044 | 76.99% |
| March (unconstrained stacker) | Selection surface only | 0.488026 | 0.157018 | 0.830044 | 78.66% |
| **April frozen (primary)** | Holdout deliverable | **0.484432** | 0.155823 | 0.862798 | 81.25% |
| April sequential backtest | Live-update sim only | 0.474545 | 0.152112 | 0.852901 | 82.29% |

Primary April file: max \(p\approx0.948\), games with \(p\ge0.90\): **4 / 96**.

## Interview position

> The pipeline is leakage-audited for same-game box scores and excludes April
> from selection code. An Elo MOV sign bug was corrected. The stacker cannot
> unconstrained-sharpen. March scores are disclosed as in-sample for the blend.
> Primary April prices are frozen pre-April.

Open next:

1. `artifacts/selection_proof.json`
2. `artifacts/selected_spec.json` (notes + temperature floor)
3. `outputs/april_predictions.csv`
4. `docs/REVIEWER_GUIDE.md`
