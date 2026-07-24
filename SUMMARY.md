# Two-minute summary

## Decision

**Deploy Elo-only.** The champion is a single logistic map on a
margin-of-victory Elo rating differential:

```text
p_home = sigmoid( c + w * z(elo_diff) )
```

fit on all games through March 31. An Elo + Bradley-Terry/recent-trend logit
blend was built, validated, and **rejected**: under honest nested rolling-origin
validation it does not beat Elo-only out-of-sample on either proper score and is
worse calibrated. Rejecting the more complex model is the honest, defensible
call.

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

Each procedure (Elo-only, rank-only, blend) selects its **own** architecture by
its **own** March log loss (Brier tie-break); no procedure is represented by an
architecture chosen for another. March is used for selection, so **March is
in-sample for selection** — reported March metrics are selection-period scores,
not a pristine holdout. The honest out-of-sample evidence is the nested audit.

## Primary April deliverable

`outputs/april_predictions.csv` is the **frozen pre-April** Elo-only file:

- performance state is frozen at March 31;
- the Elo probability map is fit on all rows through March 31;
- no April outcome can change any April price.

`outputs/april_predictions_sequential_backtest.csv` is an optional live-update
simulation; `outputs/challenger_blend_april_predictions.csv` holds the rejected
blend's frozen April prices for transparency.

April has been inspected during development; treat it as a **retrospective
demonstration set**, not an untouched final exam.

## Operational results (Elo-only champion)

| Period | Role | Log loss | Brier | AUC | Accuracy |
|---|---|---:|---:|---:|---:|
| March (Elo-only, fit through Feb) | Selection period (in-sample) | 0.506590 | 0.165880 | 0.823392 | 76.99% |
| **April frozen (primary)** | Holdout deliverable | **0.464369** | **0.149770** | **0.866847** | 78.13% |
| April sequential backtest | Live-update sim only | 0.464648 | 0.149914 | 0.866397 | 78.13% |
| Rejected blend (April frozen) | Challenger, worse | 0.468725 | 0.150465 | 0.864148 | 81.25% |

Primary April file: min \(p\approx0.069\), max \(p\approx0.970\), games with
\(p\ge0.90\): 9 / 96. Mean forecast 0.549 vs. observed 0.594 (mildly
under-forecasting, i.e. not overconfident).

## Nested out-of-sample verdict (501 games, 11 weekly folds, two policies)

| Candidate | LL (frozen / daily) | Brier (frozen / daily) | Cal. slope β |
|---|---:|---:|---:|
| **Elo-only (champion)** | **0.532 / 0.532** | **0.177 / 0.177** | 1.37 / 1.32 |
| Rank-only | 0.550 / 0.549 | 0.184 / 0.184 | 1.70 / 1.64 |
| Blend | 0.548 / 0.547 | 0.183 / 0.182 | 1.80 / 1.75 |

Blend − Elo-only block-bootstrap CIs are entirely above zero on both proper
scores; **0 of 4,000** replicates favored the blend. Champion–challenger
decision under both policies: `keep_elo_only`.

## Interview position

> The pipeline is leakage-audited for same-game box scores and excludes April
> from selection code. An Elo MOV sign bug was corrected. I implemented an Elo +
> rank blend, then a fully nested rolling-origin audit under two information
> policies showed the blend does **not** beat Elo-only out-of-sample and is
> worse calibrated — so I deploy the simpler Elo-only model and keep the blend
> only as a rejected challenger. Every headline file describes the same champion.

Open next:

1. `artifacts/selected_spec.json` (`model_family: elo_only`, rejected challenger)
2. `artifacts/nested_frozen_block_summary.json` / `nested_daily_sequential_summary.json`
3. `outputs/april_predictions.csv`
4. `docs/REVIEWER_GUIDE.md`
