# One-pager — say this if you only get five minutes

## Decision

Direct L2 logistic on three leakage-safe features:

\[
\operatorname{logit}\hat p=\beta_0+\beta_1\Delta\mathrm{Elo}+\beta_2\Delta\mathrm{BT}+\beta_3\Delta\mathrm{Trend}
\]

Selected on **January/February** expanding folds by **mean log loss**:
\(K=10\), half-life \(20\) days, \(C=0.1\).

## Primary deliverable

`outputs/april_predictions_frozen_snapshot.csv` — April scored from the
**March 31** state (literal assignment). Sequential April is sensitivity only.

## Numbers to have memorized

| Item | Value | File |
|---|---|---|
| Full-sample home rate | ≈ 55.4% | `artifacts/current/data_audit.json` |
| Pre-March selection end | 2026-02-28 | `pre_march_selection_proof.json` |
| Candidates searched | 72 (63+9) | `architecture_candidates.json` |
| Locked March correct | 182/239 | `final_metrics.json` |
| Frozen April | LL≈0.469, 74/96 | same / frozen CSV |
| Largest standardized coef | elo_diff ≈ 0.73 | `model_coefficients.json` |

## Three honesty lines

1. April previously viewed → retrospective, not pristine.  
2. No bookmaker odds → no alpha / CLV claim.  
3. April lift vs Elo unresolved under date-block intervals.

## Where every parameter lives

`docs/interview/PARAMETER_LEDGER.md`

## Navigation

`START_HERE.md` → script → ledger → Q&A.
