# Two-minute summary (audit remediation)

## Preservation

Original submission tag: `v1-original-submission`.  
This branch remediates statistical governance without hiding that history.

Authoritative remediation artifacts: `artifacts/current/`.  
Legacy v1 search/bootstrap files: `artifacts/v1_legacy/` (non-authoritative).

## Decision

Select a **direct L2 logistic** (or blend challenger only if it wins) on
`elo_diff`, `bt_logit`, and `trend_diff` using **pre-March expanding-window
validation** (January and February folds). Primary metric: mean validation
log loss. External benchmark floats do **not** gate selection.

Search budget: **63 direct-logistic candidates plus nine architecture-matched
blend challengers (72 total)**.

## Leakage and cutoffs

- Feature-before-update; same-day batching retained.
- Selection input ends **2026-02-28**. March and April rows are forbidden.
- March is a **locked test** after the specification is frozen.
- **Primary April result** is the **frozen March 31** snapshot.
- Sequential April scoring is operational sensitivity only.

## Claims discipline

- April is the assignment’s retrospective scoring period. The executable
  selection pipeline uses zero April rows, but April had previously been
  viewed during the broader project, so I do not claim that it is a pristine
  untouched holdout.
- Pre-March selection is a reconstructed governance path, not historical
  preregistration.
- The direct model won pre-March validation; April incremental value over Elo
  remains statistically unresolved.
- Outputs are model-estimated fair probabilities / zero-margin fair odds.
- No sportsbook alpha or profitability claim.
- Readiness claim: **prototype / research**, not deployable production pricing.

## Where to look

1. `docs/AUDIT_RESPONSE.md`
2. `artifacts/current/pre_march_selection_proof.json`
3. `artifacts/current/selected_spec_pre_march.json`
4. `outputs/april_predictions_frozen_snapshot.csv`
5. `artifacts/current/date_block_bootstrap_summary.json`
6. `docs/BENCHMARK_PROVENANCE.md`
7. `docs/MARKET_PRICING_LIMITATIONS.md`
