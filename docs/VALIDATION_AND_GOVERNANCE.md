# Validation and governance (audit remediation)

## Chronological periods

| Period | Role |
|---|---|
| Oct–Dec / Jan; Oct–Jan / Feb | Pre-March selection folds |
| March | Locked test after specification freeze |
| April frozen at March 31 | Primary assignment scoring |
| April sequential | Operational sensitivity |

## Selection input contract

`python -m nba_wp.cli select` truncates to `game_date < 2026-03-01`.
`run_selection` / `assert_pre_march_selection_frame` raise if any March-or-later
row is present.

Proof: `artifacts/current/pre_march_selection_proof.json`.

This is a **reconstructed governance path** on the remediation branch. It is
not claimed as historical preregistration before April was viewed.

## Search budget

Declared in `configs/model.yaml`.

| Hyperparameter | Candidate values | Why it exists | Why these values |
|---|---|---|---|
| Elo \(K\) | 10, 20, 30 | Sequential Elo adaptation speed | Slow / moderate / fast |
| Trend half-life (days) | 20, 45, 90 | Form memory | Short / medium / long |
| Logistic \(C\) | 0.01, 0.03, 0.1, 0.3, 1, 3, 10 | L2 inverse-regularization | Log-spaced shrinkage |

Direct logistics: \(3 \times 3 \times 7 = 63\).

Plus **one Platt-calibrated blend challenger per feature architecture**
(\(3 \times 3 = 9\)).

**Total: 63 direct-logistic candidates plus nine architecture-matched blend
challengers, for 72 candidates total.**

### Why the dense temperature/shift grid was removed

For \(T > 0\), \(p=\sigma(z/T+b)\) is strictly monotonic in \(z\). Changing
temperature or shift cannot change AUC/ranking and does not justify treating
hundreds of thousands of \((T,b)\) pairs as distinct ranking models.

Blend-challenger Platt calibration is fit on **training** predictions and
labels only, then applied to validation. Validation labels are not used to fit
Platt during fold scoring.

## Selection objective

- **Primary:** mean validation log loss across January and February folds.
- **Secondary:** Brier score (tie-break).
- **Descriptive only:** AUC and accuracy.
- **Not used:** external benchmark floats as a hard gate.

## Benchmarks

`configs/model.yaml (benchmarks section)` values are retrospective references only.
Provenance: `docs/BENCHMARK_PROVENANCE.md`.

## April exposure

April is the assignment’s retrospective scoring period. The executable
selection pipeline uses zero April rows, but April had previously been viewed
during the broader project, so I do not claim that it is a pristine untouched
holdout.

## Uncertainty

Primary uncertainty artifact: paired **date-block** bootstrap on frozen April
predictions (`artifacts/current/date_block_bootstrap_*.`).

**Caveat:** intervals condition on the locked selected specification. Model
selection is **not** re-run inside each bootstrap sample. These are evaluation
intervals for the locked model, not fully selection-adjusted confidence
intervals.

Paired differences versus Elo and rank-component baselines are reported. The
correct claim is:

> The direct model won the declared pre-March validation process, but its
> incremental April value over Elo remains statistically unresolved.

## Calibration diagnostics

`artifacts/current/calibration_diagnostics.json` reports intercept \(\alpha\),
slope \(\gamma\), ECE, and probability range. Extreme-bin audit:
`artifacts/current/extreme_probability_audit.csv`.

Present these as diagnostics. Frozen-April ECE ≈ 0.113 and slope ≈ 1.44 do
**not** mean calibration has been solved. April is not used to recalibrate.

## Production claim

Prototype / research readiness only. Not a deployable sportsbook system.
