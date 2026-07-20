# Validation and governance (audit remediation)

## Chronological periods

| Period | Role |
|---|---|
| Oct–Dec / Jan; Oct–Jan / Feb | Pre-March selection folds |
| March | Locked test after specification freeze |
| April frozen at March 31 | Primary assignment scoring |
| April sequential | Operational sensitivity |

## Selection input contract

`scripts/select_model.py` truncates to `game_date < 2026-03-01`.
`run_selection` / `assert_pre_march_selection_frame` raise if any March-or-later
row is present.

Proof: `artifacts/pre_march_selection_proof.json`.

## Search budget

Declared in `configs/architecture_candidates.json`.

| Hyperparameter | Candidate values | Why it exists | Why these values |
|---|---|---|---|
| Elo \(K\) | 10, 20, 30 | Controls how quickly sequential Elo adapts after each game | Slow / moderate / fast adaptation; enough to detect over- or under-updating without a dense grid |
| Trend half-life (days) | 20, 45, 90 | Exponential memory of recent margin/form | Short / medium / long form windows aligned to roughly monthly and quarterly basketball form |
| Logistic \(C\) | 0.01, 0.03, 0.1, 0.3, 1, 3, 10 | L2 inverse-regularization for the direct three-feature logistic | Log-spaced strengths covering strong shrinkage through near-MLE |

That is \(3 \times 3 \times 7 = 63\) direct-logistic specifications.

An optional **Platt-calibrated blend challenger** may be evaluated per feature
architecture. It is retained for comparison with the original submission and
is promoted only when it wins pre-March mean validation log loss.

### Why the dense temperature/shift grid was removed

For \(T > 0\),

\[
p=\sigma(z/T+b)
\]

is strictly monotonic in the latent score \(z\). Changing temperature or shift
therefore:

- cannot change AUC / ranking;
- only moves calibration and the 0.50 accuracy threshold;
- does not justify treating hundreds of thousands of \((T,b)\) pairs as distinct
  ranking models against a few hundred games.

Calibration is estimated with logistic (Platt) calibration,

\[
\operatorname{logit}(p_{\mathrm{cal}})=\alpha+\gamma\operatorname{logit}(p_{\mathrm{raw}}),
\]

rather than a brute-force temperature-shift search.

## Selection objective

- **Primary:** mean validation log loss across January and February folds.
- **Secondary:** Brier score (tie-break).
- **Descriptive only:** AUC and accuracy.
- **Not used:** external benchmark floats as a hard gate.

## Benchmarks

`configs/benchmarks.json` values are retrospective references only.
Provenance: `docs/BENCHMARK_PROVENANCE.md`. They do not determine selection.

## April exposure

April is the assignment’s retrospective scoring period. The executable
selection pipeline uses zero April rows, but April had previously been viewed
during the broader project, so I do not claim that it is a pristine untouched
holdout.

For future evidence, freeze the model under a Git tag and score genuinely
unseen future games without modification.

## Uncertainty

Primary uncertainty artifact: paired **date-block** bootstrap on the frozen
April predictions (`artifacts/date_block_bootstrap_*.`).

- Games are grouped by date; dates are sampled with replacement.
- Metrics and calibration intercept/slope are recomputed on each replicate.
- Paired differences versus Elo and rank-component probabilities are reported.
- Intervals **condition on the locked selected specification**; model selection
  is not re-run inside each bootstrap sample (stated explicitly when time
  precludes nested selection).

## Calibration diagnostics

`artifacts/calibration_diagnostics.json` reports intercept \(\alpha\), slope
\(\gamma\), ECE, and probability range. Extreme-bin audit:
`artifacts/extreme_probability_audit.csv`.

April is not used to recalibrate.

## Production claim

Prototype / research readiness only. Not a deployable sportsbook system.
