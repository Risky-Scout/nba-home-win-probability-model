# Validation, selection, and governance

The deployed champion is **Elo-only**. The logistic-stacked blend is an
implemented-but-**rejected challenger**; the stacker and temperature floor
described below belong to that challenger, not to the deployed model.

## Chronological periods

| Period | Role |
|---|---|
| October-February | Base fit for the March one-step-ahead cross-check |
| Pre-holdout weekly origins | Aggregate frozen-policy OOS surface used to select the deployed architecture |
| March | In-sample diagnostic report + rank/blend challenger references |
| October-March | Refit deployed champion coefficients (through March 31) |
| April | Final retrospective scoring period (state frozen at March 31) |

No random train/test split is used. All period boundaries are **derived from the
data** (`nba_wp/periods.py`: last calendar month = holdout, prior month =
selection), not hard-coded, so a different season re-slices automatically.

## Selection input contract

`scripts/select_model.py` executes:

```python
periods = derive_periods(games)                 # holdout = last month (April)
selection_games = games[games["game_date"] < periods.holdout_start].copy()
```

`nba_wp.selection.run_selection` then raises an exception if its maximum date
is April 1 or later.

The generated proof is:

```text
artifacts/selection_proof.json
```

It records:

- selection input maximum date (`selection_input_max_date = 2026-03-31`);
- number of April rows loaded (0);
- selected architecture per procedure;
- selected model family (`elo_only`);
- predeclared ordering rule.

`tests/test_information_policies.py` and
`tests/test_champion_promotion.py` pin that April cannot enter selection and
that the deployed model equals the champion decision.

## Architecture grid

Five named architectures are declared before execution in
`configs/architecture_candidates.json`. They vary:

- Elo update rate and home adjustment;
- Bradley-Terry regularization;
- trend horizon and short window;
- component logistic regularization.

This is a small, inspectable structural grid rather than unrestricted automated
model discovery. The **deployed Elo architecture** is chosen by aggregate
**frozen-policy rolling out-of-sample log loss** over the pre-holdout weekly
origins, with a **one-standard-error stability rule**: among all architectures
whose mean OOS log loss lies within one standard error of the best, the simplest
(lowest `K`) is deployed. For this data every architecture sits within one SE and
`conservative` wins outright (lowest mean OOS log loss *and* lowest `K`); a single
March one-step split would pick the same architecture. The full ranking, the
one-SE band, and the March cross-check are recorded under `architecture_selection`
in `artifacts/selected_spec.json` (`tests/test_architecture_selection.py`). The
per-procedure March comparison is retained as a diagnostic and for the rank/blend
challenger references.

Two further constants are also **data-driven**, profiled on the same frozen OOS
surface and recorded in `selected_spec.json`: the margin-of-victory offset
(`mov_offset_selection`, empirically confirmed at 2.2 — kept because it is within
one SE of the grid best) and an early-season provisional-K warmup
(`cold_start_selection`, profiled and kept **off** because it does not beat
no-warmup). Neither changes the deployed numbers; both replace an unexamined
constant with an auditable choice.

## Deployed champion — Elo-only

The deployed price is a single logistic map on the standardized Elo rating
differential,

\[
p = \sigma\bigl(c + w\,z(\text{elo\_diff})\bigr),
\]

fitted (L2, \(C=10\)) on all games through March 31. There is **no stacker and
no temperature floor** in the deployed model. The five deployed values
(\(w, c, \mu, s\), and the raw-unit weight) are stored under `elo_model` in
`artifacts/selected_spec.json`; there is no top-level `calibration` key.

## Rejected challenger — blend calibration

For the challenger only, the blend is fitted by penalized maximum likelihood
(logistic stacking) on March component logits:

\[
p = \sigma\bigl(a\operatorname{logit}(p_E) + b\operatorname{logit}(p_R) + c\bigr),
\]

equivalent to the \((w, \tau, s)\) parameterization via \(w = a/(a+b)\),
\(\tau = 1/(a+b)\), \(s = c\). Because the Elo and rank component logits are
near-duplicates (\(\rho\approx 0.97\)), the unconstrained fit learns \(a+b>1\)
(temperature \(\tau<1\)), which sharpens the blend. The challenger applies a
**temperature floor** \(\tau \ge 1\) — a genuine convex logit blend
(\(0\le a,b\le 1\), \(a+b=1\), \(\tau\ge 1\), pinned by
`tests/test_stacker_temperature_floor.py::test_stacker_weights_are_convex_when_stacker_is_used`).
All challenger coefficients live under the `challenger` block of
`artifacts/selected_spec.json` (`status: "rejected"`).

## Selection and promotion rule

The deployed architecture is fixed by the one-standard-error stability rule
above. The **champion model family** is then decided by the nested rolling-origin
audit under a strict promotion bar: a challenger is promoted over Elo-only only if
it beats Elo-only on **both** log loss and Brier with the block-bootstrap upper CI
below zero, under both information policies. Three challengers are evaluated and
**all three are rejected**:

- **Convex blend** (Elo + Bradley-Terry/trend): worse on both proper scores →
  `keep_elo_only`.
- **Calibrated Elo** (cross-fitted, identity-shrunk `α + β·logit(p_elo)`):
  over-corrects out-of-sample (β ≈ 1.79 vs raw 1.32) → `keep_raw_elo`.
- **Schedule Elo** (Elo + rest + back-to-back, strongly regularized): slightly
  worse OOS on both scores → `keep_raw_elo`.

The single statistically strong claim is exactly this rejection of the more
complex challengers; the simplest model is deployed.

The deployed coefficients are not copied into scoring source code. They are
written by selection to `artifacts/selected_spec.json`, and the scorer refits
the deployed Elo-only model the same way before applying it to April.

## Nested rolling-origin audit — honest out-of-sample evidence

`scripts/nested_validation.py` runs two clearly separated information policies,
never mixed into one metric.

### Frozen-block policy

For each outer weekly origin \(O\), the performance state is frozen at \(O-1\)
via `build_features(freeze_date=O)`, and the block \([O, O+7)\) is scored with
models fit strictly before \(O\). Mutating any outcome inside the block cannot
change any price in the block
(`frozen_block_leakage_guarantee_verified = true`).

### Daily-sequential policy

The price for date \(t\) uses results strictly before \(t\) (base models refit
through \(t-1\)); a live one-step-ahead simulation.

For every outer fold, each procedure independently selects its own architecture
by its own inner out-of-fold score; the convex stacker is trained on inner
out-of-fold component predictions. 11 weekly outer folds, 501 out-of-sample
games. Pooled log loss / Brier (frozen-block): constant
0.688 / 0.247, **Elo-only 0.529 / 0.176**, rank-only 0.547 / 0.183, blend
0.548 / 0.183.

Block-bootstrap blend − Elo-only (frozen-block): ΔlogLoss +0.0186
(95% CI [+0.0124, +0.0239]), ΔBrier +0.0066 (95% CI [+0.0042, +0.0087]);
daily-sequential ΔlogLoss +0.0166 (95% CI [+0.0100, +0.0234]), ΔBrier +0.0062
(95% CI [+0.0037, +0.0087]). **0 of 4,000 week-block bootstrap replicates
favored the blend** on either metric. With only ~11 weekly blocks this is
strong directional evidence, not production-grade certainty. Outputs:
`artifacts/nested_frozen_block_summary.json`,
`artifacts/nested_daily_sequential_summary.json`, the `*_folds.csv`,
`*_predictions.csv`, and reliability figures
`figures/nested_frozen_block_reliability.png` /
`figures/nested_daily_sequential_reliability.png`.

## Calibration-risk investigation

`scripts/calibration_challenger.py` is a dedicated model-risk audit of whether a
post-hoc probability calibrator legitimately improves on the deployed raw-Elo
champion. It scores three complete procedures on **identical** outer-fold rows —
**raw Elo** (deployed, no calibration), an **identity-shrunk Platt-on-logit**
recalibrator, and an **identity-shrunk Beta** recalibrator — under both the
frozen-block and daily-sequential policies, reusing the same policy-matched
nested engine. Each recalibrator is fit only on the Elo architecture's
policy-matched inner out-of-fold rows strictly earlier than the outer origin;
its L2 shrinkage strength is selected only by chronological inner validation; and
any degeneracy falls back to the numerical identity, so no outer-fold outcome is
seen while fitting or tuning.

The promotion bar is strict: a recalibrator replaces raw Elo **only if**, under
**both** policies, **all** gates hold — mean Δ log loss < 0, mean Δ Brier < 0,
week-block bootstrap upper 95% CI < 0 for both proper scores, every
leave-one-outer-fold-out aggregate Δ < 0 for both, no single outer fold
contributing ≥ 50% of the total improvement, and no material tail degradation
(challenger calibration gap in p ≥ 0.90 or p ≤ 0.10 not worse than raw by more
than 0.02). Neither recalibrator clears every gate: identity-shrunk calibration
over-corrects out-of-sample, so the decision is `keep_raw_elo` and the champion
is unchanged. The pooled in-sample slope (~1.3) is flagged as diagnostic-only
and is never applied in deployment. Evidence:
`artifacts/calibration_challenger_decision.json`,
`artifacts/calibration_challenger_frozen_predictions.csv`,
`artifacts/calibration_challenger_daily_predictions.csv`, and
`tests/test_calibration_risk.py`.

## Primary (frozen) versus sequential state

### `frozen` — primary deliverable

Performance state is fixed at March 31, and the deployed Elo-only coefficients
are fit through March 31. No April result updates any April performance state,
so April outcomes cannot influence April prices. This is the headline
`outputs/april_predictions.csv` and is pinned by
`tests/test_primary_april_frozen.py`.

It answers:

> Using only October-March information, what price is issued for each April
> game?

### `sequential_daily_backtest` — diagnostic only

This is a rolling one-step-ahead backtest. Results after each completed date
update team state for later dates. It is exported to
`outputs/april_predictions_sequential_backtest.csv`.

It answers:

> What probability would the system have issued each day using all information
> available before that day?

Both are exported, but the frozen version is the primary submission; the
sequential file is clearly labelled as a live-update simulation.

## Metric uncertainty and selection bias

March is used for selection, so its reported March metric is optimistic as an
estimate of future performance. The decisive generalization evidence is the
nested audit, which selects each procedure's architecture inside every fold.

Accuracy changes in increments of \(1/239\) in March and \(1/96\) in April.
Tiny threshold differences should not be treated as economically meaningful.

## April exposure

April had already been viewed during the broader project before this evidence
build. Therefore, it is described as a **retrospective final scoring period**,
not a pristine untouched scientific holdout.

The defensible claim is that the repository reconstructs an executable
selection path that cannot read April. It does not claim the candidate had
never seen the April figures in human development history.
