# Validation, selection, and governance

The deployed champion is **Elo-only**. The logistic-stacked blend is an
implemented-but-**rejected challenger**; the stacker and temperature floor
described below belong to that challenger, not to the deployed model.

## Chronological periods

| Period | Role |
|---|---|
| October-February | Base fit for the March one-step-ahead evaluation |
| March | Per-procedure architecture selection |
| October-March | Refit deployed champion coefficients (through March 31) |
| April | Final retrospective scoring period (state frozen at March 31) |

No random train/test split is used.

## Selection input contract

`scripts/select_model.py` executes:

```python
selection_games = games[games["game_date"] < "2026-04-01"].copy()
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
model discovery. Selection is **model-specific**: each procedure (Elo-only,
rank-only, blend) independently selects its own architecture by its own March
log loss (Brier tie-break). The deployed champion is the `conservative`
architecture selected on the Elo-only March log loss.

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

Each procedure minimizes its own March log loss (architecture-name tie-break).
The **champion** is then chosen by the nested rolling-origin audit: promote the
blend over Elo-only only if it beats Elo-only on **both** log loss and Brier
with the block-bootstrap upper CI below zero. That bar is not met, so the
decision under both information policies is **keep_elo_only**.

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
games. Pooled log loss / Brier (frozen-block ≈ daily-sequential): constant
0.688 / 0.247, **Elo-only 0.532 / 0.177**, rank-only 0.550 / 0.184, blend
0.548 / 0.183.

Block-bootstrap blend − Elo-only (frozen-block): ΔlogLoss +0.0166
(95% CI [+0.0101, +0.0232]), ΔBrier +0.0061 (95% CI [+0.0035, +0.0089]);
daily-sequential ΔlogLoss +0.0149 (95% CI [+0.0076, +0.0220]), ΔBrier +0.0055
(95% CI [+0.0025, +0.0086]). **0 of 4,000 week-block bootstrap replicates
favored the blend** on either metric. With only ~11 weekly blocks this is
strong directional evidence, not production-grade certainty. Outputs:
`artifacts/nested_frozen_block_summary.json`,
`artifacts/nested_daily_sequential_summary.json`, the `*_folds.csv`,
`*_predictions.csv`, and reliability figures
`figures/nested_frozen_block_reliability.png` /
`figures/nested_daily_sequential_reliability.png`.

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
