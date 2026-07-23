# Validation, selection, and governance

## Chronological periods

| Period | Role |
|---|---|
| October-February | Fit March component coefficients |
| March | Architecture and calibration selection |
| October-March | Refit locked component coefficients |
| April | Final retrospective scoring period |

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

- selection input maximum date;
- number of April rows loaded;
- selected architecture;
- selected calibration;
- predeclared ordering rule.

## Architecture grid

Five named architectures are declared before execution in
`configs/architecture_candidates.json`. They vary:

- Elo update rate and home adjustment;
- Bradley-Terry regularization;
- trend horizon and short window;
- component logistic regularization.

This is a small, inspectable structural grid rather than unrestricted automated
model discovery.

## Calibration fitting

For each architecture, the blend is fitted by penalized maximum likelihood
(logistic stacking) on March component logits:

\[
p = \sigma\bigl(a\operatorname{logit}(p_E) + b\operatorname{logit}(p_R) + c\bigr).
\]

The three coefficients \((a, b, c)\) are estimated by a single
`LogisticRegression(C=1.0)` fit — there is no calibration grid. This is
equivalent to the \((w, \tau, s)\) parameterization via \(w = a/(a+b)\),
\(\tau = 1/(a+b)\), \(s = c\).

Because the Elo and rank component logits are near-duplicates
(\(\rho\approx 0.97\)), the unconstrained fit learns \(a+b>1\) (temperature
\(\tau<1\)), which sharpens the blend. The model is deployed with a
**temperature floor** \(\tau \ge 1\): when the fit would sharpen, the weight
\(w\) is preserved, the coefficients are projected onto \(a+b=1\), and the
intercept is refit. Architecture selection still ranks on the unconstrained
March log loss; only the deployed coefficients carry the floor. Both sets are
recorded in `artifacts/selected_spec.json`.

## Selection rule

Minimize March log loss. Architecture name is used only for deterministic
tie-breaking of exact ties.

The selected coefficients are not copied into scoring source code. They are
written by selection to `artifacts/selected_spec.json`, and the scorer refits
the stacker the same way before applying it to April.

## Primary (frozen) versus sequential state

### `frozen_pre_april` — primary deliverable

Performance state is fixed at the last day before April, and the base-model
generator is trained through February to match the March stacker fit. No April
result updates any April performance state, so April outcomes cannot influence
April prices. This is the headline `outputs/april_predictions.csv` and is
pinned by `tests/test_primary_april_frozen.py`.

It answers:

> Using only October-March information, what price is issued for each April
> game?

### `sequential_daily_backtest` — diagnostic only

This is a rolling one-step-ahead backtest. Model coefficients remain frozen
during April, but results after each completed date update team state for later
dates. It is exported to
`outputs/april_predictions_sequential_backtest.csv`.

It answers:

> What probability would the system have issued each day using all information
> available before that day?

Both are exported, but the frozen version is the primary submission; the
sequential file is clearly labelled as a live-update simulation.

## Metric uncertainty and selection bias

March is used for selection, so its reported champion metric is optimistic as
an estimate of future performance. Dense calibration search has only three
degrees of freedom, but it still increases selection risk.

Accuracy changes in increments of \(1/239\) in March and \(1/96\) in April.
Tiny threshold differences should not be treated as economically meaningful.

A paired bootstrap against the constant baseline is included. Bootstrap
intervals against individual components are also exported. Bootstrap evidence
does not remove model-selection bias.

## April exposure

April had already been viewed during the broader project before this evidence
build. Therefore, it is described as a **retrospective final scoring period**,
not a pristine untouched scientific holdout.

The defensible claim is that the repository reconstructs an executable
selection path that cannot read April. It does not claim the candidate had
never seen the April figures in human development history.
