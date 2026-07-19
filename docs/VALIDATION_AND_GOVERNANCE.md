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

## Calibration grid

For each architecture:

- Elo logit weight: 0.000 to 0.350 in 0.005 increments;
- temperature: 0.55 to 0.85 in 0.01 increments;
- shift: 0.10 to 0.40 in 0.01 increments.

That is 68,231 calibration candidates per architecture and 341,155 total
architecture-calibration combinations.

AUC is calculated once per blend weight because a positive affine logit
calibration does not alter ranking.

## Selection rule

A candidate is eligible only when its March values satisfy:

\[
LL < 0.509645,
\]

\[
Brier < 0.167618,
\]

\[
AUC > 0.831798,
\]

\[
Accuracy > 0.7782.
\]

Eligible candidates are ordered lexicographically:

1. lower log loss;
2. lower Brier;
3. higher AUC;
4. higher accuracy;
5. architecture name for deterministic final tie-breaking.

The selected point is not copied into scoring source code. It is written by
selection to `artifacts/selected_spec.json`, and the scorer loads that file.

## Operational versus frozen state

### `sequential_daily`

This is a rolling one-step-ahead backtest. Model coefficients remain frozen
during the target month, but results after each completed date update team
state for later dates.

It answers:

> What probability would the system have issued each day using all information
> available before that day?

### `frozen_snapshot`

Performance state is fixed at the last day before the target month. Schedule
dates continue updating so rest remains meaningful.

It answers:

> What probabilities would be issued for the entire month from one month-start
> performance snapshot?

Both are exported. Reviewers can choose the information policy that matches
their interpretation of the assignment.

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
