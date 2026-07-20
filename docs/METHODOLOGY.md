# Methodology (audit remediation)

## Target

\[
Y_g = 1\{\text{home points} > \text{away points}\},\quad
\text{deliverable } = P(Y_g=1).
\]

## Feature timing

For each date \(d\), features use only history strictly before \(d\). Same-date
games are batched. Postgame box scores update future state only.

## Strength features

- **Elo difference** with margin-of-victory updates.
- **Bradley-Terry logit** from regularized +1/−1 logistic on prior games.
- **Trend difference**: short window minus EWMA half-life form.

## Champion probability model

When selected by pre-March folds, the direct model is:

\[
P(Y=1)=\sigma(\beta_0+\beta_1\Delta\mathrm{Elo}+\beta_2\Delta\mathrm{BT}+\beta_3\Delta\mathrm{Trend}).
\]

This matches the algebraic span of a calibrated Elo/BT/trend blend while
avoiding a dense temperature/shift search.

## Selection

Expanding-window folds ending before March:

1. Train through December, validate January.
2. Train through January, validate February.

Primary metric: mean validation log loss. Brier is secondary. AUC/accuracy are
descriptive. External benchmark JSON values do not gate selection.

Search budget: Elo \(K\in\{10,20,30\}\), trend half-life \(\in\{20,45,90\}\),
logistic \(C\) on a 7-point log grid (63 direct candidates). Optional blend
challenger evaluated with Platt calibration.

## April scoring

Primary: freeze performance state at 2026-03-31.  
Sensitivity: sequential daily updates within April.

## Metrics

Log loss primary for pricing quality; Brier secondary; AUC/accuracy descriptive.
Fair odds \(1/p\) are zero-margin mathematical transforms, not offered prices.
