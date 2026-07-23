# Model card

## Intended use

Retrospective NBA home-win probability estimation for the supplied 2025-26
team-level technical-task dataset.

## Not intended for

- autonomous wagering;
- player medical or availability decisions;
- production sportsbook pricing without risk controls;
- seasons or leagues with unvalidated distributions.

## Target

Binary home win derived from final points.

## Inputs

Only leakage-safe team states derived from earlier dates:

- Elo strength;
- Bradley-Terry paired strength;
- recent point-margin trend.

Candidate record, margin, rest, turnover, rebound, and foul features are
generated for audit and ablation but are not in the champion.

## Training and selection

- March component fit: October-February.
- March architecture/calibration selection: rolling one-step-ahead March, on
  the unconstrained stacker log loss. The deployed stacker is then
  temperature-floored (T >= 1) so it never sharpens duplicate component signal.
- April primary policy: **frozen pre-April** — base models trained through
  February, no April result updates any April performance state. This is the
  headline `outputs/april_predictions.csv`.
- April sequential backtest: a live-update simulation exported separately as a
  diagnostic only.

## Primary metrics

1. log loss;
2. Brier score.

Secondary diagnostics:

3. ROC AUC;
4. accuracy at 0.5.

## Performance

Primary April holdout (frozen pre-April): log loss 0.4844, Brier 0.1558,
AUC 0.8628, accuracy 81.25% (N=96). See `artifacts/final_metrics.json` for the
full set (primary holdout, March selection surface, and sequential backtest).

## Interpretability

- full mathematical formulas in `docs/METHODOLOGY.md`;
- standardized and raw-unit coefficients in
  `artifacts/coefficient_table.csv`;
- permutation importance in `artifacts/permutation_importance.csv`;
- game-level component probabilities in both output files.

## Monitoring plan for production

- calibration slope/intercept by week;
- log loss and Brier versus closing market;
- probability distribution drift;
- team-state freshness;
- missing-player-data alerts;
- model and data version on every price;
- rollback to simple Elo baseline.
