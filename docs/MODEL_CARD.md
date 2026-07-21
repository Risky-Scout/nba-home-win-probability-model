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
- March architecture/calibration selection: rolling one-step-ahead March.
- April component refit: October-March.
- April state policies: operational sequential and frozen month-start.

## Primary metrics

1. log loss;
2. Brier score.

Secondary diagnostics:

3. ROC AUC;
4. accuracy at 0.5.

## Performance

See `artifacts/final_metrics.json` for the regenerated operational and
frozen-snapshot March/April metrics.

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
