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

The deployed champion is **Elo-only**. Its single input is a leakage-safe team
state derived from earlier dates:

- margin-of-victory Elo rating differential (`elo_diff`).

Bradley-Terry paired strength and recent point-margin trend, along with
candidate record, margin, rest, turnover, rebound, and foul features, are
generated for audit, ablation, and the rejected challenger blend — they are
**not** in the champion.

## Training and selection

- Deployed champion: **Elo-only**. A logistic map on the standardized Elo
  differential, \(p = \sigma(c + w\,z(\text{elo\_diff}))\), fitted on all games
  through March 31 with the April performance state frozen at March 31. The
  champion has **no stacker and no temperature floor**.
- Selection is April-blind: the deployed Elo architecture is chosen by
  **aggregate frozen-policy rolling out-of-sample log loss** across pre-holdout
  weekly origins, with a **one-standard-error stability rule** that prefers the
  simplest/lowest-K architecture inside the noise band. All five candidates fall
  within one SE; `conservative` wins on both the lowest mean OOS log loss and the
  lowest K, and a single March split would have picked the same architecture
  (`architecture_selection` in `selected_spec.json`). The MOV offset (2.2) and
  the off cold-start warmup are likewise profiled on that surface
  (`mov_offset_selection`, `cold_start_selection`). Season boundaries are derived
  from the data (`nba_wp/periods.py`); selection loads 0 April rows
  (`selection_input_max_date = 2026-03-31`).
- Champion = Elo-only because the nested rolling-origin audit rejects the
  blend (worse out-of-sample log loss and Brier). The blend is retained only as
  a clearly-labelled **rejected challenger**; the stacker/temperature floor
  belong to it, not to the champion. Two further challengers — a cross-fitted
  calibrated Elo and an Elo + rest/back-to-back schedule Elo — were added to the
  audit and are **also rejected** under both policies (`keep_raw_elo`).
- April primary policy: **frozen** — the deployed Elo-only model, April
  performance state frozen at March 31, no April result updating any April
  price. This is the headline `outputs/april_predictions.csv`.
- April sequential backtest: a live-update simulation exported separately as a
  diagnostic only.

## Primary metrics

1. log loss;
2. Brier score.

Secondary diagnostics:

3. ROC AUC;
4. accuracy at 0.5.

## Performance

Primary April holdout (Elo-only champion, frozen): log loss 0.464369,
Brier 0.149770, AUC 0.866847, accuracy 78.1250% (N=96). Mean forecast 0.549
versus observed home rate 0.594 — the champion is mildly *under*-forecasting,
not overconfident. See `artifacts/final_metrics.json` for the full set (primary
holdout, March selection surface, and sequential backtest). For reference, the
rejected blend on the same frozen April window scores log loss 0.468725 /
Brier 0.150465 — worse on both proper scores.

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
- rollback to the constant home-rate baseline.
