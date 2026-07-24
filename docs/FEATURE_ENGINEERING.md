# Feature engineering and lineage

## Raw-field classification

| Raw field family | Timing | Treatment |
|---|---|---|
| `game_id`, `game_date`, teams | Known before game | Identification and schedule logic |
| pregame wins/losses | Known before game | Audited against derived records |
| points | Known after game | Update future Elo, margins, and trends only |
| turnovers | Known after game | Update future turnover tendencies only |
| fouls | Known after game | Update future foul tendencies only |
| rebounds | Known after game | Update future rebound tendencies only |

The task text says there are fourteen columns, but the listed schema contains
two game-level fields plus seven fields for each side, for sixteen actual
columns.

## Why direct points are forbidden

For the current row,

\[
Y_g = I(home\_points_g > away\_points_g).
\]

Using either current-game point total or point margin would reveal the target.
The feature engine therefore creates a row before it observes the row's box
score.

The same timing rule applies to turnovers, fouls, and rebounds. Even though
they do not deterministically define the winner, they are postgame information
and would make the probability unavailable at pricing time.

## Per-game and differential features

Raw historical totals are normalized by prior games. Examples:

\[
MarginAvg_{i,d}
=
\frac{\sum_{k<d} Margin_{ik}}{N_{i,d}}.
\]

\[
RecordLogitDiff_g
=
\operatorname{logit}(\tilde p_{home})
-
\operatorname{logit}(\tilde p_{away}),
\]

where

\[
\tilde p_i = \frac{W_i+4}{N_i+8}.
\]

Every candidate is expressed from the home team's perspective, so positive
values generally favor the home team.

## Rest

Rest is derived from game dates:

\[
RestAdvantage_g
=
RestDays_{home,g}
-
RestDays_{away,g},
\]

with each side capped at seven days. Back-to-back and schedule-density
indicators are also created.

Rest was not promoted to the champion because the richer linear challenger did
not improve March proper scores. Rejection is preserved in
`artifacts/feature_group_ablation.csv`.

## Turnovers and rebounds

For team \(i\),

\[
TurnoverAdvantage_i
=
E[TOV_{opp}-TOV_i],
\]

\[
ReboundAdvantage_i
=
E[REB_i-REB_{opp}].
\]

The matchup features are home minus away. They are generated for evidence and
challenger analysis but are not in the final model.

## Pace and Dean Oliver's Four Factors

The file lacks field-goal attempts, free-throw attempts, offensive rebounds,
shooting makes, and opponent shooting components. A valid possession estimate
and the traditional Four Factors cannot be reconstructed.

Defining possessions as points divided by a constant is rejected because it
makes offensive rating mathematically constant:

\[
ORtg
=
100\frac{PTS}{PTS/1.07}
=
107.
\]

The repository does not create pseudo-possession metrics.

## Selected features

The deployed champion is **Elo-only**, so its only input is `elo_diff`. The
`bt_logit` and `trend_diff` features feed the rejected challenger blend, not the
deployed model.

| Feature | Basketball interpretation | Model role |
|---|---|---|
| `elo_diff` | Updated outcome and margin strength | **Deployed champion (Elo-only)** |
| `bt_logit` | Regularized paired-comparison team strength | Rejected challenger (rank component) |
| `trend_diff` | Short-form change relative to long form | Rejected challenger (rank component) |

The complete generated dictionary is
`artifacts/feature_dictionary.csv`.
