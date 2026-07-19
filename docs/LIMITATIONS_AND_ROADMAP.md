# Limitations and roadmap

## Current limitations

1. **One season.** Team strength, home advantage, and calibration are estimated
   from a single season.
2. **Team-level only.** No injuries, starters, expected minutes, trades, or
   player impact are available.
3. **No valid possession data.** Shooting attempts, free throws, offensive
   rebounds, and makes are missing.
4. **No travel geometry.** Rest is available, but flight distance, time-zone
   changes, altitude transitions, and road-trip structure are not.
5. **No market anchor.** Closing or opening prices are unavailable.
6. **March selection reuse.** March both selects and reports the chosen
   calibration.
7. **April was historically viewed.** The code excludes it from selection, but
   the broader project cannot claim perfect human blindness.
8. **AUC target miss.** The operational April model does not exceed the stated
   AUC target.
9. **Pricing layer omitted.** Fair odds contain no overround, liability, limits,
   or trader adjustment.

## Highest-value next data

1. expected lineups and injury status;
2. player minutes and player-impact ratings;
3. several prior seasons;
4. valid possession and Four-Factor inputs;
5. travel distance, time-zone change, and altitude;
6. market-implied probabilities for benchmarking and residual modeling.

## Model roadmap

With multiple seasons, move from fixed-state ratings to a dynamic hierarchical
model:

\[
s_{i,t} = \phi s_{i,t-1} + \epsilon_{i,t},
\]

with team-specific uncertainty, offseason regression, and player-availability
adjustments.

For production betting, add:

- probability calibration monitoring;
- drift detection;
- shadow deployment;
- price comparison to market;
- overround and risk controls;
- deterministic model/version identifiers on every quote;
- alerting for stale state or missing data.
