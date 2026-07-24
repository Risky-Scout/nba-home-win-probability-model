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
6. **March selection reuse.** March is used both to select each procedure's
   architecture and to report March metrics, so March metrics are in-sample for
   selection. The out-of-sample evidence is the nested rolling-origin audit.
7. **Feb–April is no longer a pristine holdout.** These months were revisited
   across multiple revisions, an interview review, and diagnostics that were
   chosen *after* seeing failures. The code excludes April from selection, but
   the broader project cannot claim perfect human blindness. A strong,
   defensible calibration claim needs a future untouched period or a new season.
8. **Directional, not certain, model choice.** The nested audit uses only ~11
   weekly blocks; 0 of 4,000 bootstrap replicates favored the blend, but this
   is strong directional evidence, not production-grade certainty from a large
   independent sample.
9. **Pricing layer omitted.** Fair odds contain no overround, liability, limits,
   or trader adjustment.
10. **Deliberately uncalibrated champion.** The nested calibration slope is
    mildly > 1 (slight underconfidence). A dedicated calibration-risk audit
    (`scripts/calibration_challenger.py`) tested identity-shrunk Platt and Beta
    recalibrators under both information policies against a strict multi-gate
    promotion rule; both over-corrected out-of-sample, so the raw-Elo champion is
    kept uncalibrated (`decision: keep_raw_elo`). The in-sample slope is a
    diagnostic only. With more data, a properly validated recalibration layer
    (or the dynamic hierarchical model below) could revisit this.

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
