# Interview Q&A cheatsheet

Aligned to the eight-dimension evaluation matrix in
`docs/EVALUATION_MATRIX_PREP.md`. Short answers first. Offer the artifact only
if they ask for proof.

**Formula to never get wrong:**

```text
z = 0.19 * logit(p_elo) + 0.81 * logit(p_rank)
p = sigmoid(z / 0.59 + 0.33)
```

Temperature divides the **full** blended logit, not only the rank term.

## Modeling choices

### Why probabilities instead of winners?

Betting products need prices. Log loss and Brier penalize bad probabilities;
accuracy does not.

### Why Elo + Bradley-Terry instead of one model?

They capture related but different strength views: sequential margin-adjusted
ratings versus a global regularized paired-comparison fit. Trend adds recent
form relative to longer form. The blend lets calibration decide the weight.

### Why blend in log-odds?

Logits are the natural additive evidence scale for logistic models. Weighting
probabilities on [0,1] would distort extremes. Temperature then scales the
entire blended logit uniformly.

### Why temperature 0.59 and shift 0.33?

March grid search. Temperature < 1 sharpened underconfident components by
dividing the full blended logit; shift absorbed residual home baseline. Treat
cautiously: March is only 239 games and is also the selection set.

### Why not XGBoost / neural nets?

Small N, collinear team-strength features, need for deterministic training and
interpretable coefficients. Promote a nonlinear model only after a forward
proper-score improvement.

### Why L2 logistic calibration on components?

It maps a strength feature to a probability with shrinkage, intercept, and
stable optimization. It is also fast enough for live demo and production-style
recompute.

## Leakage and validation

### How do you prevent same-game leakage?

Feature-before-update, date-batched. Tests in `tests/test_feature_timing.py`.

### Did April influence selection?

Executable path: no. `select_model.py` truncates before April and
`run_selection` rejects April dates. Human project history: April was viewed,
so call April retrospective, not pristine.

### Is March unbiased?

No. March selected the champion, so its metrics are optimistic.

### Random split?

Never. This is a time-series pricing problem. Random splits leak future form.

### Operational vs frozen snapshot?

Operational updates state after each completed date. Frozen holds performance
state at month start. Both exported; do not mix silently.

## Features

### Why ignore current points / turnovers / fouls / rebounds?

They are observed after the game. Points determine the label. The others are
still unavailable at pricing time.

### Why no pace / Four Factors?

Missing FGA/FTA/OREB/makes. Fake possessions from points imply constant
offensive rating.

### Why was rest rejected?

Built and tested in the rich linear challenger; it did not improve March
proper scores enough to displace the Elo + BT/trend blend.

### What does trend mean?

Short recent point-margin mean minus longer EWMA margin. Positive means the
team is outrunning its longer form.

### Beta(4,4) on records?

Eight games of neutral pseudo-counts so early-season 0-0 / 1-0 records do not
produce infinite logits.

## Results

### Did you hit the targets?

March: all four rounded targets, with tiny AUC/accuracy margins. April: beat
log loss, Brier, accuracy; **missed AUC** (0.850 vs 0.868). No retune after
the miss.

### Why miss AUC but improve proper scores?

Different objectives. Calibration can help probability scores without fixing
all pairwise rankings.

### Is 83% April accuracy meaningful?

96 games; each game is about 1.04 percentage points. Prefer log loss/Brier for
economic interpretation.

### Bootstrap vs Elo includes zero. Why keep the blend?

Observed March improvement exists; interval is wide. Blend also improved vs the
rank component. Selection bias remains. I keep the blend because it won the
declared selection rule on proper scores, not because bootstrap "proved" it.

## Sportsbook / role fit

### How would this become a price?

Start from `home_win_probability`, convert to fair decimal odds, then apply
overround, risk limits, liability, and trader overrides. Those layers are out
of scope for this task but are the next production step.

### What would you monitor in production?

Calibration drift, log loss vs market, stale team state, missing injury feeds,
sudden rating jumps after lineup news, and version identifiers on every quote.

### How do you work with traders?

Expose component probabilities and feature contributions, not only a final
number. Traders need to know whether the price is team-strength driven, form
driven, or calibration driven.

### Computational efficiency?

Feature state is sequential and cheap; scoring is a handful of logistic
evaluations per game. Suitable for batch and near-real-time pregame use. Not
an in-play possession model.

## Failure recovery

```bash
source .venv/bin/activate
export NBA_DATA_PATH="$PWD/data/nba-win-probability-data.csv"
python validate_submission.py --root . --data "$NBA_DATA_PATH"
```

If needed:

```bash
python -m pip install -r requirements-dev.txt
python run_submission.py --root . --data "$NBA_DATA_PATH" --mode score
```
