# Interview walkthrough

## Screen-share setup

Before the call:

```bash
PROJECT="/absolute/path/to/nba-home-win-probability-model"
cd "$PROJECT"
source .venv/bin/activate
export NBA_DATA_PATH="$PROJECT/data/nba-win-probability-data.csv"
python validate_submission.py --root . --data "$NBA_DATA_PATH"
```

Keep these files open in editor tabs:

1. `SUMMARY.md`
2. `src/nba_wp/features.py`
3. `src/nba_wp/model.py`
4. `src/nba_wp/selection.py`
5. `artifacts/current/pre_march_selection_proof.json`
6. `artifacts/current/feature_group_ablation.csv`
7. `predictions/april_predictions.csv`
8. `docs/LIMITATIONS_AND_ROADMAP.md`

## Ten-minute presentation

### 0:00-1:00 - Objective and metric

> The task is to issue calibrated home-win probabilities, not just winners.
> I therefore treat log loss and Brier as primary, with AUC and accuracy as
> ranking and threshold diagnostics.

Show `SUMMARY.md`.

### 1:00-2:00 - Data audit

> The file contains 1,230 games, 30 teams, no missing values, no duplicate game
> IDs, and the supplied pregame records reconcile exactly. The task text says
> fourteen columns, but the listed schema has sixteen.

Show `artifacts/current/data_audit.json`.

### 2:00-4:00 - Leakage control

Open `src/nba_wp/features.py` at `build_features`.

> The key design decision is feature-before-update. For each date, I read team
> state, write every matchup feature, and only then update from that date's
> results. Current-game points, turnovers, rebounds, and fouls can update future
> state but cannot enter their own probability. Same-day games are batched.

Run:

```bash
python -m pytest tests/test_feature_timing.py -q
```

### 4:00-6:00 - Basketball model

> The first component is margin-of-victory Elo. The second is a regularized
> Bradley-Terry paired-comparison model plus recent point-margin trend. I blend
> their probabilities in log-odds space and apply a temperature and intercept
> calibration.

Show `docs/METHODOLOGY.md` and `src/nba_wp/model.py::blend_probabilities`.

### 6:00-7:30 - Selection proof

Open `python -m nba_wp.cli select` and `artifacts/current/pre_march_selection_proof.json`.

> The selection script truncates the raw input before April and the selection
> function rejects any frame containing April. Five declared architectures and
> a declared three-parameter calibration grid are searched. The selected JSON
> is generated and later loaded by the scorer; it is not duplicated as hidden
> constants.

### 7:30-9:00 - Feature proof

Show `artifacts/current/feature_group_ablation.csv` and
`figures/permutation_importance.png`.

> Record-only and richer box-score-style candidates were weaker. The selected
> blend reduced March log loss from 0.6806 for the training home prior to
> 0.4876. Permutation importance shows Bradley-Terry as the dominant ranking
> signal, Elo as meaningful, and trend as a smaller correction.

### 9:00-10:00 - Results and humility

> March exceeds all four rounded targets, although the AUC and accuracy margins
> are tiny. April beats log loss, Brier, and accuracy but misses AUC. I did not
> retune after that miss. I also export a strict month-start snapshot because
> rolling daily and frozen-batch evaluations answer different questions.

Show `artifacts/current/final_metrics.json`.

## Live-run routes

### Fast proof, under one minute

```bash
python validate_submission.py \
  --root . \
  --data "$NBA_DATA_PATH"
```

### Rebuild selected model and all prices

```bash
python run_submission.py \
  --root . \
  --data "$NBA_DATA_PATH" \
  --mode score
```

### Rerun March selection

```bash
python run_submission.py \
  --root . \
  --data "$NBA_DATA_PATH" \
  --mode select
```

### Full source-to-artifact reproduction

```bash
python run_submission.py \
  --root . \
  --data "$NBA_DATA_PATH" \
  --mode full
```

### Independent game-by-game check

```bash
python validate_submission.py \
  --root . \
  --data "$NBA_DATA_PATH" \
  --recompute
```

## Expected aggressive questions

### Why not XGBoost?

The dataset has only 1,230 games and the main signals are correlated team
strength estimates. Strongly regularized logistic components provide direct
probabilities, deterministic training, fast live execution, and coefficient
interpretability. A nonlinear challenger should be promoted only after
forward proper-score improvement, not because it is more complex.

### Why blend in log-odds space?

Log odds are the additive scale of logistic models. Weighting logits combines
evidence on a common scale. Temperature then controls sharpness and shift
controls the home baseline.

### Why is the temperature 0.59?

The March grid found the uncalibrated components underconfident for that
period. A temperature below one sharpened them. I treat this cautiously because
March is only 239 games and is also the selection set.

### Is March performance unbiased?

No. March is the architecture and calibration selection set, so its champion
score is optimistic. The repository calls it a selection result, not an
untouched test result.

### Did April influence selection?

The executable selection path cannot load April. The broader project had
already viewed April, so I describe April as retrospective rather than claiming
perfect human blindness.

### Why does AUC miss in April while proper scores improve?

AUC depends only on pairwise ranking. Calibration can materially improve log
loss and Brier without changing ranking. The model's probabilities were useful
but some April matchup ordering differed from the AUC benchmark.

### Are earlier April results used?

In the operational one-step-ahead output, yes: a completed earlier date may
update state for a later date. No same-day or future result is visible. The
repository also exports a strict March 31 snapshot where no April result
updates any April performance state.

### What would you productionize next?

Add multi-season hierarchical state, player availability, possession-quality
inputs, travel, drift monitoring, versioned prices, market comparison,
overround, and trader/risk controls.

### Why not calculate pace?

The required field-goal attempts, free throws, offensive rebounds, and makes
are missing. Inventing possessions from points would create a constant
offensive rating and is mathematically invalid.

## Failure recovery during screen share

If the virtual environment is not active:

```bash
source .venv/bin/activate
```

If a package is missing:

```bash
python -m pip install -r requirements-dev.txt
```

If the data path is wrong:

```bash
find "$PROJECT" \
  -name "nba-win-probability-data.csv" \
  -print
```

If you need a clean score rebuild:

```bash
rm -rf outputs figures
mkdir -p outputs figures
python run_submission.py --data "$NBA_DATA_PATH" --mode score
```
