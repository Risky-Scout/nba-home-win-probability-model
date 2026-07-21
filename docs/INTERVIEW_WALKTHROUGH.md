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
2. `nba_wp/features.py`
3. `nba_wp/model.py`
4. `nba_wp/selection.py`
5. `artifacts/selection_proof.json`
6. `artifacts/feature_group_ablation.csv`
7. `outputs/april_predictions.csv`
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

Show `artifacts/data_audit.json`.

### 2:00-4:00 - Leakage control

Open `nba_wp/features.py` at `build_features`.

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
> their probabilities with a logistic stacker fitted on March component logits
> by penalized maximum likelihood — the blend weights are estimated, not
> searched.

Show `docs/METHODOLOGY.md` and `nba_wp/model.py::fit_logit_stacker`.

### 6:00-7:30 - Selection proof

Open `scripts/select_model.py` and `artifacts/selection_proof.json`.

> The selection script truncates the raw input before April and the selection
> function rejects any frame containing April. Five declared architectures are
> compared; for each, the stacker is fitted and March log loss is computed.
> The rule is simply: minimize March log loss. The selected JSON is generated
> and later loaded by the scorer; it is not duplicated as hidden constants.

### 7:30-9:00 - Feature proof

Show `artifacts/feature_group_ablation.csv` and
`figures/permutation_importance.png`.

> Record-only and richer box-score-style candidates were weaker. The selected
> blend reduced March log loss from 0.6806 for the training home prior to
> 0.4880. Permutation importance shows Bradley-Terry as the dominant ranking
> signal, Elo as meaningful, and trend as a smaller correction.

### 9:00-10:00 - Results and humility

> The blend coefficients are fitted by penalized maximum likelihood on March
> component logits — no grid search, no external targets. March log loss is
> 0.4880 and April is 0.4587, with April accuracy 82.3%. March is in-sample
> for the stacker, so I present April as the meaningful number. I also export
> a strict month-start snapshot because rolling daily and frozen-batch
> evaluations answer different questions.

Show `artifacts/final_metrics.json`.

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

Log odds are the additive scale of logistic models. A logistic regression on
the two component logits estimates the blend weights and intercept jointly by
maximum likelihood — sharpness and the home baseline fall out of the fit
instead of a grid.

### What do the stacker coefficients mean?

a = 0.5796 on the Elo logit, b = 0.9838 on the rank logit, intercept
c = 0.3154. Equivalently w = a/(a+b) = 0.371, implied temperature
1/(a+b) = 0.640, shift = 0.315. a + b > 1 means the fit sharpened the
components. I treat this cautiously because March is only 239 games and is
also the selection set.

### Is March performance unbiased?

No. March is the architecture and calibration selection set, so its champion
score is optimistic. The repository calls it a selection result, not an
untouched test result.

### Did April influence selection?

The executable selection path cannot load April. The broader project had
already viewed April, so I describe April as retrospective rather than claiming
perfect human blindness.

### Why can proper scores improve while AUC moves little?

AUC depends only on pairwise ranking. Calibration can materially improve log
loss and Brier without changing ranking, so the stacker mostly moves proper
scores rather than AUC.

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
