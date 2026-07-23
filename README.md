# NBA Home-Win Probability Model

A leakage-audited, reproducible implementation of the **Elo +
Bradley-Terry/recent-trend logit blend**. This repository is designed for a
technical interview: every material claim maps to executable code, a generated
artifact, or an automated test.

## Start here

| Review time | Route |
|---|---|
| 2 minutes | Read `SUMMARY.md`, then open `artifacts/final_metrics.json`. |
| 10 minutes | Follow `docs/REVIEWER_GUIDE.md`. |
| Reproduce | Run `make reproduce DATA=/absolute/path/to/nba-win-probability-data.csv`. |
| Audit | Run `make verify DATA=/absolute/path/to/nba-win-probability-data.csv`. |
| Trace claims | Open `docs/CLAIM_TO_ARTIFACT_MAP.md`. |

## Objective

Estimate the probability that the listed home team wins. The assignment
contains 1,230 games and asks that October-March information be used to produce
April probabilities.

The raw box-score fields (`points`, `turnovers`, `fouls`, and `rebounds`) are
same-game outcomes. They are **never used directly to predict their own row**.
They update team state only after every game on that date has had its feature
row created.

## Model

The champion has two transparent components:

1. **Margin-of-victory Elo** - sequential team ratings with a home adjustment.
2. **Bradley-Terry + recent trend** - regularized paired-comparison strength
   plus the difference between short and long point-margin form.

Each component is calibrated by an L2-logistic model. Their probabilities are
blended by a **fitted logistic stacker** (penalized maximum likelihood):

```text
p_home = sigmoid(a * logit(p_elo) + b * logit(p_rank) + c)
```

The coefficients (a, b, c) are estimated by logistic regression on March
component logits — no grid search — and then **deployed with a temperature
floor T >= 1**. An unconstrained stacker on two highly correlated component
logits (rho approx 0.97) learns a + b > 1, i.e. temperature T = 1/(a+b) < 1,
which sharpens duplicate information and produced extreme (99%+) prices. The
deploy blend keeps the same Elo/rank weight but sets a + b = 1 (convex logit
blend, no sharpening) and refits only the intercept.

The selected March-only specification is stored in
`artifacts/selected_spec.json`:

```text
architecture            hfa_75
Elo K                   10
Elo home adjustment     75 rating points
Elo MOV multiplier      winner Elo - loser Elo (FiveThirtyEight log form)
Bradley-Terry C         0.15
trend half-life         45 days
short trend window      10 games
deploy stacker a        0.3593   (elo logit weight)
deploy stacker b        0.6407   (rank logit weight, a + b = 1, T = 1)
deploy stacker c        0.3254   (intercept)
unconstrained a, b, c   0.5696, 1.0159, 0.3132  (stored for audit only)
```

## Evaluation protocol and information policy

### Selection

- Base model coefficients for March are fitted through February.
- March is an **operational one-step-ahead validation**: after all games on a
  date are priced, that date's results may update state for later dates.
- `scripts/select_model.py` truncates the input at March 31 before feature
  construction. It cannot read an April row.
- Five declared architectures are evaluated. For each, the blend is fitted by
  penalized maximum likelihood (logistic stacking) on March component logits.
- The selection rule is: **minimize March log loss**. Nothing else.

### April

- Architecture and calibration are frozen from March selection.
- **Primary deliverable** `outputs/april_predictions.csv` is the
  **frozen pre-April** file: no April result updates any April performance
  state, and the base-model generator matches the March stacker fit (trained
  through February) so the calibrator inputs mean the same thing at fit and
  deploy time.
- `outputs/april_predictions_sequential_backtest.csv` is an optional
  live-update simulation only (earlier April dates may update team state for
  later April dates). It is clearly labelled and is not the headline result.

## Results

### Primary April holdout (frozen pre-April)

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| **April frozen (primary)** | 0.484432 | 0.155823 | 0.862798 | 81.2500% |

Frozen April price distribution: maximum `home_win_probability` approx 0.948;
4 of 96 games priced at or above 0.90; none at or above 0.95.

### March (selection / stacker-training period — in-sample)

| Stacker | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| Deploy (T >= 1) | 0.508373 | 0.164854 | 0.830044 | 76.9874% |
| Unconstrained (selection surface only) | 0.488026 | 0.157018 | 0.830044 | 78.6611% |

March is simultaneously the architecture-selection set and the stacker-training
set, so March numbers are **in-sample for the blend** and are not an unbiased
out-of-sample score. The architecture is chosen on the unconstrained March log
loss; the deployed model then applies the T >= 1 floor.

### Optional April sequential backtest (live-update simulation)

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| April sequential | 0.474545 | 0.152112 | 0.852901 | 82.2917% |

The sequential backtest scores slightly better but uses within-April state
updates; the frozen version is the honest answer to "use October-March to price
April" and is therefore the primary submission.

### Rolling-origin out-of-sample calibration (robustness check)

The frozen holdout is a single origin. To stress calibration across many
origins, `scripts/rolling_oof_calibration.py` runs an expanding-window,
one-step-ahead backtest: for each weekly fold from 2026-03-01, the base models
and the deploy stacker (`T >= 1`) are fit strictly *before* the fold, then the
fold is scored. Pooling all 7 folds (335 out-of-sample games):

| Pooled OOS | Log loss | Brier | AUC | Accuracy | ECE |
|---|---:|---:|---:|---:|---:|
| Rolling one-step-ahead | 0.505 | 0.166 | 0.835 | 78.2% | 0.118 |

Honest read: under this stricter protocol the model is **mildly underconfident
in the mid-range** (e.g. the ~0.51 decile realizes ~0.77; the ~0.60 decile
realizes ~0.82), giving an expected calibration error of about 0.12 — larger
than the single frozen-April window suggests. This is the expected cost of the
conservative `T >= 1` temperature floor plus small per-fold training windows,
and underconfidence is the safer failure mode than the overconfidence we
removed. See `figures/rolling_oof_calibration.png` and
`artifacts/rolling_oof_calibration.csv`.

## What constitutes proof

| Claim | Executable evidence |
|---|---|
| Data contain 1,230 valid games | `scripts/data_audit.py`, `artifacts/data_audit.json` |
| Pregame records reconcile | `artifacts/data_audit.json` |
| Same-game box scores are excluded | `nba_wp/features.py`, `tests/test_feature_timing.py` |
| Same-date games are batched | `tests/test_feature_timing.py::test_same_day_games_are_batched` |
| April was absent from selection | `scripts/select_model.py`, `artifacts/selection_proof.json` |
| The stacker coefficients were fitted, not typed into scoring code | `artifacts/march_architecture_results.csv`, `artifacts/selected_spec.json` |
| Elo MOV uses winner - loser rating diff (upsets move ratings more) | `nba_wp/features.py`, `tests/test_elo_mov_winner_diff.py` |
| Deploy stacker never sharpens (temperature T >= 1) | `nba_wp/model.py`, `tests/test_stacker_temperature_floor.py` |
| Primary April is frozen (April outcomes cannot change April prices) | `nba_wp/reporting.py`, `tests/test_primary_april_frozen.py` |
| Leakage battery: future/box-score/row-order do not move features; past outcomes provably do (positive controls) | `tests/test_leakage_mutations.py` |
| Calibration holds out-of-sample under rolling one-step-ahead folds | `scripts/rolling_oof_calibration.py`, `artifacts/rolling_oof_calibration.csv`, `artifacts/rolling_oof_metrics.json`, `figures/rolling_oof_calibration.png` |
| Metrics recompute from individual prices | `validate_submission.py`, `tests/test_artifacts.py` |
| Saved probabilities reproduce | `python validate_submission.py --recompute ...` |
| Feature contribution is measurable | `artifacts/feature_group_ablation.csv`, `artifacts/permutation_importance.csv` |

## Human-readable results export

For a spreadsheet-friendly view of the prices, run:

```bash
python -m scripts.export_results_spreadsheet
```

This writes an `.xlsx` (and CSV fallback) with the frozen April predictions
(primary), March predictions, the April sequential backtest, and a summary
sheet. A pre-rendered `outputs/april_predictions_readable.csv` is also committed.
Every value reconciles to `artifacts/selected_spec.json` and
`outputs/april_predictions_frozen_snapshot.csv`, which is identical to the
primary `outputs/april_predictions.csv`.

## Run locally

```bash
git clone https://github.com/Risky-Scout/nba-home-win-probability-model.git
cd nba-home-win-probability-model

bash scripts/bootstrap_macos.sh
source .venv/bin/activate

make reproduce DATA="/absolute/path/to/nba-win-probability-data.csv"
```

The full workflow performs:

1. data audit;
2. March-only architecture and calibration selection;
3. March and April scoring;
4. ablation, coefficients, permutation importance, calibration bins, figures;
5. manifest generation;
6. independent metric verification.

For a faster live demonstration:

```bash
python validate_submission.py   --root .   --data "/absolute/path/to/nba-win-probability-data.csv"
```

For a full game-by-game rebuild:

```bash
python validate_submission.py   --root .   --data "/absolute/path/to/nba-win-probability-data.csv"   --recompute
```

## Repository map

```text
.
├── nba_wp/
│   ├── data.py               # schema validation and record reconciliation
│   ├── features.py           # sequential/frozen feature state engine
│   ├── model.py              # base models, logistic stacker, metrics
│   ├── selection.py          # March-only model selection
│   └── reporting.py          # prices, evidence artifacts, figures
├── configs/                  # declared candidates, selection policy
├── scripts/                  # audit, selection, scoring, manifest entry points
├── tests/                    # leakage, artifact, and data-contract tests
├── artifacts/                # machine-readable proof
├── outputs/                  # game-level March and April probabilities
├── figures/                  # reviewer-facing diagnostics
├── docs/                     # methodology, governance, walkthrough, limitations
├── run_submission.py         # one-command workflow
└── validate_submission.py    # independent quality gate
```

## Scope

This is a one-season, team-level technical-task model, not a complete
sportsbook production price. The data omit player availability, lineups,
minutes, possession inputs, travel distance, market prices, and multiple
seasons. The calibration search uses March for both selection and reported
selection performance, so March is not an unbiased final estimate. April has
also been viewed during the broader project; the repository reconstructs a
machine-enforced pre-April selection path and reports that limitation rather
than claiming perfect historical blindness.
