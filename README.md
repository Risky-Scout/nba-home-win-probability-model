# NBA Home-Win Probability Model

A leakage-audited, reproducible implementation of the **Elo +
Bradley-Terry/recent-trend logit blend**. This repository is designed for a
technical interview: every material claim maps to executable code, a generated
artifact, or an automated test.

## Start here

| Review time | Route |
|---|---|
| 2 minutes | Read `SUMMARY.md`, then open `artifacts/final_metrics.json`. |
| 10 minutes | Follow `docs/INTERVIEW_WALKTHROUGH.md`. |
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
component logits — no grid search. This is equivalent to the earlier
(w, T, s) parameterization via w = a/(a+b), T = 1/(a+b), s = c.

The selected March-only specification is stored in
`artifacts/selected_spec.json`:

```text
architecture       hfa_75
Elo K              10
Elo home adjustment 75 rating points
Bradley-Terry C    0.15
trend half-life    45 days
short trend window 10 games
stacker a (elo logit)   0.5796
stacker b (rank logit)  0.9838
stacker c (intercept)   0.3154
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
- Base component coefficients are refitted through March.
- `sequential_daily` is the operational backtest: earlier April dates may
  update team state for later April dates, but no same-day or future result is
  visible.
- `frozen_snapshot` is also exported: no April result updates any April
  performance state.

These are different information sets. The repository reports both rather than
quietly mixing them.

## Results

### Operational one-step-ahead results

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| March model | 0.488029 | 0.157148 | 0.830336 | 78.6611% |
| April model | 0.458717 | 0.144995 | 0.853351 | 82.2917% |

### Strict month-start snapshot sensitivity

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| March frozen snapshot | 0.496766 | 0.162053 | 0.823026 | 75.7322% |
| April frozen snapshot | 0.454728 | 0.144498 | 0.863698 | 80.2083% |

The strict April snapshot improves proper scoring relative to the operational
version. Small AUC and accuracy differences at this sample size are not
presented as statistically decisive.

## What constitutes proof

| Claim | Executable evidence |
|---|---|
| Data contain 1,230 valid games | `scripts/data_audit.py`, `artifacts/data_audit.json` |
| Pregame records reconcile | `artifacts/data_audit.json` |
| Same-game box scores are excluded | `nba_wp/features.py`, `tests/test_feature_timing.py` |
| Same-date games are batched | `tests/test_feature_timing.py::test_same_day_games_are_batched` |
| April was absent from selection | `scripts/select_model.py`, `artifacts/selection_proof.json` |
| The stacker coefficients were fitted, not typed into scoring code | `artifacts/march_architecture_results.csv`, `artifacts/selected_spec.json` |
| Metrics recompute from individual prices | `validate_submission.py`, `tests/test_artifacts.py` |
| Saved probabilities reproduce | `python validate_submission.py --recompute ...` |
| Feature contribution is measurable | `artifacts/feature_group_ablation.csv`, `artifacts/permutation_importance.csv` |

## Excel companion — full mathematical transparency

[`NBA_Model_Fully_Formulated.xlsx`](NBA_Model_Fully_Formulated.xlsx) rebuilds the
entire model in live spreadsheet formulas: the sequential Elo chain for all
1,230 games, the frozen Bradley-Terry strengths, the trend EWMA, the component
logistics, the MLE logistic stacker, and the frozen April predictions. Sheet
`11_Reconcile_Dashboard` checks every stage against this repository's committed
artifacts (tolerances 1e-6 to 1e-9); all stacker coefficients and April
probabilities match `artifacts/selected_spec.json` and
`outputs/april_predictions_frozen_snapshot.csv` to machine precision.

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
