# NBA Home-Win Probability Model

A leakage-audited, reproducible implementation of the **Elo +
Bradley-Terry/recent-trend logit blend**. This repository is designed for a
technical interview: every material claim maps to executable code, a generated
artifact, or an automated test.

## Start here

| Review time | Route |
|---|---|
| 2 minutes | Read `SUMMARY.md`, then open `artifacts/final_metrics.json`. |
| 5 minutes | Memorize `docs/PRESENTATION_ONE_PAGER.md`. |
| Scorecard | Drill `docs/EVALUATION_MATRIX_PREP.md` (8 interview dimensions). |
| 12-15 minutes | Present with `docs/PRESENTATION_SCRIPT.md` (every modeling stage). |
| Q&A prep | Drill `docs/INTERVIEW_QA_CHEATSHEET.md`. |
| Submit | Follow `docs/SUBMIT_AND_PREP_CHECKLIST.md`. |
| 10 minutes | Follow `docs/INTERVIEW_WALKTHROUGH.md` for screen-share setup. |
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
combined in log-odds space:

```text
z = w * logit(p_elo) + (1 - w) * logit(p_rank)
p_home = sigmoid(z / temperature + shift)
```

The selected March-only specification is stored in
`artifacts/selected_spec.json`:

```text
architecture       hfa_75
Elo K              10
Elo home adjustment 75 rating points
Bradley-Terry C    0.15
trend half-life    45 days
short trend window 10 games
Elo blend weight   0.19
temperature        0.59
shift              0.33
```

## Evaluation protocol and information policy

### Selection

- Base model coefficients for March are fitted through February.
- March is an **operational one-step-ahead validation**: after all games on a
  date are priced, that date's results may update state for later dates.
- `scripts/select_model.py` truncates the input at March 31 before feature
  construction. It cannot read an April row.
- Five declared architectures and 68,231 calibration settings per architecture
  are evaluated.
- A candidate must beat all four March numerical targets. Eligible candidates
  are ordered by log loss, Brier, AUC, and accuracy.

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
| March model | **0.487569** | **0.156834** | **0.831798246** | **77.8243%** |
| March target | 0.509645 | 0.167618 | 0.831798000 | 77.8200% |
| April model | **0.463375** | **0.145639** | 0.850202 | **83.3333%** |
| April target | 0.468596 | 0.150628 | **0.868196** | 81.2500% |

The March selection result exceeds all four rounded targets. The AUC and
accuracy margins are very small and are not presented as statistically
decisive. On April, the model beats log loss, Brier, and accuracy, but **does
not beat the AUC target**.

### Strict month-start snapshot sensitivity

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| March frozen snapshot | 0.497736 | 0.162489 | 0.822442 | 76.1506% |
| April frozen snapshot | 0.458942 | 0.145299 | 0.862798 | 80.2083% |

The strict April snapshot improves proper scoring relative to the operational
version but remains below the stated AUC and accuracy targets.

## What constitutes proof

| Claim | Executable evidence |
|---|---|
| Data contain 1,230 valid games | `scripts/data_audit.py`, `artifacts/data_audit.json` |
| Pregame records reconcile | `artifacts/data_audit.json` |
| Same-game box scores are excluded | `nba_wp/features.py`, `tests/test_feature_timing.py` |
| Same-date games are batched | `tests/test_feature_timing.py::test_same_day_games_are_batched` |
| April was absent from selection | `scripts/select_model.py`, `artifacts/selection_proof.json` |
| The chosen grid point was generated, not typed into scoring code | `artifacts/march_architecture_results.csv`, `artifacts/march_tuning_top_candidates.csv` |
| Metrics recompute from individual prices | `validate_submission.py`, `tests/test_artifacts.py` |
| Saved probabilities reproduce | `python validate_submission.py --recompute ...` |
| Feature contribution is measurable | `artifacts/feature_group_ablation.csv`, `artifacts/permutation_importance.csv` |

## Run locally

```bash
git clone <your-repository-url>
cd <repository>

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
│   ├── model.py              # base models, blend, metrics, grid search
│   ├── selection.py          # March-only model selection
│   └── reporting.py          # prices, evidence artifacts, figures
├── configs/                  # declared candidates, targets, selection policy
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
