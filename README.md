# NBA Home-Win Probability Model

A leakage-audited, reproducible NBA home-win probability model. The **deployed
champion is Elo-only** — a logistic map on a margin-of-victory Elo rating
differential. An Elo + Bradley-Terry/recent-trend logit blend was implemented,
validated, and **rejected**: under honest nested rolling-origin validation it
does not beat Elo-only out-of-sample on either proper score, and it is worse
calibrated. This repository is designed for a technical interview: every
material claim maps to executable code, a generated artifact, or an automated
test, and the headline decision (reject the more complex model) is itself
backed by evidence.

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

### Deployed champion: Elo-only

A single, transparent component: **margin-of-victory Elo** — sequential team
ratings with a home adjustment — mapped to a probability by an L2-logistic
model on the pregame Elo rating differential:

```text
p_home = sigmoid( c + w * z(elo_diff) )
```

where `z(elo_diff)` standardizes the Elo differential by the training
mean/scale. The probability map is fit on **all eligible games through
March 31**; April performance state is frozen at March 31.

The selected specification is stored in `artifacts/selected_spec.json`:

```text
model_family            elo_only
architecture            conservative
Elo K                   7.5
Elo home adjustment     55 rating points
Elo MOV multiplier      winner Elo - loser Elo (FiveThirtyEight log form)
logistic C              10
standardized weight w    0.9272
intercept c              0.2415
training mean/scale      0.1408 / 0.2690
raw form                 raw_unit_coefficient 3.446 per elo_diff, raw_intercept -0.2438
```

The raw form is exposed in `selected_spec.json` under `elo_model` as
`raw_unit_coefficient` **and** `raw_intercept`; the raw-unit coefficient must be
paired with `raw_intercept` (which already absorbs the centering shift), never
with the standardized `intercept`.

### Rejected challenger: Elo + Bradley-Terry/recent-trend blend

The blend adds a second component (regularized Bradley-Terry strength plus a
short-minus-long point-margin trend) and combines the two component
probabilities with a **convex logistic stacker** deployed under a temperature
floor `T >= 1`:

```text
p_blend = sigmoid(a * logit(p_elo) + b * logit(p_rank) + c),  a,b >= 0, a+b <= 1
```

The stacker is a genuine convex logit blend: `fit_logit_stacker` clips negative
weights to zero and enforces `T = 1/(a+b) >= 1` (no sharpening), so
`0 <= w <= 1` for both components (`tests/test_stacker_temperature_floor.py`).
This blend is **retained only as a rejected challenger** — see the nested audit
below. Its coefficients and per-period metrics live in the `challenger` block
of `artifacts/selected_spec.json`.

## Evaluation protocol and information policy

### Selection (model-specific, April-blind)

- `scripts/select_model.py` truncates the input at the derived selection cutoff
  before feature construction (the boundary is derived from the data by
  `nba_wp/periods.py` — last calendar month is the April holdout, the prior
  month is the March selection window — and for this dataset resolves to
  `2026-03-31`). It cannot read an April row (`artifacts/selection_proof.json`
  records `april_rows_loaded = 0`).
- The deployed Elo architecture is chosen by **aggregate frozen-policy rolling
  out-of-sample log loss** across pre-holdout weekly origins, with a
  **one-standard-error stability rule** that prefers the simplest/lowest-K
  architecture inside the noise band. All five candidate architectures fall
  within one SE of the best; `conservative` is selected because it has both the
  lowest mean OOS log loss and the lowest K. A single March one-step-ahead split
  would have chosen the same architecture. See `architecture_selection` in
  `artifacts/selected_spec.json`.
- The MOV multiplier offset (2.2) and Elo cold-start warmup are also profiled on
  the same frozen OOS surface: the offset is profiled on the grid
  {1.6, 2.0, 2.2, 2.6, 3.0} and 2.2 is kept because it is within one SE of the
  nominal best (3.0), so it is now empirically confirmed on our data rather than
  merely borrowed from FiveThirtyEight; a provisional-K warmup was implemented
  and profiled but kept **off** (warmup worsened OOS log loss). A same-season
  record prior was intentionally omitted as redundant with Elo. See
  `mov_offset_selection` and `cold_start_selection` in `selected_spec.json`.
- The **champion is Elo-only**: the nested audit (below) shows the blend does
  not beat Elo-only out-of-sample, so the simpler model is deployed. The
  deployed Elo probability map is then refit on all rows through the selection
  cutoff (March 31).

### April

- **Primary deliverable** `outputs/april_predictions.csv` is the
  **frozen pre-April** file: performance state is frozen at March 31 and the
  Elo probability map is fit through March 31, so no April result can change any
  April price.
- `outputs/april_predictions_sequential_backtest.csv` is an optional
  live-update simulation only (earlier April dates may update team state for
  later April dates). It is clearly labelled and is not the headline result.
- `outputs/challenger_blend_april_predictions.csv` holds the rejected blend's
  frozen April prices for transparency.

## Results

### Primary April holdout (frozen pre-April, Elo-only champion)

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| **April frozen (primary)** | 0.464369 | 0.149770 | 0.866847 | 78.1250% |

Frozen April price distribution: min `home_win_probability` ≈ 0.069, max ≈ 0.970;
9 of 96 games priced at or above 0.90. Mean forecast 0.549 vs. observed home rate
0.594 — the champion is if anything mildly **under**-forecasting on this window
(consistent with the nested calibration slope > 1 below), not overconfident. The
high prices come from large Elo rating gaps (elite home team vs. weak visitor),
and the leakage battery + frozen policy rule out same-game/future leakage.

For reference, the **rejected blend** on the same frozen April window scores
log loss 0.4687 / Brier 0.1505 — worse than Elo-only on both proper scores.

### March (selection period — Elo-only, in-sample for selection)

| Model | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| Elo-only (fit through Feb, one-step March) | 0.506590 | 0.165880 | 0.823392 | 76.9874% |

March is used to select architectures, so it is in-sample for selection and is
not a pristine holdout. The honest out-of-sample evidence is the nested audit.

### Optional April sequential backtest (live-update simulation)

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| April sequential (Elo-only) | 0.464648 | 0.149914 | 0.866397 | 78.1250% |

The sequential backtest lets earlier April dates update team state for later
April dates; the frozen version is the honest answer to "use October-March to
price April" and is the primary submission.

### Nested rolling-origin validation (the honest verdict)

`scripts/nested_validation.py` validates the whole model-building **procedure**
under two clearly separated information policies, never mixed into one metric:

- **Frozen-block**: for each outer weekly origin O, performance state is frozen
  at O−1 (`build_features(freeze_date=O)`); every price in the block `[O, O+7)`
  uses only information available before O. Mutating any outcome inside the
  block cannot change any price in the block (verified in-script:
  `frozen_block_leakage_guarantee_verified = true`, and
  `tests/test_information_policies.py`). Its inner architecture selection now
  runs under a **frozen inner policy** (policy-matched frozen-inner/frozen-outer),
  fixing a prior mismatch where the frozen block used sequential inner selection.
- **Daily-sequential**: a price for date t uses results strictly before t
  (base models refit through t−1) — a live one-step-ahead simulation, with a
  matching **sequential inner** selection policy.

For every outer fold, each procedure **independently** selects its own
architecture by its own inner out-of-fold score (under the policy-matched inner
information policy); the convex stacker is trained on inner **out-of-fold**
component predictions. Six candidates are priced on identical outer-fold rows
(501 out-of-sample games, 11 weekly folds):

| Candidate (nested OOS, 501 games) | LL (frozen) | Brier (frozen) | LL (daily) | Brier (daily) | Cal. slope β |
|---|---:|---:|---:|---:|---:|
| Constant home rate | 0.688 | 0.247 | 0.688 | 0.247 | — |
| **Elo-only (champion)** | **0.532** | **0.177** | **0.532** | **0.177** | 1.32–1.37 |
| Rank-only (BT + trend) | 0.550 | 0.184 | 0.549 | 0.184 | 1.64–1.70 |
| Convex blend (rejected) | 0.548 | 0.183 | 0.547 | 0.182 | 1.75–1.80 |
| Calibrated Elo (rejected) | 0.548 | 0.183 | 0.549 | 0.184 | 1.76–1.78 |
| Schedule Elo (rejected) | 0.531 | 0.177 | 0.533 | 0.177 | 1.33 |

Week-block bootstrap, blend − Elo-only (frozen-block): **ΔlogLoss = +0.017**
(95% CI `[+0.010, +0.023]`), **ΔBrier = +0.006** (95% CI `[+0.004, +0.009]`).
**0 of 4,000** week-block bootstrap replicates favored the blend on either
metric (with only ~11 weekly blocks, treat this as strong directional evidence,
not production-grade certainty from a large independent sample).

**Calibration is now reported for every candidate**, not just the loser. On the
frozen-block policy the Elo-only champion is the best-calibrated candidate:
intercept α = −0.05 (95% CI `[−0.32, +0.22]`), slope β = 1.37 (95% CI
`[1.22, 1.57]`), ECE ≈ 0.059, mean forecast 0.554 vs. observed 0.557. The blend
is worse: β ≈ 1.80 and ECE ≈ 0.092–0.115 (more compressed toward 0.5). β > 1
means the Elo-only champion is *mildly under*confident (it could be sharpened
~1.3×), which is a much safer failure mode for pricing than overconfidence.

**Champion–challenger decision (`decision: keep_elo_only` under both policies).**
The rule "promote the blend only if it beats Elo-only on **both** log loss and
Brier with a block-bootstrap upper CI below zero" is **not** met. Elo-only wins
on discrimination, proper scores, and calibration. See
`artifacts/nested_frozen_block_summary.json`,
`artifacts/nested_daily_sequential_summary.json`,
`figures/nested_frozen_block_reliability.png`, and
`figures/nested_daily_sequential_reliability.png`.

**Two further challengers were added to the audit and also rejected under both
policies** (both `decision: keep_raw_elo`):

- **Calibrated Elo** (`calibration_challenger`): a cross-fitted, identity-shrunk
  Elo calibrator (`logit(p) = α + β·logit(p_elo)`, fit only on inner OOF and
  shrunk toward α=0, β=1). It **over-corrects** — the calibration slope rises
  from ~1.33 to ~1.79 and ECE worsens from ~0.06 to ~0.09, with worse log loss —
  because inner-OOF Elo is trained on less data and is more underconfident than
  the deployed Elo. Keep the raw Elo.
- **Schedule Elo** (`schedule_challenger`): Elo + rest + back-to-back logistic
  (strongly regularized). Slightly worse OOS log loss and Brier under both
  policies. Keep the raw Elo.

## What constitutes proof

| Claim | Executable evidence |
|---|---|
| Data contain 1,230 valid games | `scripts/data_audit.py`, `artifacts/data_audit.json` |
| Pregame records reconcile | `artifacts/data_audit.json` |
| Same-game box scores are excluded | `nba_wp/features.py`, `tests/test_feature_timing.py` |
| Same-date games are batched | `tests/test_feature_timing.py::test_same_day_games_are_batched` |
| April was absent from selection | `scripts/select_model.py`, `artifacts/selection_proof.json` |
| Deployed model is Elo-only and matches the champion decision | `artifacts/selected_spec.json`, `tests/test_champion_promotion.py` |
| Elo architecture selected by aggregate frozen-policy OOS log loss with a one-SE stability rule (lowest K within band; single March split would pick the same) | `nba_wp/selection.py`, `artifacts/selected_spec.json` (`architecture_selection`), `tests/test_architecture_selection.py`, `tests/test_champion_promotion.py::test_elo_architecture_is_selected_on_elo_oof_loss` |
| MOV offset (2.2) and cold-start warmup (off) are data-driven, profiled on the frozen OOS surface | `nba_wp/selection.py`, `artifacts/selected_spec.json` (`mov_offset_selection`, `cold_start_selection`) |
| Deployed spec, final metrics, and prices all describe the same champion | `tests/test_champion_promotion.py` |
| Primary April prices recompute from the deployed Elo coefficients | `tests/test_champion_promotion.py::test_primary_april_probabilities_recompute_from_deployed_elo` |
| Elo MOV uses winner - loser rating diff (upsets move ratings more) | `nba_wp/features.py`, `tests/test_elo_mov_winner_diff.py` |
| Challenger stacker is a genuine convex blend (0<=w<=1, T>=1) | `nba_wp/model.py`, `tests/test_stacker_temperature_floor.py::test_stacker_weights_are_convex_when_stacker_is_used` |
| Primary April is frozen (April outcomes cannot change April prices) | `nba_wp/reporting.py`, `tests/test_primary_april_frozen.py`, `tests/test_information_policies.py` |
| Frozen outer block ignores all in-block outcomes; daily uses only prior dates | `tests/test_information_policies.py` |
| Leakage battery: future/box-score/row-order do not move features; past outcomes provably do (positive controls) | `tests/test_leakage_mutations.py` |
| Nested rolling-origin validation under two policies (frozen-block + daily-sequential), policy-matched inner selection (frozen-inner/frozen-outer, sequential-inner/sequential-outer) + OOF stacker, candidate comparison, block-bootstrap, calibration for every candidate, champion-challenger | `scripts/nested_validation.py`, `artifacts/nested_frozen_block_summary.json`, `artifacts/nested_daily_sequential_summary.json`, `artifacts/nested_frozen_block_folds.csv`, `figures/nested_frozen_block_reliability.png`, `figures/nested_daily_sequential_reliability.png` |
| Calibrated-Elo challenger (cross-fitted, identity-shrunk) is rejected — it over-corrects out-of-sample | `scripts/nested_validation.py`, `artifacts/nested_frozen_block_summary.json` / `artifacts/nested_daily_sequential_summary.json` (`calibration_challenger`), `tests/test_calibration_challenger.py` |
| Schedule-Elo challenger (Elo + rest + back-to-back) is rejected — slightly worse OOS proper scores | `scripts/nested_validation.py`, `artifacts/nested_frozen_block_summary.json` / `artifacts/nested_daily_sequential_summary.json` (`schedule_challenger`) |
| `raw_unit_coefficient` must be paired with `raw_intercept` (not the standardized intercept) | `artifacts/selected_spec.json` (`elo_model.raw_intercept`), `tests/test_workbook_reconstruction.py` |
| Excel workbook reconciles to the spec and metrics from one command | `scripts/workbook_reconciliation.py` (`python -m scripts.workbook_reconciliation`), `artifacts/workbook_reconciliation.json` (status PASS), `tests/test_workbook_reconstruction.py` |
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

The full workflow (`python run_submission.py --mode full`) performs the complete
chain:

1. data audit;
2. April-blind architecture, MOV-offset, and cold-start selection (dates derived
   from the data by `nba_wp/periods.py`);
3. March and April scoring (ablation, coefficients, permutation importance,
   calibration bins, figures);
4. nested rolling-origin validation (both policies);
5. Excel workbook rebuild;
6. workbook reconciliation (`artifacts/workbook_reconciliation.json`);
7. pytest report;
8. manifest generation;
9. independent metric verification.

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
│   ├── selection.py          # frozen-policy OOS one-SE architecture selection
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
seasons. March is used for both selection and reported selection performance,
so March is not an unbiased final estimate. April has also been viewed during
the broader project (multiple model revisions, an interview review, and
diagnostic procedures chosen after observing failures), so February–April is no
longer a pristine holdout; a strong sportsbook-calibration claim would require a
future untouched period or a separate season. The repository reconstructs a
machine-enforced pre-April selection path and reports these limitations rather
than claiming perfect historical blindness.

Honest verdict: the fully nested rolling-origin audit
(`scripts/nested_validation.py`) shows the Elo + rank blend does **not** beat
Elo-only out-of-sample — it is worse on log loss and Brier under both
information policies and worse calibrated (blend β ≈ 1.8 vs. Elo-only β ≈ 1.35).
Two further challengers — a cross-fitted calibrated Elo and an Elo + schedule
(rest/back-to-back) logistic — were added to the same audit and are **also
rejected** under both policies. The single statistically strong claim is
precisely this: the more complex challengers (blend, calibrated Elo, schedule
Elo) are rejected out-of-sample. The repository therefore **deploys Elo-only as
the champion** and keeps the alternatives only as clearly labelled, rigorously
validated rejected challengers. The value of this project is the reproducible,
leakage-controlled pipeline and a validation honest enough to reject its own
added complexity.
