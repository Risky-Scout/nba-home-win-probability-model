# NBA Home-Win Probability Model (Audit Remediation)

Leakage-controlled, reproducible **probability forecasting prototype**.

Original submission preserved at git tag **`v1-original-submission`**.  
This branch (`cursor/audit-remediation-7b85`) remediates statistical governance.

## Start here

| Review time | Route |
|---|---|
| 2 minutes | `SUMMARY.md` + `artifacts/current/final_metrics.json` |
| Audit response | `docs/AUDIT_RESPONSE.md` |
| Selection proof | `artifacts/current/pre_march_selection_proof.json` |
| Primary April result | `outputs/april_predictions_frozen_snapshot.csv` |
| Artifact layout | `artifacts/README.md` |
| Reproduce | `make reproduce DATA=/absolute/path/to/nba-win-probability-data.csv` |

## Objective

Estimate \(P(\text{home win})\). Assignment ask: use October–March information
to produce April probabilities. Primary April artifact is the **frozen
March 31 snapshot**.

## Model (remediated)

Pre-March expanding-window selection chooses among:

- **63 direct L2 logistic** candidates on `elo_diff`, `bt_logit`, `trend_diff`
  (Elo \(K\) × trend half-life × logistic \(C\));
- **nine architecture-matched Platt-calibrated blend challengers**;
- **72 candidates total**.

Dense ~341k temperature/shift search and four-target March gates are removed.

## Evaluation policy

| Stage | Role |
|---|---|
| Oct–Dec / Jan fold; Oct–Jan / Feb fold | Architecture & hyperparameter selection |
| March | Locked pre-final test |
| April frozen March 31 | **Primary assignment result** |
| April sequential daily | Operational sensitivity only |

External values in `configs/benchmarks.json` are retrospective reference only.
See `docs/BENCHMARK_PROVENANCE.md`.

## Results language

Do **not** claim “beats all four March targets.” Prefer:

> The model produced lower March log loss and Brier score than the retrospective
> reference values when that occurs. March AUC and accuracy were effectively
> ties at the reported precision. Report exact correct-game counts
> (e.g. `correct_games / 239`).

April is the assignment’s retrospective scoring period. The executable
selection pipeline uses zero April rows, but April had previously been viewed
during the broader project, so I do not claim that it is a pristine untouched
holdout.

Pre-March selection on this branch is a **reconstructed governance path**, not
historical preregistration before April was viewed.

## Model comparison claim (important)

The direct logistic won the declared pre-March validation process, but its
incremental April value over a simpler Elo model remains **statistically
unresolved**: date-block proper-score differences versus Elo are small and
include zero (point estimate for log-loss difference can favor Elo).

## Market / production claims

Outputs are **model-estimated fair probabilities** and **zero-margin fair odds**.
This conversion excludes overround, market consensus, liability, limits,
injuries, news, and trader adjustments. See
`docs/MARKET_PRICING_LIMITATIONS.md`.

This repository claims **prototype / research readiness**, not deployable
sportsbook production readiness.

## Run locally

```bash
git checkout cursor/audit-remediation-7b85
bash scripts/bootstrap_macos.sh
source .venv/bin/activate
# place CSV at data/nba-win-probability-data.csv
make reproduce DATA=data/nba-win-probability-data.csv
```

## Acceptance posture

Leakage-controlled and reproducible probability modeling with pre-March
chronological selection, locked March test, frozen March 31 April primary
scoring, and date-block uncertainty. **Not** proof of sportsbook alpha and
**not** a production betting system.
