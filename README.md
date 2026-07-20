# NBA Home-Win Probability Model (Audit Remediation)

Leakage-controlled, reproducible **probability forecasting prototype**.

Original submission preserved at git tag **`v1-original-submission`**.  
This branch (`cursor/audit-remediation-7b85`) remediates statistical governance.

## Start here

| Review time | Route |
|---|---|
| 2 minutes | `SUMMARY.md` + `artifacts/final_metrics.json` |
| Audit response | `docs/AUDIT_RESPONSE.md` |
| Selection proof | `artifacts/pre_march_selection_proof.json` |
| Primary April result | `outputs/april_predictions_frozen_snapshot.csv` |
| Reproduce | `make reproduce DATA=/absolute/path/to/nba-win-probability-data.csv` |

## Objective

Estimate \(P(\text{home win})\). Assignment ask: use October–March information
to produce April probabilities. Primary April artifact is the **frozen
March 31 snapshot**.

## Model (remediated)

Pre-March expanding-window selection chooses among:

- **Direct L2 logistic** on `elo_diff`, `bt_logit`, `trend_diff` (63-candidate
  compact grid over Elo \(K\), trend half-life, and logistic \(C\));
- optional **Platt-calibrated blend challenger** (not promoted unless it wins
  pre-March mean log loss).

Dense 341k temperature/shift search and four-target March gates are removed.

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
