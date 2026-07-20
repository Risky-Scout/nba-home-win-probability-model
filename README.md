# NBA Home-Win Probability — Bet365 take-home

Interview landing page: **[`START_HERE.md`](START_HERE.md)**

Leakage-controlled, reproducible **probability forecasting prototype** for the
Bet365 Sports Quantitative Analyst assignment.

## Open first (interview)

| Priority | File |
|---|---|
| 1 | [`START_HERE.md`](START_HERE.md) |
| 2 | [`SUMMARY.md`](SUMMARY.md) |
| 3 | [`docs/interview/PRESENTATION_SCRIPT_90MIN.md`](docs/interview/PRESENTATION_SCRIPT_90MIN.md) |
| 4 | [`docs/interview/PARAMETER_LEDGER.md`](docs/interview/PARAMETER_LEDGER.md) |
| 5 | [`docs/SENIOR_QUANT_QA.md`](docs/SENIOR_QUANT_QA.md) |
| 6 | [`docs/interview/FILE_INDEX.md`](docs/interview/FILE_INDEX.md) |

## Assignment

| Item | Path |
|---|---|
| Instructions PDF | `docs/assignment/nba-win-probability-instructions.pdf` |
| Job description PDF | `docs/assignment/Quantitative_Analyst_Sports_JD.pdf` |
| Data CSV | `data/nba-win-probability-data.csv` (gitignored; place locally) |

**Task:** Oct–Mar → home-win probabilities for April.

## Champion model (current branch)

Direct L2 logistic:

\[
\operatorname{logit}\hat p
=
\beta_0+\beta_1\Delta\mathrm{Elo}+\beta_2\Delta\mathrm{BT}+\beta_3\Delta\mathrm{Trend}
\]

| Item | Value |
|---|---|
| Selection | Jan/Feb expanding folds; mean log loss |
| Architecture | `k10_hl20` (\(K=10\), half-life 20d) |
| Regularization | \(C=0.1\) |
| Primary April | Frozen March 31 snapshot |
| Coefficients | `artifacts/current/model_coefficients.json` |

Original submission preserved at git tag `v1-original-submission`.

## Repository map

```text
START_HERE.md                 Interview hub
SUMMARY.md                    Two-minute summary
docs/interview/               Script, parameter ledger, file index
docs/assignment/              Original PDFs
docs/                         Methodology, governance, results
nba_wp/                       Library: data → features → model → selection
configs/                      Declared search + selection policy
scripts/                      CLI entry points
artifacts/current/            Authoritative metrics & proofs
artifacts/v1_legacy/          Old v1 search artifacts only
outputs/                      Prediction CSVs (April frozen = primary)
figures/                      Plots
tests/                        Leakage / cutoff / synthetic E2E
```

## Reproduce

```bash
bash scripts/bootstrap_macos.sh   # or: python -m pip install -r requirements-dev.txt
source .venv/bin/activate         # if created
# place CSV at data/nba-win-probability-data.csv
make reproduce DATA=data/nba-win-probability-data.csv
python validate_submission.py --root . --data data/nba-win-probability-data.csv --recompute
```

## Claims discipline

- Model-estimated fair probabilities / zero-margin fair odds only  
- No sportsbook alpha without bookmaker prices  
- April is retrospective, not pristine  
- Prototype readiness — not production deployable  

Details: `docs/AUDIT_RESPONSE.md`, `docs/MARKET_PRICING_LIMITATIONS.md`.
