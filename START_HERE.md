# Start here — Bet365 interview navigation

Open this file first. Everything else hangs off it.

## What you are defending

**Assignment:** home-win probability for April games using Oct–Mar information  
**Selected model:** direct L2 logistic on `elo_diff`, `bt_logit`, `trend_diff`  
**Selection:** Jan/Feb expanding folds only (no March/April in selection)  
**Primary April result:** frozen March 31 snapshot → `predictions/april_predictions.csv`  
**Claim:** probability-forecasting prototype — not sportsbook alpha

## 90-minute path (open these in order)

| # | Open this | Why |
|---|---|---|
| 1 | `START_HERE.md` | You are here |
| 2 | `SUMMARY.md` | 2-minute story |
| 3 | `reports/presentation_guide.md` | Timed 90-min session plan + Q&A |
| 4 | `docs/interview/PRESENTATION_SCRIPT_90MIN.md` | Speak-from script |
| 5 | `docs/interview/PARAMETER_LEDGER.md` | Every number → fitted / selected / fixed |
| 6 | `reports/model_report.md` | All metrics (generated from metrics.json) |
| 7 | `docs/SENIOR_QUANT_QA.md` | Interrupt answer key |
| 8 | `notebooks/00_interview_walkthrough.ipynb` | Live-demo notebook (HTML backup beside it) |

## Mathematical spine (one screen)

\[
\operatorname{logit}\hat p
=
\hat\beta_0
+
\hat\beta_1\,\Delta\mathrm{Elo}
+
\hat\beta_2\,\Delta\mathrm{BT}
+
\hat\beta_3\,\Delta\mathrm{Trend}
\]

Fitted coefficients (through March, standardized then mapped to raw units):

| Term | Standardized \(\hat\beta\) | Where |
|---|---|---|
| intercept | 0.2385 | `artifacts/current/model_coefficients.json` |
| elo_diff | 0.7321 | same |
| bt_logit | 0.1560 | same |
| trend_diff | 0.0617 | same |

Selected hyperparameters:

| Parameter | Value | Class | Evidence |
|---|---|---|---|
| Elo \(K\) | 10 | Selected (pre-March) | `selected_spec_pre_march.json` |
| Elo HFA | 65 | Fixed in search (default) | `configs/model.yaml` |
| Trend half-life | 20 days | Selected | same |
| Logistic \(C\) | 0.1 | Selected | same |
| Elo start / scale | 1500 / 400 | Fixed by design | `src/nba_wp/features.py` |

## Assignment artifacts (primary)

| Role | Path |
|---|---|
| April probabilities (primary) | `predictions/april_predictions.csv` |
| Selection proof | `artifacts/current/pre_march_selection_proof.json` |
| Locked metrics | `artifacts/current/final_metrics.json` |
| Parameter ledger | `docs/interview/PARAMETER_LEDGER.md` |
| Audit posture | `docs/AUDIT_RESPONSE.md` |

## Folder map

```text
START_HERE.md          ← interview landing
SUMMARY.md             ← 2-minute summary
README.md              ← repo overview
docs/
  interview/           ← presentation script, ledger, file index
  assignment/          ← original PDF task + JD
  *.md                 ← methodology, governance, results
nba_wp/                ← source (data → features → model → selection)
configs/               ← search budget + selection policy
artifacts/current/     ← authoritative numbers (open these)
artifacts/v1_legacy/   ← original submission search files only
outputs/               ← March/April prediction CSVs
figures/               ← calibration / importance plots
tests/                 ← leakage + selection cutoff tests
```

## Pre-interview check (2 minutes)

```bash
uv sync --frozen
uv run pytest -q
uv run python validate_submission.py --root . --data data/nba-win-probability-data.csv --recompute
```

Expect all tests green and `"status": "PASS"`.

## What not to claim

- Sportsbook edge / CLV / profitability  
- April as pristine untouched holdout  
- That the three-feature model clearly beats Elo-only (paired CI includes 0)  
- Production deployability

## Disclose before they find it

1. April was viewed earlier in the project → retrospective, not pristine.
2. Elo-only baseline is statistically indistinguishable — kept as challenger.
3. Calibration slope ≈1.44 at n=96 is a diagnostic, not a solved property.
4. December-fold protocol sensitivity (see `configs/model.yaml` comment and
   `reports/model_report.md`).
