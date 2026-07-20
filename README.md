# NBA Home-Win Probability — Bet365 technical task

Calibrated home-team win probabilities for April 2026 NBA games, trained,
validated, and tested on October–March only.

## 1. Objective

Predict \(P(\text{home win})\) for each of the 96 April games. Treated as a
**pricing** problem: the deliverable is a probability and its zero-margin fair
decimal odds \(1/p\), optimized and judged by log loss.

## 2. Main result

| Stage | Log loss | Brier | AUC | Correct |
|---|---|---|---|---|
| Locked March test (scored once) | 0.5103 | 0.1672 | 0.826 | 182/239 |
| **April forecast (frozen 2026-03-31)** | **0.4695** | **0.1511** | **0.862** | **74/96** |

Deliverable: [`predictions/april_predictions.csv`](predictions/april_predictions.csv).
All report numbers are generated programmatically from
[`reports/metrics.json`](reports/metrics.json) — see
[`reports/model_report.md`](reports/model_report.md).

## 3. Information-timing rule

A prediction for game *g* uses only games completed strictly before *g*'s
date. Postgame columns (points, turnovers, fouls, rebounds) never enter
same-game features. Team state updates **after** features are written;
same-date games are batched. The supplied wins/losses columns were verified
pregame by replaying all 1,230 results (0 mismatches).

## 4. Temporal split

| Period | Role |
|---|---|
| Oct–Feb | Development |
| train < Jan → validate Jan; train < Feb → validate Feb | Expanding-window selection folds |
| March | Locked test — scored once after the spec froze |
| April | Forecast — refit through Mar 31, state frozen |

## 5. Model selection

Declared budget in [`configs/model.yaml`](configs/model.yaml): 3 Elo K × 3
trend half-lives × 7 L2 strengths = 63 direct logistics + 9 blend challengers
= **72 candidates**. Primary metric: mean validation log loss.

**Selected:** direct L2 logistic on `elo_diff`, `bt_logit`, `trend_diff`
(K=10, half-life 20 d, C=0.1). Coefficients:
[`artifacts/current/model_coefficients.json`](artifacts/current/model_coefficients.json).

- **Was March untouched?** Yes — selection input ends 2026-02-28; guards +
  tests raise on March rows; proof:
  [`artifacts/current/pre_march_selection_proof.json`](artifacts/current/pre_march_selection_proof.json).
- **Was April used in selection?** No (0 rows). April was, however, viewed
  earlier in this project's life, so it is a retrospective scoring period,
  not a pristine holdout.
- **Baselines:** the model decisively beats constant-rate and
  record-difference baselines (paired CIs exclude 0); its edge over an
  Elo-only logistic is statistically unresolved — disclosed, with Elo-only
  recorded as the simpler challenger.

## 6. April forecast process

Refit the unchanged selected pipeline on October–March; freeze team state at
2026-03-31; score all 96 April games from that state. A rolling-daily
simulation is kept as a **separate descriptive scenario**
(`predictions/april_predictions_rolling_scenario.csv`) and is never mixed with
the frozen output.

Fair odds are zero-margin model prices — **not** recommended bookmaker offer
prices (no overround, liability, market consensus, or trader adjustments).

## 7. Repository structure

```text
configs/model.yaml        one declared configuration (protocol + budget)
src/nba_wp/               data → features → model → selection → evaluation → cli
scripts/                  report generation, verification, manifest
tests/                    leakage, protocol, prediction, CLI contracts
predictions/              April deliverable (frozen) + rolling scenario
reports/                  audit, model report, presentation guide, metrics.json
artifacts/current/        machine-readable proofs and diagnostics
artifacts/v1_legacy/      superseded v1 outputs (tag v1-original-submission)
outputs/, figures/        prediction CSVs and plots
notebooks/                00_interview_walkthrough.ipynb
docs/                     methodology, governance, interview pack
```

## 8. Reproduce

macOS / Linux:

```bash
uv sync --frozen
uv run pytest -q
uv run python -m nba_wp.cli predict \
  --data data/nba-win-probability-data.csv \
  --config configs/model.yaml \
  --output predictions/april_predictions.csv
uv run python validate_submission.py --root . \
  --data data/nba-win-probability-data.csv --recompute
```

Windows PowerShell:

```powershell
uv sync --frozen
uv run pytest -q
uv run python -m nba_wp.cli predict `
  --data data\nba-win-probability-data.csv `
  --config configs\model.yaml `
  --output predictions\april_predictions.csv
uv run python validate_submission.py --root . `
  --data data\nba-win-probability-data.csv --recompute
```

(Without uv: `python -m pip install -e . pytest` then the same
`python -m nba_wp.cli ...` commands.)

Place the assignment CSV at `data/nba-win-probability-data.csv` (gitignored).
Full regeneration including selection: `make reproduce DATA=data/nba-win-probability-data.csv`.

## 9. Limitations

1. April is retrospective (viewed earlier in the project), not pristine.
2. No bookmaker prices → no edge/CLV/profitability claims.
3. 96 forecast games → wide intervals; Elo-only equivalence unresolved.
4. No injury/lineup/travel data; late-season rotation changes unmodeled.

## 10. Production extensions

Market-odds ingestion and de-vigging, lineup/injury feeds, scheduled refits
with calibration monitoring and rollback, multi-season backtesting.

---

Original submission preserved at tag `v1-original-submission`. Audit trail:
[`reports/repository_audit.md`](reports/repository_audit.md) and
[`docs/AUDIT_RESPONSE.md`](docs/AUDIT_RESPONSE.md). Interview materials:
[`reports/presentation_guide.md`](reports/presentation_guide.md) and
[`docs/interview/`](docs/interview/).
