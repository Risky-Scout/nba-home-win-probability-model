# Reviewer guide — the repository in ten minutes

A suggested route through the evidence, in reading order.

## 1. Verify everything first (~1 minute)

```bash
python validate_submission.py --root . --data data/nba-win-probability-data.csv --recompute
```

Reloads the raw CSV, re-audits it, rebuilds the model from
`artifacts/selected_spec.json` in a temporary directory, and compares every
March and April probability to the committed CSVs. Expected output:
`"status": "PASS"`.

## 2. Data audit

`artifacts/data_audit.json` — 1,230 games, 30 teams × 82, no missing values,
no duplicate IDs. The supplied pregame win/loss columns reconcile exactly
against a full replay of the season (0 mismatches), which is how the pregame
claim was verified rather than assumed. The task text says fourteen columns;
the enumerated schema is sixteen, and the loader whitelists exactly those.

## 3. Leakage control

`nba_wp/features.py::build_features` — for each date: read team state, write
every feature row, and only then update state from that date's results.
Same-day games are batched; a game's own box score cannot enter its own
features. Executable proof:

```bash
python -m pytest tests/test_feature_timing.py -q
```

## 4. Model

`docs/METHODOLOGY.md` — margin-of-victory Elo and a regularized Bradley–Terry
model with a recent-form trend, each calibrated by a small logistic model,
combined by a logistic stacker fitted by penalized maximum likelihood on March
component logits (coefficients in `artifacts/selected_spec.json`, with the
(w, τ, s) equivalence note).

## 5. Selection provenance

- Declared candidates: `configs/architecture_candidates.json` (all five)
- All five scored: `artifacts/march_architecture_results.csv`
- Rule: minimize March log loss (verbatim in `selected_spec.json`)
- April exclusion proof: `artifacts/selection_proof.json` (max input date
  2026-03-31, zero April rows; the selection function raises otherwise)

## 6. Baselines and feature evidence

`artifacts/feature_group_ablation.csv` — constant prior through record-only,
Elo-only, BT+trend, a rich linear challenger, and the selected blend, all on
the same protocol. `artifacts/permutation_importance.csv` and
`artifacts/paired_bootstrap_vs_*.json` quantify component contributions with
paired uncertainty; the blend-versus-Elo-alone interval crosses zero and is
reported as such.

## 7. Results

`artifacts/final_metrics.json` — March (the selection set, in-sample for the
stacker) and April under two clearly separated information policies:
sequential daily and a strict March-31 frozen snapshot. Reliability diagrams in
`figures/`, bin tables in `artifacts/*_calibration_bins.csv`. Fair decimal
odds in the prediction CSVs are zero-margin transforms of the probabilities.

## 8. Known limitations

`docs/LIMITATIONS_AND_ROADMAP.md` — March double-use, prior April exposure at
the project level, no injury/lineup/travel data, single season, and the
production extensions that would address each.

## Excel companion

`NBA_Model_Fully_Formulated.xlsx` rebuilds the model in live spreadsheet
formulas and reconciles every stage against the committed artifacts
(`11_Reconcile_Dashboard`).
