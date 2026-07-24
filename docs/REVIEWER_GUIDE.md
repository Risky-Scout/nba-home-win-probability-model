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

`docs/METHODOLOGY.md` — the **deployed champion is Elo-only**: a logistic map
on the standardized margin-of-victory Elo rating differential (MOV multiplier
on the winner − loser rating difference), `p = sigmoid(c + w·z(elo_diff))`, fit
on all games through March 31 with the April state frozen at March 31. It has
**no stacker and no temperature floor**. The deployed Elo coefficients are in
`artifacts/selected_spec.json` under `elo_model`. A regularized Bradley–Terry +
recent-trend logistic-stacked **blend** was implemented and validated but
**rejected** — it lives under the `challenger` block (`status: "rejected"`),
where the temperature floor / convex-stacker note belongs.

## 5. Selection provenance

- Declared candidates: `configs/architecture_candidates.json` (all five)
- All five scored: `artifacts/march_architecture_results.csv`
- Rule: the deployed Elo architecture is chosen by **aggregate frozen-policy
  rolling OOS log loss with a one-standard-error stability rule** (simplest/lowest-K
  within the band; `conservative` wins outright and a single March split agrees).
  See `architecture_selection` in `selected_spec.json` and
  `tests/test_architecture_selection.py`. The MOV offset (2.2) and cold-start
  warmup (off) are likewise data-driven (`mov_offset_selection`,
  `cold_start_selection`). Champion family is Elo-only because the nested audit
  rejects every challenger (`selected_spec.json`: `model_family = "elo_only"`,
  `champion = "elo_only"`)
- April exclusion proof: `artifacts/selection_proof.json`
  (`selection_input_max_date = 2026-03-31`, zero April rows; the selection
  function raises otherwise)
- Pinned by `tests/test_champion_promotion.py` and
  `tests/test_information_policies.py`

## 6. Baselines and feature evidence

`artifacts/feature_group_ablation.csv` — constant prior through record-only,
Elo-only, BT+trend, a rich linear challenger, and the blend, all on the same
protocol. The decisive out-of-sample comparison is the **nested rolling-origin
audit** (`scripts/nested_validation.py`): pooled Elo-only 0.529 log loss /
0.176 Brier versus blend 0.548 / 0.183, with block-bootstrap blend − Elo-only
CIs entirely above zero on both metrics (0 of 4,000 replicates favored the
blend). Two further challengers are also rejected on the same surface: a
cross-fitted **calibrated Elo** (0.548 / 0.183; over-corrects, β ≈ 1.79) and a
**schedule Elo** (Elo + rest + back-to-back, 0.531 / 0.177; slightly worse on
both scores). See `artifacts/nested_frozen_block_summary.json` and
`artifacts/nested_daily_sequential_summary.json` (`calibration_challenger`,
`schedule_challenger`).

## 7. Results

`artifacts/final_metrics.json` — the **primary April holdout** (Elo-only
champion, frozen: LL 0.464369, Brier 0.149770, AUC 0.866847, accuracy 78.125%),
the March one-step-ahead surface (Elo-only, in-sample for selection: LL
0.506590), and an optional April sequential backtest (live-update simulation,
LL 0.464648). For reference the rejected blend scores LL 0.468725 / Brier
0.150465 on the same frozen April window (worse on both proper scores).
Reliability diagrams in `figures/nested_frozen_block_reliability.png` and
`figures/nested_daily_sequential_reliability.png`. Fair decimal odds in the
prediction CSVs are zero-margin transforms of the probabilities.

## 8. Known limitations

`docs/LIMITATIONS_AND_ROADMAP.md` — March double-use, prior April exposure at
the project level, no injury/lineup/travel data, single season, and the
production extensions that would address each.

## Human-readable results export

Run `python -m scripts.export_results_spreadsheet` to write an `.xlsx`/CSV with
the frozen April predictions (primary), March predictions, the April sequential
backtest, and a summary sheet. A pre-rendered
`outputs/april_predictions_readable.csv` is also committed. Every value
reconciles to `artifacts/selected_spec.json` and the committed prediction CSVs.
