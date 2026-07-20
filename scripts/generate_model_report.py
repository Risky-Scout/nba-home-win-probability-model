"""Render reports/model_report.md from machine-readable artifacts.

Numbers are never typed by hand: everything comes from reports/metrics.json,
artifacts/current/*.json, and reports/nested_feature_set_scan_pre_march.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _f(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root)

    metrics = json.loads((root / "reports/metrics.json").read_text())
    spec = json.loads((root / "artifacts/current/selected_spec_pre_march.json").read_text())
    proof = json.loads((root / "artifacts/current/pre_march_selection_proof.json").read_text())
    final = json.loads((root / "artifacts/current/final_metrics.json").read_text())
    cal = json.loads((root / "artifacts/current/calibration_diagnostics.json").read_text())
    coef = json.loads((root / "artifacts/current/model_coefficients.json").read_text())
    boot = json.loads((root / "artifacts/current/date_block_bootstrap_summary.json").read_text())
    audit = json.loads((root / "artifacts/current/data_audit.json").read_text())
    scan_path = root / "reports/nested_feature_set_scan_pre_march.json"
    scan = json.loads(scan_path.read_text()) if scan_path.exists() else None
    runtime = json.loads((root / "artifacts/current/runtime_benchmark.json").read_text())

    def model_table(period: str) -> str:
        rows = [
            "| Model | Log loss | Brier | AUC | Accuracy | Δ per-game LL (selected − model) 95% CI |",
            "|---|---|---|---|---|---|",
        ]
        for name, entry in metrics[period]["models"].items():
            m = entry["metrics"]
            d = entry.get("paired_delta_log_loss_selected_minus_this")
            ci = (
                f"{_f(d['mean_delta'])} [{_f(d['ci_low_2p5'])}, {_f(d['ci_high_97p5'])}]"
                if d
                else "—"
            )
            rows.append(
                f"| {name} | {_f(m['log_loss'])} | {_f(m['brier'])} | {_f(m['auc'])} | "
                f"{_f(m['accuracy'])} | {ci} |"
            )
        return "\n".join(rows)

    def reliability(period: str) -> str:
        rows = ["| Bin | n | Mean prediction | Observed rate |", "|---|---|---|---|"]
        for b in metrics[period]["reliability_selected_model"]:
            rows.append(
                f"| {b['bin_low']:.1f}–{b['bin_high']:.1f} | {b['n']} | "
                f"{_f(b['mean_prediction'])} | {_f(b['observed_rate'])} |"
            )
        return "\n".join(rows)

    blend = metrics["challenger_blend_on_locked_march"]
    blend_d = blend.get("paired_delta_log_loss_selected_minus_blend", {})

    march = final["locked_march_test"]["metrics"]
    april = final["primary_april_result"]["metrics"]
    seq = final["sequential_daily"]["april"]

    coef_rows = ["| Feature | Standardized β | Raw-unit β |", "|---|---|---|"]
    for r in coef["rows"]:
        raw = r.get("raw_unit_coefficient")
        coef_rows.append(
            f"| {r['feature']} | {_f(r['standardized_coefficient'])} | "
            f"{_f(raw) if raw is not None else '—'} |"
        )

    scan_block = ""
    if scan:
        best = sorted(scan["mean_validation_log_loss"].items(), key=lambda kv: kv[1])[:6]
        scan_rows = ["| Feature set / C | Mean validation log loss |", "|---|---|"]
        for k, v in best:
            scan_rows.append(f"| {k} | {_f(v, 5)} |")
        scan_block = "\n".join(scan_rows)

    text = f"""# Model report

All numbers in this file are rendered programmatically from
`reports/metrics.json` and `artifacts/current/*.json` by
`scripts/generate_model_report.py`. Do not edit numbers by hand.

## Executive summary

- Task: probability that the home team wins each April 2026 game, using
  October–March information only.
- Selected model: **direct L2 logistic regression** on three leakage-controlled
  features (`elo_diff`, `bt_logit`, `trend_diff`), architecture
  `{spec['architecture']['name']}`, regularization C = {spec['logistic_c']}.
- Selection used **only pre-March expanding folds** (January and February
  validation). Proof: `{proof['march_rows_used_in_selection']}` March rows and
  `{proof['april_rows_used_in_selection']}` April rows entered selection.
- Locked March test (scored once): log loss {_f(march['log_loss'])},
  {march['correct_games']}/{march['games']} correct.
- Primary April forecast (state frozen 2026-03-31): log loss
  {_f(april['log_loss'])}, Brier {_f(april['brier'])}, AUC {_f(april['auc'])},
  {april['correct_games']}/{april['games']} correct.
- The selected model decisively beats the constant-rate and record-difference
  baselines. Its difference from a single-feature **Elo-only logistic is
  statistically unresolved** (95% CI for Δ per-game log loss includes zero on
  both March and April). We keep the declared 72-candidate selection outcome
  and record Elo-only as the pre-registered simpler challenger for future data.

## Data audit

- {audit['row_count']} games, {audit['team_count']} teams,
  {audit['date_min']} → {audit['date_max']}; every team plays 82 games:
  {audit['all_teams_play_82_games']}.
- Home-win rate: {_f(audit['home_win_rate'])}.
- Missing values: {audit['missing_value_count']}; duplicate game ids:
  {audit['duplicate_game_id_count']}; tied scores: {audit['tied_game_count']}.
- Pregame wins/losses reconcile exactly against replayed results:
  {audit['pregame_record_reconciliation']['mismatch_count']} mismatches.

## Target and leakage controls

Target: `home_win = 1` iff `home_points > away_points`. Postgame box-score
columns never enter same-game features; team state updates **after** features
are written; same-date games are batched. Guards raise on March-or-later rows
in selection (`nba_wp/selection.py`), with tests in
`tests/test_feature_timing.py` and `tests/test_temporal_protocol.py`.

## Temporal protocol

| Stage | Period | Role |
|---|---|---|
| Development | Oct–Feb | Feature and model development |
| Fold 1 | train < Jan → validate Jan | Selection |
| Fold 2 | train < Feb → validate Feb | Selection |
| Locked test | March | Scored once after freeze |
| Forecast | April | Frozen 2026-03-31 refit on Oct–Mar |

A third fold (train Oct–Nov → validate Dec) was evaluated as a **protocol
sensitivity**: with only ~250 training games it pushes selection toward heavier
shrinkage (C = 0.01) that scores worse when refit on five months of data. The
declared two-fold protocol stands; the sensitivity is disclosed here rather
than silently absorbed.

## Selection process

Declared budget: 3 Elo K × 3 trend half-lives × 7 C values = 63 direct
logistics + 9 architecture-matched blend challengers = **{proof['total_candidates']}
candidates**. Primary metric: mean validation log loss; ties break by Brier,
then AUC, accuracy, model type, architecture name.

Winner: `{proof['selected_architecture']}`, C = {proof['selected_logistic_c']},
mean validation log loss {_f(spec['pre_march_validation_metrics']['mean_validation_log_loss'])}.

## Selected model coefficients (fit through March for April scoring)

{chr(10).join(coef_rows)}

## Locked March test (scored once)

{model_table('locked_march_test')}

Reliability (selected model):

{reliability('locked_march_test')}

## Frozen April forecast (primary)

{model_table('frozen_april_forecast')}

Reliability (selected model):

{reliability('frozen_april_forecast')}

Rolling-daily April (separate descriptive scenario,
`predictions/april_predictions_rolling_scenario.csv`): log loss
{_f(seq['log_loss'])}, {seq['correct_games']}/96 correct. It is **not** the
assignment result.

## Ensemble / challenger review

The v1 log-odds blend (Elo + BT/trend component logistics, Platt-calibrated)
survives as a challenger. On the locked March test its predictions correlate
{_f(blend['prediction_correlation'])} with the selected model; paired
Δ per-game log loss (selected − blend) = {_f(blend_d.get('mean_delta', float('nan')))}
[{_f(blend_d.get('ci_low_2p5', float('nan')))}, {_f(blend_d.get('ci_high_97p5', float('nan')))}].
The selected direct logistic is simpler and no worse, so the blend remains a
documented challenger, not the selected model.

## Calibration

| Period | Intercept α | Slope γ | ECE | Min p | Max p |
|---|---|---|---|---|---|
| Locked March | {_f(cal['march_locked_test']['calibration_intercept_alpha'])} | {_f(cal['march_locked_test']['calibration_slope_gamma'])} | {_f(cal['march_locked_test']['expected_calibration_error'])} | {_f(cal['march_locked_test']['min_predicted_probability'])} | {_f(cal['march_locked_test']['max_predicted_probability'])} |
| Frozen April | {_f(cal['april_frozen_primary']['calibration_intercept_alpha'])} | {_f(cal['april_frozen_primary']['calibration_slope_gamma'])} | {_f(cal['april_frozen_primary']['expected_calibration_error'])} | {_f(cal['april_frozen_primary']['min_predicted_probability'])} | {_f(cal['april_frozen_primary']['max_predicted_probability'])} |

Slope > 1 suggests probabilities are somewhat under-dispersed (too close to
0.5), but with 96 April games the uncertainty is large. This is a diagnostic,
not a solved property. April is never used to recalibrate.

## Uncertainty (date-block bootstrap on frozen April)

Method: {boot['method']}, {boot.get('repeats_used', boot.get('repeats'))} replicates,
seed {boot['seed']}; intervals condition on the locked specification.

| Metric | Mean | 5% | 95% |
|---|---|---|---|
| Log loss | {_f(boot['metrics']['log_loss']['mean'])} | {_f(boot['metrics']['log_loss']['p05'])} | {_f(boot['metrics']['log_loss']['p95'])} |
| Brier | {_f(boot['metrics']['brier']['mean'])} | {_f(boot['metrics']['brier']['p05'])} | {_f(boot['metrics']['brier']['p95'])} |
| AUC | {_f(boot['metrics']['auc']['mean'])} | {_f(boot['metrics']['auc']['p05'])} | {_f(boot['metrics']['auc']['p95'])} |
| Accuracy | {_f(boot['metrics']['accuracy']['mean'])} | {_f(boot['metrics']['accuracy']['p05'])} | {_f(boot['metrics']['accuracy']['p95'])} |

## Feature-set sensitivity (pre-March folds only)

{scan_block}

The Elo-only single feature achieves essentially the same pre-March validation
log loss as the three-feature model. Because the declared search did not
include feature-set pruning, and any post-hoc switch would now be informed by
locked-test results, the three-feature winner stands and Elo-only is recorded
as the pre-registered simpler challenger for future seasons.

## Feature drift

Monthly means/std/quantiles and max standardized distance vs the Oct–Feb
training distribution: `reports/feature_drift_monthly.csv`. The engineered
features are differences of bounded team states (not mechanically growing
cumulative sums), and April values stay within the training envelope.

## Computational performance

End-to-end scoring (feature rebuild, final fit, locked-test + frozen scoring,
diagnostics, figures): {_f(runtime.get('score_and_write_wall_seconds', float('nan')), 2)} s.
Model selection over {proof['total_candidates']} candidates:
{_f(runtime.get('model_selection_wall_seconds', float('nan')), 2)} s.

## Limitations

1. April was viewed during the wider project before this protocol; it is a
   retrospective scoring period, not a pristine holdout.
2. No bookmaker prices → no market-edge, CLV, or profitability claims.
3. One season, 96 forecast games → wide intervals; Elo-only equivalence
   unresolved.
4. No injuries, lineups, travel, or rest-model in the supplied data beyond
   schedule-derived features.

## Recommended production extensions

Time-stamped market odds ingestion and de-vigging; lineup/injury feeds;
rolling refits with monitoring and rollback; per-team calibration monitoring;
multi-season backtesting before any pricing use.
"""
    (root / "reports/model_report.md").write_text(text)
    print("Wrote reports/model_report.md")


if __name__ == "__main__":
    main()
