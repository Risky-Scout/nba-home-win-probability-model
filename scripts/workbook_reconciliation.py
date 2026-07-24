"""Emit a machine-readable reconciliation between the committed artifacts and the
deployed Elo-only champion, so the workbook's "reconciles to ~0" claim is
auditable from one command instead of by eye.

openpyxl does not evaluate Excel formulas headlessly, so rather than read the
workbook's computed cells we independently recompute the champion April prices
from ``selected_spec.json`` (both the standardized and the raw-intercept closed
forms the workbook uses) and compare to ``outputs/april_predictions.csv``. We
also check the coefficient table and the reported metrics, and hash the inputs.

Usage:
  python -m scripts.workbook_reconciliation
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_wp.data import sha256_file
from nba_wp.model import evaluate

ROOT = Path(__file__).resolve().parents[1]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    spec = json.loads((ROOT / "artifacts" / "selected_spec.json").read_text())
    elo = spec["elo_model"]
    april = pd.read_csv(ROOT / "outputs" / "april_predictions.csv", dtype={"game_id": "string"})
    metrics = json.loads((ROOT / "artifacts" / "final_metrics.json").read_text())
    coef = pd.read_csv(ROOT / "artifacts" / "coefficient_table.csv")

    x = april["elo_diff"].to_numpy()
    deployed = april["home_win_probability"].to_numpy()

    p_std = _sigmoid(elo["standardized_coefficient"] * (x - elo["training_mean"]) / elo["training_scale"] + elo["intercept"])
    p_raw = _sigmoid(elo["raw_intercept"] + elo["raw_unit_coefficient"] * x)

    max_diff_std = float(np.max(np.abs(p_std - deployed)))
    max_diff_raw = float(np.max(np.abs(p_raw - deployed)))

    elo_std_row = coef[(coef["component"] == "elo") & (coef["feature"] == "elo_diff")].iloc[0]
    coef_diff = float(abs(elo_std_row["standardized_coefficient"] - elo["standardized_coefficient"]))

    recomputed = evaluate(april["home_win"].to_numpy(dtype=int), deployed)
    reported = metrics["primary_holdout"]["april"]
    metric_diffs = {k: float(abs(recomputed[k] - reported[k])) for k in ["log_loss", "brier", "auc", "accuracy"]}
    max_metric_diff = max(metric_diffs.values())

    tol = 1e-9
    checks = {
        "standardized_form_reconstructs_prices": max_diff_std < tol,
        "raw_form_reconstructs_prices": max_diff_raw < tol,
        "coefficient_table_matches_spec": coef_diff < tol,
        "reported_metrics_match_recompute": max_metric_diff < 1e-8,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "max_abs_probability_diff_standardized_form": max_diff_std,
        "max_abs_probability_diff_raw_form": max_diff_raw,
        "max_abs_coefficient_diff": coef_diff,
        "metric_abs_diffs": metric_diffs,
        "max_abs_metric_diff": max_metric_diff,
        "hashes": {
            "workbook_sha256": sha256_file(ROOT / "NBA_Model_Fully_Formulated.xlsx"),
            "selected_spec_sha256": sha256_file(ROOT / "artifacts" / "selected_spec.json"),
            "april_predictions_sha256": sha256_file(ROOT / "outputs" / "april_predictions.csv"),
        },
        "note": (
            "Champion is Elo-only. Both closed forms in selected_spec.json reproduce the "
            "deployed April prices to < 1e-9; the workbook's sheet-10 formula uses the "
            "standardized form. openpyxl cannot evaluate formulas headlessly, so this "
            "reconciles artifacts and spec, not the workbook's own recalculated cells."
        ),
    }
    out = ROOT / "artifacts" / "workbook_reconciliation.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], **{k: report[k] for k in
          ["max_abs_probability_diff_standardized_form", "max_abs_probability_diff_raw_form",
           "max_abs_coefficient_diff", "max_abs_metric_diff"]}}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit("Workbook reconciliation FAILED")


if __name__ == "__main__":
    main()
