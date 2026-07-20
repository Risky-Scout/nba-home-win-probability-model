"""CLI smoke test on the synthetic fixture: select -> predict -> schema."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_wp.cli import PREDICTION_COLUMNS, main

FIXTURE = Path("tests/fixtures/synthetic_nba_season.csv")

TINY_CONFIG = """
selection:
  cutoff: "2026-03-01"
  metric: mean_validation_log_loss
  secondary_metrics: [mean_validation_brier]
  descriptive_metrics: [mean_validation_auc]
  folds:
    - {name: fold1_jan, train_end: "2026-01-01", validation_start: "2026-01-01", validation_end: "2026-02-01"}
    - {name: fold2_feb, train_end: "2026-02-01", validation_start: "2026-02-01", validation_end: "2026-03-01"}
search_budget:
  elo_k: [10.0]
  trend_half_life_days: [45.0]
  logistic_c: [0.3, 1.0]
feature_defaults:
  elo_hfa: 65.0
  elo_mov: log
  bt_c: 0.15
  trend_short_games: 5
  elo_model_c: 1.0
  rank_model_c: 0.1
challenger:
  enabled: false
evaluation:
  locked_test: {period_start: "2026-03-01", period_end: "2026-04-01", scored_once: true}
  forecast: {information_cutoff: "2026-03-31", period_start: "2026-04-01"}
  bootstrap: {repeats: 50, seed: 1, block: game_date}
output:
  model_version: "test"
  predictions_dir: predictions
  primary_file: april_predictions.csv
  rolling_scenario_file: april_predictions_rolling_scenario.csv
"""


def test_cli_select_then_predict_smoke(tmp_path) -> None:
    config = tmp_path / "model.yaml"
    config.write_text(TINY_CONFIG)
    artifact_dir = tmp_path / "artifacts"
    output = tmp_path / "predictions" / "april_predictions.csv"

    rc = main(
        [
            "select",
            "--data",
            str(FIXTURE),
            "--config",
            str(config),
            "--artifact-dir",
            str(artifact_dir),
        ]
    )
    assert rc == 0
    proof = json.loads((artifact_dir / "pre_march_selection_proof.json").read_text())
    assert proof["march_rows_used_in_selection"] == 0
    assert proof["april_rows_used_in_selection"] == 0

    rc = main(
        [
            "predict",
            "--data",
            str(FIXTURE),
            "--config",
            str(config),
            "--spec",
            str(artifact_dir / "selected_spec_pre_march.json"),
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    frame = pd.read_csv(output)
    assert list(frame.columns) == PREDICTION_COLUMNS
    p = frame["home_win_probability"].to_numpy()
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    assert np.allclose(
        frame["home_win_probability"] + frame["away_win_probability"], 1.0
    )
