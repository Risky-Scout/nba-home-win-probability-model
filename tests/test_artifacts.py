
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_wp.model import evaluate


def test_selection_proof_excludes_march_and_april() -> None:
    path = Path("artifacts/pre_march_selection_proof.json")
    if not path.exists():
        path = Path("artifacts/selection_proof.json")
    if not path.exists():
        pytest.skip("Selection artifacts not generated yet.")
    proof = json.loads(path.read_text())
    assert proof.get("april_rows_used_in_selection", proof.get("april_rows_loaded")) == 0
    assert proof.get("march_rows_used_in_selection", 0) == 0
    assert proof.get("selection_data_end", proof.get("selection_input_max_date")) <= "2026-02-28"


def test_saved_metrics_recompute_from_predictions() -> None:
    metrics_path = Path("artifacts/final_metrics.json")
    frozen_path = Path("outputs/april_predictions_frozen_snapshot.csv")
    march_path = Path("outputs/march_predictions.csv")
    if not (metrics_path.exists() and frozen_path.exists() and march_path.exists()):
        pytest.skip("Scoring artifacts not generated yet.")

    metrics = json.loads(metrics_path.read_text())
    march = pd.read_csv(march_path)
    frozen = pd.read_csv(frozen_path)
    assert len(march) == 239
    assert len(frozen) == 96

    observed_march = evaluate(march["home_win"], march["home_win_probability"].to_numpy())
    observed_april = evaluate(
        frozen["home_win"], frozen["home_win_probability"].to_numpy()
    )
    for key in ["log_loss", "brier", "auc", "accuracy"]:
        assert np.isclose(
            observed_march[key],
            metrics["locked_march_test"]["metrics"][key],
            atol=1e-12,
            rtol=0,
        )
        assert np.isclose(
            observed_april[key],
            metrics["primary_april_result"]["metrics"][key],
            atol=1e-12,
            rtol=0,
        )
