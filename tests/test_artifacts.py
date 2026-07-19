
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_wp.model import evaluate


def test_selection_proof_excludes_april() -> None:
    proof = json.loads(Path("artifacts/selection_proof.json").read_text())
    assert proof["april_rows_loaded"] == 0
    assert proof["selection_input_max_date"] == "2026-03-31"


def test_saved_metrics_recompute_from_predictions() -> None:
    metrics = json.loads(Path("artifacts/final_metrics.json").read_text())
    for period, expected_count in [("march", 239), ("april", 96)]:
        predictions = pd.read_csv(f"outputs/{period}_predictions.csv")
        assert len(predictions) == expected_count
        observed = evaluate(
            predictions["home_win"],
            predictions["home_win_probability"].to_numpy(),
        )
        for key in ["log_loss", "brier", "auc", "accuracy"]:
            assert np.isclose(
                observed[key],
                metrics["sequential_daily"][period][key],
                atol=1e-12,
                rtol=0,
            )
