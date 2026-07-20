
from __future__ import annotations

from pathlib import Path

from nba_wp.data import load_games
from nba_wp.reporting import score_and_write
from nba_wp.selection import run_selection

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_nba_season.csv"

TINY = {
    "search_budget": {
        "elo_k": [10.0],
        "trend_half_life_days": [45.0],
        "logistic_c": [0.3, 1.0],
    },
    "feature_defaults": {
        "elo_hfa": 65.0,
        "elo_mov": "log",
        "bt_c": 0.15,
        "trend_short_games": 5,
        "elo_model_c": 1.0,
        "rank_model_c": 0.1,
    },
    "blend_challenger": {"enabled": False},
}
FOLDS = {
    "folds": [
        {
            "name": "fold1_jan",
            "train_end": "2026-01-01",
            "validation_start": "2026-01-01",
            "validation_end": "2026-02-01",
        },
        {
            "name": "fold2_feb",
            "train_end": "2026-02-01",
            "validation_start": "2026-02-01",
            "validation_end": "2026-03-01",
        },
    ]
}


def test_end_to_end_synthetic(tmp_path):
    games = load_games(FIXTURE)
    selection_games = games[games["game_date"] < "2026-03-01"].copy()
    selected, fold_table, selection_table = run_selection(
        selection_games, TINY, FOLDS
    )
    assert selected["march_rows_used_in_selection"] == 0
    assert selected["april_rows_used_in_selection"] == 0
    assert not fold_table.empty
    assert not selection_table.empty

    metrics = score_and_write(
        games,
        selected,
        tmp_path / "outputs",
        tmp_path / "artifacts",
        tmp_path / "figures",
    )
    assert "primary_april_result" in metrics
    assert (tmp_path / "outputs" / "april_predictions_frozen_snapshot.csv").exists()
