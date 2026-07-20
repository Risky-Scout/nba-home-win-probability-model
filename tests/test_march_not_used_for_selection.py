
from __future__ import annotations

from nba_wp.selection import run_selection

TINY = {
    "search_budget": {
        "elo_k": [10.0],
        "trend_half_life_days": [45.0],
        "logistic_c": [0.3],
    },
    "feature_defaults": {
        "elo_hfa": 65.0,
        "elo_mov": "log",
        "bt_c": 0.15,
        "trend_short_games": 10,
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


def test_selected_spec_records_zero_march_rows(real_games):
    selection_games = real_games[real_games["game_date"] < "2026-03-01"].copy()
    selected, _, table = run_selection(selection_games, TINY, FOLDS)
    assert selected["march_rows_used_in_selection"] == 0
    assert selected["selection_data_end"] <= "2026-02-28"
    assert not table.empty
    assert selected["selection_metric"] == "mean_validation_log_loss"
