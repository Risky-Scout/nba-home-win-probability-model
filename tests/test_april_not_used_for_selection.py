
from __future__ import annotations

from nba_wp.selection import assert_pre_march_selection_frame, run_selection

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


def test_selection_frame_excludes_april(real_games):
    selection_games = real_games[real_games["game_date"] < "2026-03-01"].copy()
    assert (selection_games["game_date"] >= "2026-04-01").sum() == 0
    assert_pre_march_selection_frame(selection_games)
    selected, _, _ = run_selection(selection_games, TINY, FOLDS)
    assert selected["april_rows_used_in_selection"] == 0
