
from __future__ import annotations

import pandas as pd
import pytest

from nba_wp.selection import assert_pre_march_selection_frame, run_selection


def test_assert_pre_march_rejects_march_rows():
    frame = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2026-02-28", "2026-03-01"]),
            "home_win": [1, 0],
        }
    )
    with pytest.raises(ValueError, match="March or later"):
        assert_pre_march_selection_frame(frame)


def test_assert_pre_march_accepts_february_end():
    frame = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2026-02-27", "2026-02-28"]),
            "home_win": [1, 0],
        }
    )
    assert_pre_march_selection_frame(frame)


def test_run_selection_rejects_march_frame(real_games):
    with pytest.raises(ValueError, match="March or later"):
        run_selection(
            real_games[real_games["game_date"] < "2026-04-01"].copy(),
            {
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
            },
            {
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
            },
        )
