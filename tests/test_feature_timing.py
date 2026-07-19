
from __future__ import annotations

import numpy as np
import pandas as pd

from nba_wp.features import Architecture, build_features


ARCHITECTURE = Architecture(
    name="test",
    elo_k=10.0,
    elo_hfa=65.0,
    elo_mov="log",
    bt_c=0.15,
    trend_half_life_days=45.0,
    trend_short_games=3,
    elo_model_c=1.0,
    rank_model_c=0.1,
)


def synthetic_games() -> pd.DataFrame:
    teams = ["AAA", "BBB", "CCC", "DDD"]
    rows = []
    game_id = 1
    records = {team: [0, 0] for team in teams}
    schedule = [
        ("2025-10-21", "AAA", "BBB", 110, 100),
        ("2025-10-21", "CCC", "DDD", 95, 105),
        ("2025-10-23", "AAA", "CCC", 108, 101),
        ("2025-10-23", "BBB", "DDD", 99, 102),
        ("2025-10-25", "DDD", "AAA", 104, 98),
        ("2025-10-25", "CCC", "BBB", 107, 103),
        ("2025-10-27", "AAA", "BBB", 111, 109),
        ("2025-10-27", "CCC", "DDD", 100, 96),
    ]
    for date, home, away, home_points, away_points in schedule:
        rows.append(
            {
                "game_id": str(game_id).zfill(10),
                "game_date": pd.Timestamp(date),
                "away": away,
                "away_wins": records[away][0],
                "away_losses": records[away][1],
                "away_points": away_points,
                "away_turnovers": 12,
                "away_fouls": 19,
                "away_rebounds": 44,
                "home": home,
                "home_wins": records[home][0],
                "home_losses": records[home][1],
                "home_points": home_points,
                "home_turnovers": 11,
                "home_fouls": 18,
                "home_rebounds": 46,
                "home_win": int(home_points > away_points),
            }
        )
        if home_points > away_points:
            records[home][0] += 1
            records[away][1] += 1
        else:
            records[away][0] += 1
            records[home][1] += 1
        game_id += 1
    return pd.DataFrame(rows)


def test_current_game_postgame_values_do_not_change_current_features() -> None:
    games = synthetic_games()
    baseline = build_features(games, ARCHITECTURE)
    mutated = games.copy()
    target = 4
    mutated.loc[target, "home_points"] += 80
    mutated.loc[target, "away_turnovers"] += 30
    rebuilt = build_features(mutated, ARCHITECTURE)
    columns = ["elo_diff", "bt_logit", "trend_diff", "record_logit_diff"]
    assert np.allclose(
        baseline.loc[target, columns].to_numpy(dtype=float),
        rebuilt.loc[target, columns].to_numpy(dtype=float),
    )


def test_same_day_games_are_batched() -> None:
    games = synthetic_games()
    baseline = build_features(games, ARCHITECTURE)
    mutated = games.copy()
    mutated.loc[0, "home_points"] = 10
    mutated.loc[0, "away_points"] = 200
    mutated.loc[0, "home_win"] = 0
    rebuilt = build_features(mutated, ARCHITECTURE)
    same_day_other_game = 1
    columns = ["elo_diff", "bt_logit", "trend_diff", "record_logit_diff"]
    assert np.allclose(
        baseline.loc[same_day_other_game, columns].to_numpy(dtype=float),
        rebuilt.loc[same_day_other_game, columns].to_numpy(dtype=float),
    )


def test_frozen_snapshot_stops_performance_updates() -> None:
    games = synthetic_games()
    frozen = build_features(
        games,
        ARCHITECTURE,
        freeze_date="2025-10-25",
    )
    mutated = games.copy()
    mutated.loc[4:, "home_points"] += 100
    mutated.loc[4:, "home_win"] = 1
    rebuilt = build_features(
        mutated,
        ARCHITECTURE,
        freeze_date="2025-10-25",
    )
    target_rows = frozen["game_date"] >= "2025-10-25"
    columns = ["elo_diff", "bt_logit", "trend_diff", "record_logit_diff"]
    assert np.allclose(
        frozen.loc[target_rows, columns].to_numpy(dtype=float),
        rebuilt.loc[target_rows, columns].to_numpy(dtype=float),
    )
