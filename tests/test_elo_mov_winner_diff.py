from __future__ import annotations

import numpy as np
import pandas as pd

from nba_wp.features import Architecture, _elo_multiplier, build_features


def test_mov_multiplier_is_larger_for_upset() -> None:
    favorite_win = _elo_multiplier(
        margin=10,
        rating_difference_without_hfa=200,
        mode="log",
    )
    underdog_win = _elo_multiplier(
        margin=10,
        rating_difference_without_hfa=-200,
        mode="log",
    )
    assert underdog_win > favorite_win


def test_elo_update_is_team_swap_symmetric_with_zero_hfa() -> None:
    """With HFA=0, swapping home/away and flipping the outcome must match."""
    architecture = Architecture(
        name="sym",
        elo_k=10.0,
        elo_hfa=0.0,
        elo_mov="log",
        bt_c=0.15,
        trend_half_life_days=45.0,
        trend_short_games=10,
        elo_model_c=100.0,
        rank_model_c=0.1,
    )

    def two_game_frame(home: str, away: str, home_pts: int, away_pts: int) -> pd.DataFrame:
        # Warm-up game so both teams exist in history, then the target game.
        return pd.DataFrame(
            [
                {
                    "game_id": "1",
                    "game_date": pd.Timestamp("2025-10-21"),
                    "away": "ZZZ",
                    "home": "YYY",
                    "away_wins": 0,
                    "away_losses": 0,
                    "home_wins": 0,
                    "home_losses": 0,
                    "away_points": 100,
                    "home_points": 100,
                    "away_turnovers": 12,
                    "home_turnovers": 11,
                    "away_fouls": 18,
                    "home_fouls": 18,
                    "away_rebounds": 40,
                    "home_rebounds": 40,
                    "home_win": 0,
                },
                {
                    "game_id": "2",
                    "game_date": pd.Timestamp("2025-10-23"),
                    "away": away,
                    "home": home,
                    "away_wins": 0,
                    "away_losses": 0,
                    "home_wins": 0,
                    "home_losses": 0,
                    "away_points": away_pts,
                    "home_points": home_pts,
                    "away_turnovers": 12,
                    "home_turnovers": 11,
                    "away_fouls": 18,
                    "home_fouls": 18,
                    "away_rebounds": 40,
                    "home_rebounds": 40,
                    "home_win": int(home_pts > away_pts),
                },
            ]
        )

    # AAA beats BBB by 12 at home.
    forward = build_features(two_game_frame("AAA", "BBB", 112, 100), architecture)
    # Swap sides: BBB hosts AAA, AAA still wins by 12 (away win).
    swapped = build_features(two_game_frame("BBB", "AAA", 100, 112), architecture)

    # After the target game, Elo diffs for a neutral follow-up should match
    # team strengths up to labeling. Compare stored elo_diff on a third date
    # by rebuilding with an extra probe game that does not update before read.
    def with_probe(base_home: str, base_away: str, home_pts: int, away_pts: int) -> pd.DataFrame:
        frame = two_game_frame(base_home, base_away, home_pts, away_pts)
        probe = {
            "game_id": "3",
            "game_date": pd.Timestamp("2025-10-25"),
            "away": "CCC",
            "home": "AAA",
            "away_wins": 0,
            "away_losses": 0,
            "home_wins": 0,
            "home_losses": 0,
            "away_points": 100,
            "home_points": 100,
            "away_turnovers": 12,
            "home_turnovers": 11,
            "away_fouls": 18,
            "home_fouls": 18,
            "away_rebounds": 40,
            "home_rebounds": 40,
            "home_win": 0,
        }
        return pd.concat([frame, pd.DataFrame([probe])], ignore_index=True)

    fwd = build_features(with_probe("AAA", "BBB", 112, 100), architecture)
    swp = build_features(with_probe("BBB", "AAA", 100, 112), architecture)
    fwd_elo = float(fwd.loc[fwd["game_id"] == "3", "elo_diff"].iloc[0])
    swp_elo = float(swp.loc[swp["game_id"] == "3", "elo_diff"].iloc[0])
    # AAA should have the same strength advantage vs CCC in both histories.
    assert np.isclose(fwd_elo, swp_elo, atol=1e-9)
