
from __future__ import annotations

import numpy as np

from nba_wp.data import audit_games
from nba_wp.features import Architecture, build_features


ARCHITECTURE = Architecture(
    name="balanced",
    elo_k=10.0,
    elo_hfa=65.0,
    elo_mov="log",
    bt_c=0.15,
    trend_half_life_days=45.0,
    trend_short_games=10,
    elo_model_c=100.0,
    rank_model_c=0.1,
)


def test_real_data_audit(real_games) -> None:
    audit = audit_games(real_games)
    assert audit["row_count"] == 1230
    assert audit["team_count"] == 30
    assert audit["pregame_record_reconciliation"]["mismatch_count"] == 0


def test_april_mutation_cannot_change_march_features(real_games) -> None:
    selection = real_games[real_games["game_date"] < "2026-04-01"].copy()
    baseline = build_features(selection, ARCHITECTURE)

    mutated = real_games.copy()
    april = mutated["game_date"] >= "2026-04-01"
    mutated.loc[april, "home_points"] += 500
    mutated.loc[april, "away_points"] = 1
    selection_mutated = mutated[mutated["game_date"] < "2026-04-01"].copy()
    rebuilt = build_features(selection_mutated, ARCHITECTURE)

    columns = ["elo_diff", "bt_logit", "trend_diff", "record_logit_diff"]
    assert np.allclose(
        baseline[columns].to_numpy(dtype=float),
        rebuilt[columns].to_numpy(dtype=float),
    )
