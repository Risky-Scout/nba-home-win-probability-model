
from __future__ import annotations

from nba_wp.features import Architecture, build_features


def test_frozen_april_does_not_use_april_outcomes(real_games):
    architecture = Architecture(
        name="probe",
        elo_k=10.0,
        elo_hfa=65.0,
        elo_mov="log",
        bt_c=0.15,
        trend_half_life_days=45.0,
        trend_short_games=10,
        elo_model_c=1.0,
        rank_model_c=0.1,
    )
    frozen = build_features(real_games, architecture, freeze_date="2026-04-01")
    april = frozen[frozen["game_date"] >= "2026-04-01"].copy()
    assert not april.empty
    assert (april["state_policy"] == "frozen_snapshot").all()
    assert (april["performance_cutoff"] == "2026-03-31").all()
