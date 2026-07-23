from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_wp.data import load_games
from nba_wp.features import Architecture, build_features
from nba_wp.model import (
    apply_logit_stacker,
    component_probabilities,
    fit_base_models,
    fit_logit_stacker,
)


def test_primary_april_probabilities_ignore_all_april_results() -> None:
    root = Path(__file__).resolve().parents[1]
    games = load_games(root / "data" / "nba-win-probability-data.csv")
    architecture = Architecture.from_dict(
        json.loads((root / "artifacts" / "selected_spec.json").read_text())[
            "architecture"
        ]
    )

    baseline_features = build_features(
        games, architecture, freeze_date="2026-04-01"
    )
    mutated = games.copy()
    april_mask = mutated["game_date"] >= "2026-04-01"
    mutated.loc[april_mask, "home_points"] = (
        mutated.loc[april_mask, "home_points"] + 40
    )
    mutated.loc[april_mask, "away_points"] = (
        mutated.loc[april_mask, "away_points"] + 5
    )
    mutated.loc[april_mask, "home_win"] = 1 - mutated.loc[april_mask, "home_win"]
    mutated_features = build_features(
        mutated, architecture, freeze_date="2026-04-01"
    )

    april_base = baseline_features[
        baseline_features["game_date"] >= "2026-04-01"
    ].sort_values("game_id")
    april_mut = mutated_features[
        mutated_features["game_date"] >= "2026-04-01"
    ].sort_values("game_id")
    feature_cols = ["elo_diff", "bt_logit", "trend_diff"]
    assert np.allclose(
        april_base[feature_cols].to_numpy(dtype=float),
        april_mut[feature_cols].to_numpy(dtype=float),
        atol=1e-12,
    )

    train = baseline_features[baseline_features["game_date"] < "2026-03-01"]
    march = baseline_features[
        (baseline_features["game_date"] >= "2026-03-01")
        & (baseline_features["game_date"] < "2026-04-01")
    ]
    models = fit_base_models(train, architecture)
    pE, pR = component_probabilities(models, march)
    stacker = fit_logit_stacker(
        march["home_win"].to_numpy(dtype=int), pE, pR, min_temperature=1.0
    )
    p_base = apply_logit_stacker(
        stacker, *component_probabilities(models, april_base)
    )
    p_mut = apply_logit_stacker(
        stacker, *component_probabilities(models, april_mut)
    )
    assert np.allclose(p_base, p_mut, atol=1e-12)
