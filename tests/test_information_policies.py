"""Information-policy invariants for the two nested validation protocols.

Frozen-block: within an outer block frozen at origin O, no outcome inside the
block may change any prediction in the block.

Daily-sequential: a prediction for date t may use results strictly before t, so
mutating date t's own outcome must not change date t's prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_wp.data import load_games
from nba_wp.features import Architecture, build_features
from nba_wp.model import component_probabilities, fit_base_models

ROOT = Path(__file__).resolve().parents[1]


def _architecture() -> Architecture:
    spec = json.loads((ROOT / "artifacts" / "selected_spec.json").read_text())
    return Architecture.from_dict(spec["architecture"])


def _games() -> pd.DataFrame:
    return load_games(ROOT / "data" / "nba-win-probability-data.csv")


def _swap_scores(games: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    mutated = games.copy()
    hp = mutated.loc[mask, "home_points"].to_numpy().copy()
    ap = mutated.loc[mask, "away_points"].to_numpy().copy()
    mutated.loc[mask, "home_points"] = ap
    mutated.loc[mask, "away_points"] = hp
    mutated["home_win"] = (mutated["home_points"] > mutated["away_points"]).astype("int8")
    return mutated


def test_frozen_outer_block_ignores_all_outer_block_outcomes() -> None:
    arch = _architecture()
    games = _games()
    origin = pd.Timestamp("2026-03-15")
    end = origin + pd.Timedelta(days=7)

    base_train = build_features(games, arch)
    base_train = base_train[base_train["game_date"] < origin]
    elo_base = fit_base_models(base_train, arch)

    def block_elo(frame_games: pd.DataFrame) -> np.ndarray:
        feats = build_features(frame_games, arch, freeze_date=origin)
        block = feats[(feats["game_date"] >= origin) & (feats["game_date"] < end)].sort_values("game_id")
        return component_probabilities(elo_base, block)[0]

    baseline = block_elo(games)
    mutated = _swap_scores(games, (games["game_date"] >= origin) & (games["game_date"] < end))
    changed = block_elo(mutated)
    assert baseline.shape == changed.shape and len(baseline) > 0
    assert np.allclose(baseline, changed, atol=1e-12)


def test_daily_sequential_uses_only_strictly_prior_dates() -> None:
    arch = _architecture()
    games = _games()
    t = pd.Timestamp("2026-03-15")

    def day_elo(frame_games: pd.DataFrame) -> np.ndarray:
        feats = build_features(frame_games, arch)
        train = feats[feats["game_date"] < t]
        fold = feats[feats["game_date"] == t].sort_values("game_id")
        base = fit_base_models(train, arch)
        return component_probabilities(base, fold)[0]

    baseline = day_elo(games)
    mutated = _swap_scores(games, games["game_date"] == t)
    changed = day_elo(mutated)
    assert baseline.shape == changed.shape and len(baseline) > 0
    assert np.allclose(baseline, changed, atol=1e-12)
