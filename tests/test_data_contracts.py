"""Loader contracts: duplicates, invalid records, ordering invariance."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_wp.data import DataValidationError, load_games

FIXTURE = Path("tests/fixtures/synthetic_nba_season.csv")


def _write(tmp_path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "games.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture()
def fixture_frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, dtype={"game_id": "string"})


def test_duplicate_game_id_raises(tmp_path, fixture_frame) -> None:
    frame = pd.concat([fixture_frame, fixture_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="Duplicate game IDs"):
        load_games(_write(tmp_path, frame))


def test_missing_values_raise(tmp_path, fixture_frame) -> None:
    frame = fixture_frame.copy()
    frame.loc[0, "home_points"] = np.nan
    with pytest.raises(DataValidationError, match="Missing values"):
        load_games(_write(tmp_path, frame))


def test_tied_score_raises(tmp_path, fixture_frame) -> None:
    frame = fixture_frame.copy()
    frame.loc[0, "home_points"] = frame.loc[0, "away_points"]
    with pytest.raises(DataValidationError, match="Tied"):
        load_games(_write(tmp_path, frame))


def test_self_play_raises(tmp_path, fixture_frame) -> None:
    frame = fixture_frame.copy()
    frame.loc[0, "away"] = frame.loc[0, "home"]
    with pytest.raises(DataValidationError, match="both home and away"):
        load_games(_write(tmp_path, frame))


def test_row_order_does_not_change_loaded_output(tmp_path, fixture_frame) -> None:
    shuffled = fixture_frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a = load_games(_write(tmp_path, fixture_frame))
    b = load_games(_write(tmp_path, shuffled))
    pd.testing.assert_frame_equal(a, b)


def test_row_order_does_not_change_predictions(tmp_path, fixture_frame) -> None:
    """Shuffling raw CSV rows leaves model probabilities untouched."""
    from nba_wp.features import Architecture, build_features
    from nba_wp.model import fit_direct_logistic, predict_direct_logistic

    architecture = Architecture(
        name="test",
        elo_k=10.0,
        elo_hfa=65.0,
        elo_mov="log",
        bt_c=0.15,
        trend_half_life_days=20.0,
        trend_short_games=5,
        elo_model_c=1.0,
        rank_model_c=0.1,
    )

    def probs(frame: pd.DataFrame) -> np.ndarray:
        games = load_games(_write(tmp_path, frame))
        feats = build_features(games, architecture)
        train = feats[feats["game_date"] < "2026-03-01"]
        score = feats[feats["game_date"] >= "2026-03-01"]
        model = fit_direct_logistic(train, 0.3)
        return predict_direct_logistic(model, score)

    base = probs(fixture_frame)
    shuffled = fixture_frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
    assert np.allclose(base, probs(shuffled), atol=0, rtol=0)
