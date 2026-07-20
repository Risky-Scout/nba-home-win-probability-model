"""Prediction-output contracts: schema, bounds, odds, determinism."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_wp.cli import PREDICTION_COLUMNS

PRIMARY = Path("predictions/april_predictions.csv")
ROLLING = Path("predictions/april_predictions_rolling_scenario.csv")


@pytest.fixture(scope="module")
def primary() -> pd.DataFrame:
    if not PRIMARY.exists():
        pytest.skip("Run `python -m nba_wp.cli predict` first.")
    return pd.read_csv(PRIMARY, dtype={"game_id": "string"})


def test_schema_exact(primary: pd.DataFrame) -> None:
    assert list(primary.columns) == PREDICTION_COLUMNS


def test_probabilities_bounded_and_finite(primary: pd.DataFrame) -> None:
    p = primary["home_win_probability"]
    q = primary["away_win_probability"]
    assert np.isfinite(p).all() and np.isfinite(q).all()
    assert ((p > 0) & (p < 1)).all()
    assert ((q > 0) & (q < 1)).all()


def test_home_away_probabilities_sum_to_one(primary: pd.DataFrame) -> None:
    total = primary["home_win_probability"] + primary["away_win_probability"]
    assert np.allclose(total, 1.0, atol=1e-12)


def test_fair_decimal_odds_are_reciprocal(primary: pd.DataFrame) -> None:
    assert np.allclose(
        primary["home_fair_decimal_odds"],
        1.0 / primary["home_win_probability"],
        rtol=0,
        atol=1e-9,
    )
    assert np.allclose(
        primary["away_fair_decimal_odds"],
        1.0 / primary["away_win_probability"],
        rtol=0,
        atol=1e-9,
    )


def test_primary_is_frozen_cutoff_only(primary: pd.DataFrame) -> None:
    assert (primary["information_cutoff"] == "2026-03-31").all()
    assert (pd.to_datetime(primary["game_date"]) >= "2026-04-01").all()
    assert len(primary) == 96


def test_rolling_scenario_is_separate_file() -> None:
    if not ROLLING.exists():
        pytest.skip("Rolling scenario not generated.")
    rolling = pd.read_csv(ROLLING)
    assert (rolling["information_cutoff"] == "rolling_daily_scenario").all()


def test_frozen_predictions_are_deterministic(real_games) -> None:
    """Re-fitting the locked spec reproduces the committed probabilities."""
    from nba_wp.features import Architecture, build_features
    from nba_wp.model import predict_from_spec

    spec_path = Path("artifacts/current/selected_spec_pre_march.json")
    if not (spec_path.exists() and PRIMARY.exists()):
        pytest.skip("Artifacts missing.")
    spec = json.loads(spec_path.read_text())
    architecture = Architecture.from_dict(spec["architecture"])
    frozen = build_features(real_games, architecture, freeze_date="2026-04-01")
    train = frozen[frozen["game_date"] < "2026-04-01"]
    april = frozen[frozen["game_date"] >= "2026-04-01"].copy()
    probability, _, _, _ = predict_from_spec(spec, train, april)
    saved = pd.read_csv(PRIMARY, dtype={"game_id": "string"}).sort_values("game_id")
    april = april.assign(p=probability).sort_values("game_id")
    assert np.allclose(
        saved["home_win_probability"].to_numpy(),
        april["p"].to_numpy(),
        atol=1e-10,
        rtol=0,
    )


def test_changing_april_outcomes_does_not_change_frozen_probabilities(real_games) -> None:
    """Flip April winners; frozen April probabilities must be identical."""
    from nba_wp.features import Architecture, build_features
    from nba_wp.model import predict_from_spec

    spec_path = Path("artifacts/current/selected_spec_pre_march.json")
    if not spec_path.exists():
        pytest.skip("Spec missing.")
    spec = json.loads(spec_path.read_text())
    architecture = Architecture.from_dict(spec["architecture"])

    def frozen_probs(games: pd.DataFrame) -> np.ndarray:
        frozen = build_features(games, architecture, freeze_date="2026-04-01")
        train = frozen[frozen["game_date"] < "2026-04-01"]
        april = frozen[frozen["game_date"] >= "2026-04-01"]
        p, _, _, _ = predict_from_spec(spec, train, april)
        return np.asarray(p)

    base = frozen_probs(real_games)

    flipped = real_games.copy()
    mask = flipped["game_date"] >= "2026-04-01"
    # Swap home/away points so every April outcome flips.
    flipped.loc[mask, ["home_points", "away_points"]] = flipped.loc[
        mask, ["away_points", "home_points"]
    ].to_numpy()
    flipped["home_win"] = (flipped["home_points"] > flipped["away_points"]).astype("int8")
    altered = frozen_probs(flipped)
    assert np.allclose(base, altered, atol=1e-12, rtol=0)


def test_saved_model_reproduces_known_prediction(tmp_path) -> None:
    """A serialized fitted model reproduces its own predictions after reload."""
    import joblib

    from nba_wp.model import fit_direct_logistic, predict_direct_logistic

    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "elo_diff": rng.normal(0, 0.3, n),
            "bt_logit": rng.normal(0, 0.5, n),
            "trend_diff": rng.normal(0, 3.0, n),
        }
    )
    frame["home_win"] = (
        (2.0 * frame["elo_diff"] + rng.normal(0, 1, n)) > 0
    ).astype(int)
    model = fit_direct_logistic(frame, 0.1)
    expected = predict_direct_logistic(model, frame)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    observed = predict_direct_logistic(reloaded, frame)
    assert np.allclose(expected, observed, atol=0, rtol=0)
    # Training and prediction share one schema.
    assert reloaded.feature_names == ["elo_diff", "bt_logit", "trend_diff"]
