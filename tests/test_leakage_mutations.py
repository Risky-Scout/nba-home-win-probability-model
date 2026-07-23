"""Leakage mutation battery.

These tests treat the feature builder as a black box and mutate the raw game
table to prove two complementary properties:

* Negative controls: information that must NOT influence a prediction (future
  outcomes, target-month box scores, input row order) leaves the relevant
  features bit-for-bit unchanged.
* Positive controls: information that legitimately SHOULD influence a prediction
  (a past outcome that feeds pregame state) provably changes the features.

The positive controls are the important half: they prove the invariance
assertions are not vacuously true (i.e. the harness can actually detect a
change), so a real look-ahead regression would fail these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_wp.data import load_games
from nba_wp.features import Architecture, build_features

ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLS = ["elo_diff", "bt_logit", "trend_diff"]


def _architecture() -> Architecture:
    spec = json.loads((ROOT / "artifacts" / "selected_spec.json").read_text())
    return Architecture.from_dict(spec["architecture"])


def _games() -> pd.DataFrame:
    return load_games(ROOT / "data" / "nba-win-probability-data.csv")


def _flip_outcomes(games: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Swap the score of every masked game so the winner flips.

    Scores are swapped (not perturbed) so the mutated rows remain valid NBA
    box scores with no ties, and ``home_win`` is recomputed from points the
    same way :func:`load_games` derives it.
    """
    mutated = games.copy()
    home = mutated.loc[mask, "home_points"].to_numpy().copy()
    away = mutated.loc[mask, "away_points"].to_numpy().copy()
    mutated.loc[mask, "home_points"] = away
    mutated.loc[mask, "away_points"] = home
    mutated["home_win"] = (
        mutated["home_points"] > mutated["away_points"]
    ).astype("int8")
    return mutated


def _features_on_or_after(frame: pd.DataFrame, date: str) -> pd.DataFrame:
    return (
        frame[frame["game_date"] >= date]
        .sort_values("game_id")
        .reset_index(drop=True)
    )


def _features_between(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return (
        frame[(frame["game_date"] >= start) & (frame["game_date"] < end)]
        .sort_values("game_id")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Negative controls: these mutations must NOT move the relevant features.
# --------------------------------------------------------------------------- #
def test_future_april_outcomes_do_not_change_march_features() -> None:
    """The past cannot see the future: flipping every April game leaves the
    sequential March feature rows bit-for-bit identical."""
    arch = _architecture()
    games = _games()
    baseline = build_features(games, arch)

    mutated_games = _flip_outcomes(games, games["game_date"] >= "2026-04-01")
    mutated = build_features(mutated_games, arch)

    base_march = _features_between(baseline, "2026-03-01", "2026-04-01")
    mut_march = _features_between(mutated, "2026-03-01", "2026-04-01")
    assert np.allclose(
        base_march[FEATURE_COLS].to_numpy(dtype=float),
        mut_march[FEATURE_COLS].to_numpy(dtype=float),
        atol=1e-12,
    )


def test_april_boxscore_stats_do_not_leak_into_frozen_april_features() -> None:
    """Frozen April features must not read April box scores. Mutating only the
    April rebound/turnover/foul columns (leaving scores and winners untouched)
    must not move any frozen-April component feature."""
    arch = _architecture()
    games = _games()
    baseline = build_features(games, arch, freeze_date="2026-04-01")

    mutated_games = games.copy()
    april = mutated_games["game_date"] >= "2026-04-01"
    for column in [
        "home_rebounds",
        "away_rebounds",
        "home_turnovers",
        "away_turnovers",
        "home_fouls",
        "away_fouls",
    ]:
        mutated_games.loc[april, column] = mutated_games.loc[april, column] + 25
    mutated = build_features(mutated_games, arch, freeze_date="2026-04-01")

    base_april = _features_on_or_after(baseline, "2026-04-01")
    mut_april = _features_on_or_after(mutated, "2026-04-01")
    assert np.allclose(
        base_april[FEATURE_COLS].to_numpy(dtype=float),
        mut_april[FEATURE_COLS].to_numpy(dtype=float),
        atol=1e-12,
    )


def test_input_row_order_does_not_change_features() -> None:
    """Feature construction is deterministic under input permutation: shuffling
    the raw rows yields the same features after the internal chronological
    sort. Order dependence would be a subtle leakage vector."""
    arch = _architecture()
    games = _games()
    baseline = build_features(games, arch)

    shuffled = games.sample(frac=1.0, random_state=17).reset_index(drop=True)
    mutated = build_features(shuffled, arch)

    base = baseline.sort_values("game_id").reset_index(drop=True)
    mut = mutated.sort_values("game_id").reset_index(drop=True)
    assert np.allclose(
        base[FEATURE_COLS].to_numpy(dtype=float),
        mut[FEATURE_COLS].to_numpy(dtype=float),
        atol=1e-12,
    )


# --------------------------------------------------------------------------- #
# Positive controls: these mutations SHOULD move features. They prove the
# invariance assertions above are not vacuous and that a real look-ahead
# regression would be caught.
# --------------------------------------------------------------------------- #
def test_positive_control_march_outcome_changes_frozen_april_features() -> None:
    """Frozen April state is carried forward from games through March 31, so
    flipping March outcomes MUST change the frozen-April Elo differentials.
    This proves the frozen-April invariance test can actually fail."""
    arch = _architecture()
    games = _games()
    baseline = build_features(games, arch, freeze_date="2026-04-01")

    mutated_games = _flip_outcomes(
        games,
        (games["game_date"] >= "2026-03-01") & (games["game_date"] < "2026-04-01"),
    )
    mutated = build_features(mutated_games, arch, freeze_date="2026-04-01")

    base_april = _features_on_or_after(baseline, "2026-04-01")
    mut_april = _features_on_or_after(mutated, "2026-04-01")
    assert not np.allclose(
        base_april["elo_diff"].to_numpy(dtype=float),
        mut_april["elo_diff"].to_numpy(dtype=float),
        atol=1e-6,
    )


def test_positive_control_february_outcome_changes_march_features() -> None:
    """State must propagate forward: flipping February outcomes MUST change
    March component features."""
    arch = _architecture()
    games = _games()
    baseline = build_features(games, arch)

    mutated_games = _flip_outcomes(games, games["game_date"] < "2026-03-01")
    mutated = build_features(mutated_games, arch)

    base_march = _features_between(baseline, "2026-03-01", "2026-04-01")
    mut_march = _features_between(mutated, "2026-03-01", "2026-04-01")
    assert not np.allclose(
        base_march["elo_diff"].to_numpy(dtype=float),
        mut_march["elo_diff"].to_numpy(dtype=float),
        atol=1e-6,
    )
