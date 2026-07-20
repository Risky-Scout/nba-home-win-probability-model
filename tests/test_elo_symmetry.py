import pytest

from nba_wp.features import _elo_multiplier, _winner_rating_difference


def test_mov_multiplier_is_symmetric_for_mirrored_winners() -> None:
    home_difference = _winner_rating_difference(1600.0, 1400.0, 1)
    away_difference = _winner_rating_difference(1400.0, 1600.0, 0)

    assert home_difference == 200.0
    assert away_difference == 200.0

    assert _elo_multiplier(12.0, home_difference, "log") == pytest.approx(
        _elo_multiplier(12.0, away_difference, "log")
    )
