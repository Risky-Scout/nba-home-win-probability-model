from __future__ import annotations

import numpy as np

from nba_wp.model import (
    apply_logit_stacker,
    fit_logit_stacker,
    logit,
    stacker_calibration_dict,
)


def test_stacker_temperature_floor_removes_sharpening() -> None:
    rng = np.random.default_rng(0)
    # Separable component probs encourage an unconstrained sharp stacker.
    p_elo = np.clip(rng.beta(2.0, 2.0, size=400), 0.02, 0.98)
    p_rank = np.clip(p_elo + rng.normal(0.0, 0.05, size=400), 0.02, 0.98)
    logits = 1.8 * logit(p_elo) + 1.6 * logit(p_rank) + 0.2
    y = (rng.uniform(size=400) < 1.0 / (1.0 + np.exp(-logits))).astype(int)

    sharp = fit_logit_stacker(y, p_elo, p_rank, min_temperature=None)
    a0 = float(sharp.coef_[0, 0] + sharp.coef_[0, 1])
    # Synthetic labels follow a sharp logit stack; unconstrained total should exceed 1.
    assert a0 > 1.0 + 1e-6

    floored = fit_logit_stacker(y, p_elo, p_rank, min_temperature=1.0)
    a1 = float(floored.coef_[0, 0] + floored.coef_[0, 1])
    assert abs(a1 - 1.0) < 1e-5

    cal = stacker_calibration_dict(floored)
    assert cal["temperature_floor_applied"] is True
    assert abs(cal["temperature"] - 1.0) < 1e-5

    p = apply_logit_stacker(floored, p_elo, p_rank)
    assert p.min() > 0.0 and p.max() < 1.0


def test_stacker_weights_are_convex_when_stacker_is_used() -> None:
    """When the temperature floor is applied, the fitted blend must be a genuine
    convex logit blend: 0 <= w <= 1 for both components, weights summing to 1,
    and temperature T >= 1. (The blend is the rejected challenger, not deployed.)"""
    import pytest

    rng = np.random.default_rng(3)

    def _check(cal: dict) -> None:
        assert 0.0 <= cal["elo_weight"] <= 1.0
        assert 0.0 <= cal["rank_weight"] <= 1.0
        assert cal["elo_weight"] + cal["rank_weight"] == pytest.approx(1.0)
        assert cal["temperature"] >= 1.0 - 1e-9

    # Case 1: sharp data (unconstrained T < 1) -> projected onto T = 1.
    p_elo = np.clip(rng.beta(2.0, 2.0, size=400), 0.02, 0.98)
    p_rank = np.clip(p_elo + rng.normal(0.0, 0.05, size=400), 0.02, 0.98)
    logits = 1.9 * logit(p_elo) + 1.7 * logit(p_rank) + 0.1
    y = (rng.uniform(size=400) < 1.0 / (1.0 + np.exp(-logits))).astype(int)
    _check(stacker_calibration_dict(fit_logit_stacker(y, p_elo, p_rank, min_temperature=1.0)))

    # Case 2: data where one unconstrained coefficient is negative -> the
    # fitted blend must still be convex (non-negative weights).
    p_a = np.clip(rng.beta(2.0, 5.0, size=300), 0.02, 0.98)
    p_b = np.clip(1.0 - p_a + rng.normal(0.0, 0.02, size=300), 0.02, 0.98)
    yy = (rng.uniform(size=300) < p_a).astype(int)
    _check(stacker_calibration_dict(fit_logit_stacker(yy, p_a, p_b, min_temperature=1.0)))
