"""Unit tests for the cross-fitted identity-shrunk Elo calibrator.

The nested audit rejects this calibrator (it over-corrects out-of-sample), but
its *mechanics* must still be correct and honest: it shrinks toward the identity
map, never toward a constant, and sharpens an underconfident base.
"""
from __future__ import annotations

import numpy as np

from nba_wp.model import apply_calibrator, fit_identity_shrunk_calibrator, logit, sigmoid


def _underconfident_sample(n=4000, true_beta=1.5, seed=0):
    """Generate p that is deliberately underconfident: the true log-odds are
    true_beta times the stated log-odds, so MLE beta should exceed 1."""
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(2.0, 2.0, size=n), 0.02, 0.98)
    y = (rng.uniform(size=n) < sigmoid(true_beta * logit(p))).astype(int)
    return p, y


def test_small_sample_falls_back_to_identity():
    p, y = _underconfident_sample(n=10)
    a, b = fit_identity_shrunk_calibrator(p, y)
    assert (a, b) == (0.0, 1.0)


def test_huge_n0_shrinks_to_identity():
    p, y = _underconfident_sample()
    a, b = fit_identity_shrunk_calibrator(p, y, n0=10_000_000.0)
    assert abs(a) < 1e-2
    assert abs(b - 1.0) < 1e-2


def test_underconfident_base_is_sharpened_between_1_and_mle():
    p, y = _underconfident_sample(true_beta=1.5)
    a, b = fit_identity_shrunk_calibrator(p, y, n0=200.0)
    # Shrunk beta must move above 1 (sharpen) but not overshoot the raw MLE.
    a_mle, b_mle = fit_identity_shrunk_calibrator(p, y, n0=0.0)
    assert 1.0 < b < b_mle
    assert b_mle > 1.3


def test_apply_calibrator_identity_is_noop():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(apply_calibrator(0.0, 1.0, p), p)
