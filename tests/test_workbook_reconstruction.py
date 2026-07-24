"""Protect the workbook (and any downstream consumer) from mis-reconstructing the
deployed Elo-only price.

The pipeline standardizes elo_diff before the logistic regression, so there are
two equivalent closed forms for the champion probability:

  standardized: p = sigmoid(intercept + coef * (elo_diff - mean)/scale)
  raw:          p = sigmoid(raw_intercept + raw_unit_coefficient * elo_diff)

A naive consumer that pairs ``raw_unit_coefficient`` with the *standardized*
``intercept`` gets the WRONG probability. These tests pin both correct forms to
the committed deployed April prices and prove the naive form is actually wrong
(so ``raw_intercept`` is genuinely required).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "artifacts" / "selected_spec.json"
APRIL = ROOT / "outputs" / "april_predictions.csv"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@pytest.fixture(scope="module")
def elo_and_april():
    spec = json.loads(SPEC.read_text())
    elo = spec["elo_model"]
    april = pd.read_csv(APRIL)
    return elo, april


def test_raw_intercept_present_and_consistent(elo_and_april):
    elo, _ = elo_and_april
    assert "raw_intercept" in elo, "selected_spec elo_model must expose raw_intercept"
    expected = elo["intercept"] - elo["standardized_coefficient"] * elo["training_mean"] / elo["training_scale"]
    assert elo["raw_intercept"] == pytest.approx(expected, abs=1e-12)
    assert elo["raw_unit_coefficient"] == pytest.approx(
        elo["standardized_coefficient"] / elo["training_scale"], abs=1e-12
    )


def test_standardized_form_reproduces_deployed_prices(elo_and_april):
    elo, april = elo_and_april
    x = april["elo_diff"].to_numpy()
    z = (x - elo["training_mean"]) / elo["training_scale"]
    p = _sigmoid(elo["standardized_coefficient"] * z + elo["intercept"])
    assert np.allclose(p, april["home_win_probability"].to_numpy(), atol=1e-9)


def test_raw_form_reproduces_deployed_prices(elo_and_april):
    elo, april = elo_and_april
    x = april["elo_diff"].to_numpy()
    p = _sigmoid(elo["raw_intercept"] + elo["raw_unit_coefficient"] * x)
    assert np.allclose(p, april["home_win_probability"].to_numpy(), atol=1e-9)


def test_naive_raw_coef_with_standardized_intercept_is_wrong(elo_and_april):
    """The footgun: raw_unit_coefficient + standardized intercept must NOT match,
    proving raw_intercept is required rather than decorative."""
    elo, april = elo_and_april
    x = april["elo_diff"].to_numpy()
    wrong = _sigmoid(elo["intercept"] + elo["raw_unit_coefficient"] * x)
    assert not np.allclose(wrong, april["home_win_probability"].to_numpy(), atol=1e-3)


def test_workbook_reconciliation_artifact_passes():
    path = ROOT / "artifacts" / "workbook_reconciliation.json"
    if not path.exists():
        import pytest
        pytest.skip("run `make reproduce` to generate workbook_reconciliation.json")
    report = json.loads(path.read_text())
    assert report["status"] == "PASS"
    assert report["max_abs_probability_diff_standardized_form"] < 1e-9
    assert report["max_abs_probability_diff_raw_form"] < 1e-9

