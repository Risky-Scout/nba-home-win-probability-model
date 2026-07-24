"""Consistency tests: the deployed artifacts must match the champion decision.

These guard against the exact defect an auditor caught earlier — documentation
that claimed Elo-only while the repository still deployed the rejected blend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_wp.data import load_games
from nba_wp.features import Architecture, build_features
from nba_wp.model import component_probabilities, evaluate, fit_base_models

ROOT = Path(__file__).resolve().parents[1]


def _selected_spec() -> dict:
    return json.loads((ROOT / "artifacts" / "selected_spec.json").read_text())


def _final_metrics() -> dict:
    return json.loads((ROOT / "artifacts" / "final_metrics.json").read_text())


def test_selected_spec_is_elo_only_after_promotion() -> None:
    spec = _selected_spec()
    assert spec["model_family"] == "elo_only"
    assert spec["champion"] == "elo_only"
    # Deployed probability map is a logistic on elo_diff, not a stacker.
    assert spec["elo_model"]["method"] == "logistic_on_elo_diff"
    assert "calibration" not in spec  # no deployed stacker
    # Blend retained only as a rejected challenger.
    assert spec["challenger"]["status"] == "rejected"


def test_final_metrics_model_family_matches_selected_spec() -> None:
    assert _final_metrics()["model_family"] == _selected_spec()["model_family"] == "elo_only"


def test_deployed_model_matches_champion_decision() -> None:
    for policy in ["nested_frozen_block_summary", "nested_daily_sequential_summary"]:
        summary = json.loads((ROOT / "artifacts" / f"{policy}.json").read_text())
        assert summary["champion_challenger"]["decision"] == "keep_elo_only"
    assert _selected_spec()["model_family"] == "elo_only"


def test_calibration_report_contains_every_candidate() -> None:
    for policy in ["nested_frozen_block_summary", "nested_daily_sequential_summary"]:
        summary = json.loads((ROOT / "artifacts" / f"{policy}.json").read_text())
        for candidate in ["constant", "elo_only", "rank_only", "blend"]:
            cal = summary["calibration"][candidate]
            for key in ["calibration_intercept_alpha", "calibration_slope_beta", "ece_10bin"]:
                assert key in cal and cal[key] is not None


def test_primary_april_probabilities_recompute_from_deployed_elo() -> None:
    """The committed frozen-April prices must recompute exactly from the deployed
    Elo-only coefficients in selected_spec.json."""
    spec = _selected_spec()
    em = spec["elo_model"]
    april = pd.read_csv(ROOT / "outputs" / "april_predictions.csv")
    z = (april["elo_diff"].to_numpy() - em["training_mean"]) / em["training_scale"]
    p = 1.0 / (1.0 + np.exp(-(em["intercept"] + em["standardized_coefficient"] * z)))
    assert np.allclose(p, april["home_win_probability"].to_numpy(), atol=1e-9)


def test_elo_architecture_is_selected_on_elo_oof_loss() -> None:
    """The deployed Elo architecture must minimize Elo-only March log loss
    (base fit through February) among the candidates."""
    spec = _selected_spec()
    games = load_games(ROOT / "data" / "nba-win-probability-data.csv")
    selection = games[games["game_date"] < "2026-04-01"].copy()
    candidates = json.loads(
        (ROOT / "configs" / "architecture_candidates.json").read_text()
    )["architectures"]

    scores: dict[str, float] = {}
    for values in candidates:
        arch = Architecture.from_dict(values)
        feats = build_features(selection, arch)
        train = feats[feats["game_date"] < "2026-03-01"]
        march = feats[feats["game_date"] >= "2026-03-01"]
        models = fit_base_models(train, arch)
        elo_prob, _ = component_probabilities(models, march)
        scores[arch.name] = evaluate(march["home_win"], elo_prob)["log_loss"]

    best = min(scores, key=scores.get)
    assert spec["architecture"]["name"] == best
