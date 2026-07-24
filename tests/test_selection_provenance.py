"""Regression tests for selection provenance, march reference metrics, dynamic
nested fold counts, and champion-output identity.

These guard the three confirmed provenance/metadata findings:
  1. selection_rule / provenance must never claim March log loss selected the
     DEPLOYED architecture (it is chosen on the frozen-policy rolling OOS
     one-standard-error surface).
  2. march_reference_metrics must be recomputed from the FINAL profiled
     architecture (not the pre-profiling per-architecture loop).
  3. Nested fold counts must be derived numerically from the fold artifacts, so
     no stale hand-written "N of M" claim can drift.
Plus the rejected-blend reporting clarification: the champion Elo-only
probability path can never be overwritten by the reporting stacker.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_wp import selection as selection_module
from nba_wp.data import load_games
from nba_wp.features import Architecture, build_features
from nba_wp.model import elo_probability, evaluate, fit_elo_model
from nba_wp.periods import derive_periods
from nba_wp.selection import load_json, run_selection

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nba-win-probability-data.csv"


def _spec() -> dict:
    return json.loads((ROOT / "artifacts" / "selected_spec.json").read_text())


def _final_metrics() -> dict:
    return json.loads((ROOT / "artifacts" / "final_metrics.json").read_text())


# --------------------------------------------------------------------------- #
# D1. Selection-provenance consistency
# --------------------------------------------------------------------------- #
def test_selection_provenance_is_single_source_of_truth() -> None:
    spec = _spec()
    prov = spec["selection_provenance"]
    assert prov["deployed_architecture_method"] == "frozen_rolling_oos_one_se"
    assert prov["march_reference_method"] == "march_reference_log_loss_brier_tiebreak"
    assert prov["nested_decision_method"] == "nested_policy_matched_champion_challenger"
    # architecture_selection.chosen must equal the deployed architecture name.
    assert spec["architecture_selection"]["chosen"] == spec["architecture"]["name"]


def test_deployed_selection_text_is_frozen_oos_one_se() -> None:
    """The deployed-selection narrative must attribute selection to the frozen
    rolling-OOS one-SE rule, and any 'March log loss' mention must be scoped to
    a reference/cross-check, never as the selector of the deployed architecture."""
    spec = _spec()
    rule = spec["selection_rule"].lower()
    assert "frozen-policy rolling out-of-sample log loss" in rule
    assert "one-standard-error" in rule
    # Any March-log-loss mention is explicitly scoped as reference-only.
    assert "reference" in rule

    forbidden = "march log loss selected the deployed"
    for text in [
        spec["selection_rule"],
        spec["selected_using"],
        spec["architecture_selection"]["rule"],
        " ".join(spec["notes"]),
    ]:
        assert forbidden not in text.lower()
        # No text may claim the deployed architecture was chosen by March log loss.
        assert "march log loss" not in text.lower() or "reference" in text.lower()


def test_march_reference_metadata_status() -> None:
    spec = _spec()
    prov = spec["march_reference_metrics_provenance"]
    assert prov["status"] == "descriptive_reference_not_pristine_holdout"
    assert prov["information_policy"] == "sequential_daily_march_state"
    # The alias must equal the reference metrics.
    assert spec["march_validation_metrics"] == spec["march_reference_metrics"]


def test_march_single_split_is_labelled_reference_only() -> None:
    sel = _spec()["architecture_selection"]
    assert "march_single_split_would_choose" in sel
    note = sel["march_single_split_would_choose_note"].lower()
    assert "reference" in note or "cross-check" in note
    assert "not the selector" in note


# --------------------------------------------------------------------------- #
# D2. Final-profiled-architecture metric reconciliation
# --------------------------------------------------------------------------- #
def test_march_reference_metrics_recompute_from_final_profiled_arch(monkeypatch) -> None:
    games = load_games(DATA)
    periods = derive_periods(games)
    selection_games = games[games["game_date"] < periods.holdout_start].copy()

    def fake_mov(games_, architecture, periods_, **kwargs):
        return {
            "chosen_offset": 2.6,
            "rationale": "forced_for_test",
            "best_offset": 2.6,
            "grid": [2.6],
            "profile": [{"mov_offset": 2.6, "mean_log_loss": 0.0,
                         "se_log_loss": 0.0, "mean_brier": 0.0}],
        }

    def fake_cold(games_, architecture, periods_, **kwargs):
        k = architecture.elo_k
        return {
            "chosen": {"warmup_games": 5, "elo_k_warmup": 2.0 * k},
            "rationale": "forced_for_test",
            "profile": [],
        }

    monkeypatch.setattr(selection_module, "profile_mov_offset", fake_mov)
    monkeypatch.setattr(selection_module, "profile_cold_start", fake_cold)

    spec, _ = run_selection(
        selection_games,
        load_json(ROOT / "configs" / "architecture_candidates.json"),
        load_json(ROOT / "configs" / "selection_policy.json"),
        periods=periods,
    )

    # The forced profiling must actually have moved the deployed architecture off
    # its defaults, otherwise the test would not exercise the recompute path.
    arch_dict = spec["architecture"]
    assert arch_dict["mov_offset"] == 2.6
    assert arch_dict["warmup_games"] == 5
    assert arch_dict["elo_k_warmup"] == 2.0 * arch_dict["elo_k"]

    # Independently rebuild the final architecture's pre-March Elo model and
    # March probabilities and reconcile to the reported reference metrics.
    arch = Architecture.from_dict(arch_dict)
    feats = build_features(selection_games, arch)
    train = feats[feats["game_date"] < periods.selection_start]
    march = feats[
        (feats["game_date"] >= periods.selection_start)
        & (feats["game_date"] < periods.holdout_start)
    ]
    model = fit_elo_model(train, arch)
    p = elo_probability(model, march)
    recomputed = evaluate(march["home_win"].to_numpy(dtype=int), p)

    for key in ["log_loss", "brier", "auc", "accuracy"]:
        assert abs(spec["march_reference_metrics"][key] - recomputed[key]) < 1e-12
        assert abs(spec["march_validation_metrics"][key] - recomputed[key]) < 1e-12

    # Provenance architecture dict must equal the deployed architecture exactly.
    assert spec["march_reference_metrics_provenance"]["architecture"] == arch_dict
    assert spec["march_reference_metrics_provenance"]["training_row_count"] == int(len(train))
    assert spec["march_reference_metrics_provenance"]["scoring_row_count"] == int(len(march))


# --------------------------------------------------------------------------- #
# D3. Fold-evidence derivation (dynamic counts)
# --------------------------------------------------------------------------- #
def _independent_fold_counts(folds: pd.DataFrame, tol: float = 1e-12) -> dict:
    ll_e = folds["log_loss_elo"].to_numpy(dtype=float)
    ll_b = folds["log_loss_blend"].to_numpy(dtype=float)
    br_e = folds["brier_elo"].to_numpy(dtype=float)
    br_b = folds["brier_blend"].to_numpy(dtype=float)
    ll_worse = ll_b > ll_e + tol
    ll_better = ll_b < ll_e - tol
    br_worse = br_b > br_e + tol
    br_better = br_b < br_e - tol
    return {
        "n_outer_folds": int(len(folds)),
        "blend_worse_log_loss_folds": int(ll_worse.sum()),
        "blend_better_log_loss_folds": int(ll_better.sum()),
        "blend_tied_log_loss_folds": int((~(ll_worse | ll_better)).sum()),
        "blend_worse_brier_folds": int(br_worse.sum()),
        "blend_better_brier_folds": int(br_better.sum()),
        "blend_tied_brier_folds": int((~(br_worse | br_better)).sum()),
        "blend_worse_both_folds": int((ll_worse & br_worse).sum()),
        "blend_better_both_folds": int((ll_better & br_better).sum()),
    }


@pytest.mark.parametrize(
    "folds_csv,summary_json",
    [
        ("nested_frozen_block_folds.csv", "nested_frozen_block_summary.json"),
        ("nested_daily_sequential_folds.csv", "nested_daily_sequential_summary.json"),
    ],
)
def test_nested_fold_counts_reconcile_to_artifacts(folds_csv, summary_json) -> None:
    folds = pd.read_csv(ROOT / "artifacts" / folds_csv)
    for col in ["log_loss_elo", "log_loss_blend", "brier_elo", "brier_blend"]:
        assert col in folds.columns, f"{folds_csv} missing per-fold column {col}"

    summary = json.loads((ROOT / "artifacts" / summary_json).read_text())
    block = summary["blend_vs_elo_fold_counts"]
    recomputed = _independent_fold_counts(folds)

    for key, value in recomputed.items():
        assert block[key] == value, f"{summary_json}:{key} {block[key]} != {value}"

    # The three log-loss categories must sum to n_outer_folds (same for Brier).
    assert (block["blend_worse_log_loss_folds"] + block["blend_better_log_loss_folds"]
            + block["blend_tied_log_loss_folds"]) == block["n_outer_folds"]
    assert (block["blend_worse_brier_folds"] + block["blend_better_brier_folds"]
            + block["blend_tied_brier_folds"]) == block["n_outer_folds"]


def test_no_hardcoded_fold_count_phrase() -> None:
    """No committed source or generated spec may hard-code the stale '10 of 11'
    fold claim (the true dynamic count differs)."""
    spec_text = (ROOT / "artifacts" / "selected_spec.json").read_text()
    src_text = (ROOT / "nba_wp" / "selection.py").read_text()
    assert "10 of 11" not in spec_text
    assert "10 of 11" not in src_text


# --------------------------------------------------------------------------- #
# D4. Champion-output identity / rejected-blend separation
# --------------------------------------------------------------------------- #
def test_champion_april_probability_is_elo_component() -> None:
    """The deployed champion is Elo-only: the frozen-April home_win_probability
    must equal the elo_component_probability exactly. This proves the reporting
    stacker (rejected challenger) never enters the champion prediction path."""
    april = pd.read_csv(ROOT / "outputs" / "april_predictions.csv")
    assert np.allclose(
        april["home_win_probability"].to_numpy(),
        april["elo_component_probability"].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )


def test_final_metrics_champion_is_elo_only_and_blend_is_separate() -> None:
    fm = _final_metrics()
    assert fm["model_family"] == "elo_only"
    assert fm["champion"] == "elo_only"

    blend = fm["rejected_challenger_blend"]
    assert blend["status"] == "rejected_challenger_reference"
    # Reporting provenance must document both stacker base-model windows.
    assert "base_model_training_window_for_stacker_fit" in blend
    assert "full_refit_base_model_window" in blend

    # The rejected blend's April metrics must DIFFER from the champion's,
    # proving the challenger path is a separate computation.
    champion = fm["primary_holdout"]["april"]
    assert blend["april_frozen"]["log_loss"] != champion["log_loss"]
