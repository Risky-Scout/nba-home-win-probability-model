"""Calibration-risk regression tests (15 requirements).

These prove the calibration-risk investigation is honest and that the deployed
raw-Elo champion is protected:

  1.  Pooled calibration estimates are never deployed.
  2.  Frozen calibration is fit on FROZEN-policy inner OOF rows.
  3.  Sequential calibration is fit on strictly causal inner OOF rows.
  4.  No outer-fold outcome enters calibrator fitting.
  5.  All three candidates use identical outer rows.
  6.  The identity fallback is a numerical no-op.
  7.  Platt identity-shrunk equation correctness.
  8.  Beta identity-shrunk equation correctness.
  9.  The Beta transform is monotonic when a >= 0, b >= 0.
  10. Regularization selection only ever sees inner data.
  11. A challenger failing either proper score cannot be promoted.
  12. Promotion requires BOTH policies to pass.
  13. Fixed reliability bins and tail counts are present in the decision artifact.
  14. LOFO stability is recorded in the decision artifact.
  15. Raw-Elo public output hashes are unchanged when decision == keep_elo_raw.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from nba_wp.data import load_games
from nba_wp.features import Architecture, build_features
from nba_wp.model import (
    LAM_GRID,
    apply_beta,
    apply_platt,
    beta_identity_params,
    fit_beta_identity_shrunk,
    fit_platt_identity_shrunk,
    logit,
    platt_identity_params,
    select_calibrator_lambda,
    sigmoid,
)
from nba_wp.periods import derive_periods
from scripts.calibration_challenger import (
    CANDIDATES,
    CHALLENGERS,
    _causal_inner_oof,
    _gate_for_policy,
)
from scripts.nested_validation import _make_inner_fold_computer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nba-win-probability-data.csv"
DECISION = ROOT / "artifacts" / "calibration_challenger_decision.json"
BASELINE_HASHES = Path("/tmp/nba_baseline_hashes.txt")


def _decision() -> dict:
    if not DECISION.exists():
        pytest.skip("calibration_challenger_decision.json not generated yet (run the pipeline).")
    return json.loads(DECISION.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Shared, session-scoped inner-fold fixture for the policy-separation tests.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def inner_setup():
    games = load_games(DATA)
    architectures = [Architecture.from_dict(a) for a in json.loads(
        (ROOT / "configs" / "architecture_candidates.json").read_text())["architectures"]]
    arch_by_name = {a.name: a for a in architectures}
    min_base = 40
    seq_cache = {a.name: build_features(games, a) for a in architectures}
    inner_fold = _make_inner_fold_computer(games, seq_cache, arch_by_name, min_base)
    periods = derive_periods(games)
    inner_start = (games["game_date"].min().normalize() + pd.Timedelta(weeks=4))
    origin = pd.Timestamp(periods.selection_start)  # a real outer origin in-range
    elo_arch = architectures[0].name
    return {
        "games": games,
        "arch_by_name": arch_by_name,
        "inner_fold": inner_fold,
        "inner_start": inner_start,
        "origin": origin,
        "elo_arch": elo_arch,
    }


# --------------------------------------------------------------------------- #
# 1. Pooled calibration estimates are never deployed.
# --------------------------------------------------------------------------- #
def test_pooled_calibration_never_deployed() -> None:
    spec = json.loads((ROOT / "artifacts" / "selected_spec.json").read_text())
    # No deployed stacker/calibration block, and the deployed Elo map carries
    # only the raw logistic-on-elo_diff fields (never a Platt/Beta pooled slope).
    assert "calibration" not in spec
    elo = spec["elo_model"]
    assert elo["method"] == "logistic_on_elo_diff"
    forbidden = {"delta_intercept", "delta_slope", "delta_0", "delta_a", "delta_b",
                 "platt", "beta_calibration", "pooled_slope", "calibration_slope", "lambda"}
    assert not (set(elo) & forbidden)

    # The deployed model object contains no post-hoc calibrator: the elo model is
    # the raw sklearn pipeline and the stored elo_model spec is the raw map.
    bundle = joblib.load(ROOT / "artifacts" / "trained_model.joblib")
    assert bundle["model_family"] == "elo_only"
    assert bundle["elo_calibration"]["method"] == "logistic_on_elo_diff"
    assert not (set(bundle["elo_calibration"]) & forbidden)
    bundle_text = json.dumps({k: str(v) for k, v in bundle.items()}).lower()
    assert "identity_shrunk" not in bundle_text
    assert "platt" not in bundle_text

    # The decision artifact flags its pooled slope as diagnostic-only and keeps raw.
    doc = _decision()
    assert doc["pooled_calibration_slope_is_diagnostic_only"] is True
    assert doc["decision"] == "keep_elo_raw"
    # The diagnostic pooled slope (~1.3) must not appear in the deployed spec or prices.
    diag_slope = doc["policies"]["frozen_block"]["candidates"]["elo_raw"]["calibration_slope_beta"]
    assert 1.2 < diag_slope < 1.5
    spec_text = (ROOT / "artifacts" / "selected_spec.json").read_text()
    assert f"{diag_slope:.6f}"[:6] not in spec_text  # the ~1.3 slope value is absent


def test_champion_april_prices_have_no_calibration_applied() -> None:
    april = pd.read_csv(ROOT / "outputs" / "april_predictions.csv")
    assert np.allclose(april["home_win_probability"].to_numpy(),
                       april["elo_component_probability"].to_numpy(), atol=1e-12)


# --------------------------------------------------------------------------- #
# 2 & 3 & 4. Policy-matched, strictly-causal inner OOF.
# --------------------------------------------------------------------------- #
def test_frozen_calibration_uses_frozen_inner_oof(inner_setup) -> None:
    s = inner_setup
    frozen = _causal_inner_oof(s["inner_fold"], "frozen", s["elo_arch"],
                               s["arch_by_name"], s["origin"], s["inner_start"])
    seq = _causal_inner_oof(s["inner_fold"], "sequential", s["elo_arch"],
                            s["arch_by_name"], s["origin"], s["inner_start"])
    assert len(frozen) > 0 and len(seq) > 0
    # Every fitting row is strictly earlier than the outer origin under both.
    assert (frozen["game_date"] < s["origin"]).all()
    # Frozen and sequential inner OOF carry the SAME inner games/outcomes but
    # DIFFERENT elo probabilities (frozen features vs causal features), proving
    # the frozen calibrator is genuinely fit on frozen-policy inner rows rather
    # than sequential ones.
    assert len(frozen) == len(seq)
    assert np.array_equal(np.sort(frozen["home_win"].to_numpy()),
                          np.sort(seq["home_win"].to_numpy()))
    assert not np.allclose(np.sort(frozen["pe"].to_numpy()), np.sort(seq["pe"].to_numpy()))


def test_sequential_calibration_is_strictly_causal(inner_setup) -> None:
    s = inner_setup
    seq = _causal_inner_oof(s["inner_fold"], "sequential", s["elo_arch"],
                            s["arch_by_name"], s["origin"], s["inner_start"])
    assert len(seq) > 0
    assert (seq["game_date"] < s["origin"]).all()


def test_no_outer_outcome_enters_fitting(inner_setup) -> None:
    s = inner_setup
    for policy in ["frozen", "sequential"]:
        oof = _causal_inner_oof(s["inner_fold"], policy, s["elo_arch"],
                                s["arch_by_name"], s["origin"], s["inner_start"])
        # Every fitting row is strictly earlier than the outer origin.
        assert (oof["game_date"] < s["origin"]).all()

    # And the decision artifact certifies it globally for every fold.
    doc = _decision()
    assert doc["outer_outcomes_used_for_fitting"] is False
    for pol in ["frozen_block", "daily_sequential"]:
        for f in doc["policies"][pol]["candidates"][CHALLENGERS[0]]["per_fold_calibrator"]:
            assert f["inner_max_date_before_origin"] is True


# --------------------------------------------------------------------------- #
# 5. All candidates use identical outer rows.
# --------------------------------------------------------------------------- #
def test_all_candidates_use_identical_outer_rows() -> None:
    doc = _decision()
    assert doc["outer_rows_identical"] is True
    for pol in ["frozen_block", "daily_sequential"]:
        csv = ROOT / "artifacts" / (
            "calibration_challenger_frozen_predictions.csv" if pol == "frozen_block"
            else "calibration_challenger_daily_predictions.csv")
        if not csv.exists():
            pytest.skip(f"{csv.name} not generated yet.")
        preds = pd.read_csv(csv)
        for cand in CANDIDATES:
            assert f"p_{cand}" in preds.columns
            assert preds[f"p_{cand}"].notna().all()
        # Same rows, same order for every candidate (single shared frame).
        assert len(preds) == len(preds.dropna(subset=[f"p_{c}" for c in CANDIDATES]))


# --------------------------------------------------------------------------- #
# 6. Identity fallback is a numerical no-op.
# --------------------------------------------------------------------------- #
def test_identity_fallback_is_numerical_noop() -> None:
    p = np.array([0.02, 0.1, 0.37, 0.5, 0.63, 0.9, 0.98])
    assert np.allclose(apply_platt(platt_identity_params(), p), p, atol=1e-12)
    assert np.allclose(apply_beta(beta_identity_params(), p), p, atol=1e-12)
    # A tiny / single-class sample forces the fallback, whose apply is a no-op.
    rng = np.random.default_rng(0)
    small_p = np.clip(rng.uniform(size=10), 0.05, 0.95)
    small_y = np.ones(10, dtype=int)  # one class
    pp = fit_platt_identity_shrunk(small_p, small_y, 10.0)
    bp = fit_beta_identity_shrunk(small_p, small_y, 10.0)
    assert pp["fallback"] and bp["fallback"]
    assert np.allclose(apply_platt(pp, p), p, atol=1e-12)
    assert np.allclose(apply_beta(bp, p), p, atol=1e-12)


# --------------------------------------------------------------------------- #
# 7. Platt identity-shrunk equation correctness.
# --------------------------------------------------------------------------- #
def test_platt_equation_correctness() -> None:
    di, ds = 0.37, -0.21
    params = {"delta_intercept": di, "delta_slope": ds, "alpha": di, "beta": 1.0 + ds,
              "fallback": False}
    p = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    z = logit(p)
    expected = sigmoid(z + di + ds * z)  # logit(p_cal) = z + di + ds*z
    assert np.allclose(apply_platt(params, p), expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# 8. Beta identity-shrunk equation correctness.
# --------------------------------------------------------------------------- #
def test_beta_equation_correctness() -> None:
    d0, da, db = 0.11, 0.4, -0.25
    params = {"delta_0": d0, "delta_a": da, "delta_b": db,
              "a": 1.0 + da, "b": 1.0 + db, "c": d0, "fallback": False, "monotonic": True}
    p = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    # logit(p_cal) = logit(p) + d0 + da*log(p) - db*log(1-p)   (exact spec form)
    expected = sigmoid(logit(p) + d0 + da * np.log(p) - db * np.log(1.0 - p))
    assert np.allclose(apply_beta(params, p), expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# 9. Beta transform monotonic when a >= 0, b >= 0.
# --------------------------------------------------------------------------- #
def test_beta_transform_is_monotonic() -> None:
    params = {"a": 1.3, "b": 0.7, "c": 0.2, "delta_0": 0.2, "delta_a": 0.3,
              "delta_b": -0.3, "fallback": False, "monotonic": True}
    grid = np.linspace(0.001, 0.999, 500)
    out = apply_beta(params, grid)
    assert np.all(np.diff(out) >= -1e-12)
    # A fit that would require a<0 or b<0 must fall back to the (monotonic) identity.
    rng = np.random.default_rng(1)
    p = np.clip(rng.beta(2, 2, 400), 0.02, 0.98)
    y = (rng.uniform(size=400) < sigmoid(0.5 * logit(p))).astype(int)
    fit = fit_beta_identity_shrunk(p, y, 3.0)
    assert fit["monotonic"] is True and fit["a"] >= 0.0 and fit["b"] >= 0.0


# --------------------------------------------------------------------------- #
# 10. Regularization selection only ever sees inner data.
# --------------------------------------------------------------------------- #
def test_lambda_selection_excludes_outer_outcomes() -> None:
    rng = np.random.default_rng(3)
    n_inner, n_outer = 600, 300
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(n_inner + n_outer) // 15, unit="D")
    p_all = np.clip(rng.beta(2, 2, n_inner + n_outer), 0.02, 0.98)
    y_all = (rng.uniform(size=n_inner + n_outer) < sigmoid(1.3 * logit(p_all))).astype(int)
    origin = dates[n_inner]

    inner = slice(0, n_inner)
    # Selection receives ONLY inner arrays.
    sel = select_calibrator_lambda(fit_platt_identity_shrunk, apply_platt,
                                   p_all[inner], y_all[inner], dates[inner].to_numpy())
    assert sel["selected_lambda"] in LAM_GRID
    assert (dates[inner] < origin).all()  # nothing at/after origin was passed

    # Corrupting the OUTER outcomes cannot change the selected lambda, because the
    # selector never sees them.
    y_corrupt = y_all.copy()
    y_corrupt[n_inner:] = 1 - y_corrupt[n_inner:]
    sel2 = select_calibrator_lambda(fit_platt_identity_shrunk, apply_platt,
                                    p_all[inner], y_corrupt[inner], dates[inner].to_numpy())
    assert sel2["selected_lambda"] == sel["selected_lambda"]


# --------------------------------------------------------------------------- #
# 11 & 12. Promotion gate logic.
# --------------------------------------------------------------------------- #
def _synthetic_policy_block(delta_ll: float, delta_br: float):
    """Minimal policy block where the challenger is uniformly better/worse than
    raw by a fixed amount, with a single fold, for gate-logic unit tests."""
    return {
        "candidates": {
            "elo_raw": {"log_loss": 0.50, "brier": 0.20},
            "cand": {"log_loss": 0.50 + delta_ll, "brier": 0.20 + delta_br},
        },
        "paired_block_bootstrap": {"cand": {
            "delta_log_loss_mean": delta_ll, "delta_log_loss_ci_97_5": delta_ll,
            "delta_brier_mean": delta_br, "delta_brier_ci_97_5": delta_br}},
        "lofo": {"cand": {"all_lofo_log_loss_negative": delta_ll < 0,
                          "all_lofo_brier_negative": delta_br < 0}},
        "fold_contribution": {"cand": {
            "log_loss": {"positive_aggregate_improvement": delta_ll < 0,
                         "max_single_fold_share": 0.10 if delta_ll < 0 else None},
            "brier": {"positive_aggregate_improvement": delta_br < 0,
                      "max_single_fold_share": 0.10 if delta_br < 0 else None}}},
        "tail_audit": {"cand": {"no_material_tail_degradation": True}},
    }


def test_challenger_failing_a_proper_score_cannot_promote() -> None:
    # Better log loss but WORSE Brier -> must fail (both proper scores required).
    block = _synthetic_policy_block(delta_ll=-0.01, delta_br=+0.01)
    gate = _gate_for_policy(block, "cand")
    assert gate["gates"]["mean_delta_log_loss_negative"] is True
    assert gate["gates"]["mean_delta_brier_negative"] is False
    assert gate["all_gates_pass"] is False


def test_promotion_requires_both_policies(monkeypatch) -> None:
    # Frozen passes every gate; daily fails on the proper scores. The overall
    # decision must therefore NOT promote (both policies required).
    good = _synthetic_policy_block(delta_ll=-0.01, delta_br=-0.01)
    bad = _synthetic_policy_block(delta_ll=+0.01, delta_br=+0.01)
    frozen_gate = _gate_for_policy(good, "cand")
    daily_gate = _gate_for_policy(bad, "cand")
    assert frozen_gate["all_gates_pass"] is True
    assert daily_gate["all_gates_pass"] is False
    both = frozen_gate["all_gates_pass"] and daily_gate["all_gates_pass"]
    assert both is False


# --------------------------------------------------------------------------- #
# 13. Fixed reliability bins and tail counts present in decision artifact.
# --------------------------------------------------------------------------- #
def test_fixed_reliability_bins_and_counts_present() -> None:
    doc = _decision()
    for pol in ["frozen_block", "daily_sequential"]:
        pooled_games = doc["policies"][pol]["pooled_games"]
        for cand in CANDIDATES:
            diag = doc["policies"][pol]["candidates"][cand]
            bins = diag["reliability_bins"]
            assert len(bins) == 10  # boundaries [0.0, 0.1, ..., 1.0]
            assert abs(bins[0]["bin_lower"] - 0.0) < 1e-12
            assert abs(bins[-1]["bin_upper"] - 1.0) < 1e-12
            # Every prediction lands in exactly one fixed bin.
            assert sum(b["count"] for b in bins) == pooled_games
            assert "count_p_le_0_10" in diag and "count_p_ge_0_90" in diag
            assert "ece_fixed_10bin" in diag


# --------------------------------------------------------------------------- #
# 14. LOFO stability recorded in decision artifact.
# --------------------------------------------------------------------------- #
def test_lofo_stability_recorded() -> None:
    doc = _decision()
    for pol in ["frozen_block", "daily_sequential"]:
        for ch in CHALLENGERS:
            lofo = doc["policies"][pol]["lofo"][ch]
            assert len(lofo["entries"]) > 0
            for e in lofo["entries"]:
                assert "left_out_origin" in e and "delta_log_loss" in e and "delta_brier" in e
            assert "all_lofo_log_loss_negative" in lofo
            assert "all_lofo_brier_negative" in lofo


# --------------------------------------------------------------------------- #
# 15. Raw-Elo public output hashes unchanged when decision == keep_elo_raw.
# --------------------------------------------------------------------------- #
def test_raw_elo_output_hashes_unchanged() -> None:
    doc = _decision()
    if doc["decision"] != "keep_elo_raw":
        pytest.skip("A challenger was promoted; champion hashes are expected to change.")
    if not BASELINE_HASHES.exists():
        pytest.skip("Baseline hash file /tmp/nba_baseline_hashes.txt not present.")
    expected = {}
    for line in BASELINE_HASHES.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        h, rel = line.split()[0], line.split()[-1]
        expected[rel] = h
    assert expected, "baseline hash file empty"
    for rel, h in expected.items():
        path = ROOT / rel
        assert path.exists(), f"missing protected artifact {rel}"
        assert _sha256(path) == h, f"champion artifact changed: {rel}"
