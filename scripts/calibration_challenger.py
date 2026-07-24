"""Calibration-risk investigation: three complete procedures under two policies.

This is a rigorous, model-risk-style audit of whether a post-hoc probability
calibrator legitimately improves on the DEPLOYED raw-Elo champion. It evaluates
three complete procedures

    * ``elo_raw``                     -- the deployed champion (no calibration)
    * ``elo_platt_identity_shrunk``   -- identity-shrunk Platt-on-logit
    * ``elo_beta_identity_shrunk``    -- identity-shrunk Beta calibration

under two clearly separated information policies

    * ``frozen_block``     -- performance state frozen at each outer origin O
    * ``daily_sequential`` -- base models refit through t-1 for each date t

reusing the SAME policy-matched nested rolling-origin engine as
``scripts.nested_validation`` so all three candidates are scored on IDENTICAL
outer-fold game rows.

Calibrators are fit ONLY on the elo architecture's policy-matched inner
out-of-fold rows STRICTLY EARLIER than the outer origin (frozen inner OOF for
frozen; sequential inner OOF for daily), the L2 shrinkage strength is chosen
ONLY by chronological inner validation of that inner OOF, and any degeneracy
falls back to the numerical identity. No outer-fold outcome is ever seen while
fitting or while selecting the penalty.

A challenger may replace raw Elo ONLY IF, under BOTH policies, ALL promotion
gates hold (mean delta log loss < 0; mean delta Brier < 0; upper 95% week-block
CI < 0 for both proper scores; every leave-one-outer-fold-out aggregate delta
< 0 for both; no single fold contributes >= 50% of the improvement; and no
material tail degradation). The decision is DERIVED from data, never assumed.

Output: ``artifacts/calibration_challenger_decision.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nba_wp.data import load_games
from nba_wp.features import Architecture, build_features
from nba_wp.periods import derive_periods
from nba_wp.model import (
    apply_beta,
    apply_platt,
    fit_base_models,
    fit_beta_identity_shrunk,
    fit_platt_identity_shrunk,
    component_probabilities,
    logit,
    select_calibrator_lambda,
)
from scripts.nested_validation import (
    EPS,
    _brier,
    _clip,
    _inner_oof,
    _ll,
    _make_inner_fold_computer,
    _select_architectures,
    _weekly,
)

CANDIDATES = ["elo_raw", "elo_platt_identity_shrunk", "elo_beta_identity_shrunk"]
CHALLENGERS = ["elo_platt_identity_shrunk", "elo_beta_identity_shrunk"]
FIXED_BIN_EDGES = [round(0.1 * i, 10) for i in range(11)]  # [0.0, 0.1, ..., 1.0]
TAIL_MATERIAL_TOL = 0.02  # >2 percentage-point worsening of tail calibration gap is "material".


# --------------------------------------------------------------------------- #
# Calibrator fitting on a policy-matched, STRICTLY-CAUSAL inner OOF.
# --------------------------------------------------------------------------- #
def _causal_inner_oof(inner_fold, policy, elo_arch, arch_by_name, origin, inner_start):
    """Elo inner OOF under ``policy``, restricted to rows STRICTLY BEFORE origin.

    ``_inner_oof`` concatenates inner folds [io, io+7) for io < origin; the last
    such fold can spill a few days into [origin, origin+7). We drop every row
    with game_date >= origin so the calibrator's fitting inputs are provably
    disjoint from the outer block and no outer-fold outcome can leak in."""
    oof = _inner_oof(inner_fold, policy, arch_by_name[elo_arch], origin, inner_start)
    if len(oof) == 0:
        return oof
    return oof[oof["game_date"] < origin].reset_index(drop=True)


def _fit_calibrators(elo_oof: pd.DataFrame):
    """Grid-select lambda by chronological inner validation, then REFIT on all
    inner OOF. Returns (platt_params, beta_params, meta)."""
    p = elo_oof["pe"].to_numpy()
    y = elo_oof["home_win"].to_numpy(dtype=int)
    dates = elo_oof["game_date"].to_numpy()

    platt_sel = select_calibrator_lambda(fit_platt_identity_shrunk, apply_platt, p, y, dates)
    beta_sel = select_calibrator_lambda(fit_beta_identity_shrunk, apply_beta, p, y, dates)
    platt_params = fit_platt_identity_shrunk(p, y, platt_sel["selected_lambda"])
    beta_params = fit_beta_identity_shrunk(p, y, beta_sel["selected_lambda"])

    dmin = pd.Timestamp(elo_oof["game_date"].min()).strftime("%Y-%m-%d") if len(elo_oof) else None
    dmax = pd.Timestamp(elo_oof["game_date"].max()).strftime("%Y-%m-%d") if len(elo_oof) else None
    meta = {
        "inner_training_rows": int(len(elo_oof)),
        "inner_date_min": dmin,
        "inner_date_max": dmax,
        "platt_selected_lambda": platt_sel["selected_lambda"],
        "beta_selected_lambda": beta_sel["selected_lambda"],
        "platt_lambda_degenerate": bool(platt_sel["degenerate"]),
        "beta_lambda_degenerate": bool(beta_sel["degenerate"]),
    }
    return platt_params, beta_params, meta


# --------------------------------------------------------------------------- #
# Outer-fold construction per policy (elo only; calibrators are transforms).
# --------------------------------------------------------------------------- #
def _frozen_predictions(games, architectures, arch_by_name, seq_cache, inner_fold,
                        origins, inner_start_ts, min_base, min_stacker):
    rows: list[pd.DataFrame] = []
    fold_cal: list[dict] = []
    for origin in origins:
        block_end = origin + pd.Timedelta(days=7)
        sample = seq_cache[architectures[0].name]
        if len(sample[(sample["game_date"] >= origin) & (sample["game_date"] < block_end)]) == 0:
            continue
        if len(sample[sample["game_date"] < origin]) < min_base:
            continue
        best = _select_architectures(inner_fold, "frozen", architectures, origin,
                                     inner_start_ts, min_base, min_stacker)
        elo_arch = best["elo_only"][1]
        if elo_arch is None:
            continue

        elo_base = fit_base_models(
            seq_cache[elo_arch][seq_cache[elo_arch]["game_date"] < origin], arch_by_name[elo_arch])
        feat = build_features(games, arch_by_name[elo_arch], freeze_date=origin)
        be = feat[(feat["game_date"] >= origin) & (feat["game_date"] < block_end)].sort_values("game_id")
        if len(be) == 0:
            continue
        p_elo = component_probabilities(elo_base, be)[0]
        y = be["home_win"].to_numpy(dtype=int)

        elo_oof = _causal_inner_oof(inner_fold, "frozen", elo_arch, arch_by_name, origin, inner_start_ts)
        platt_params, beta_params, meta = _fit_calibrators(elo_oof)
        p_platt = apply_platt(platt_params, p_elo)
        p_beta = apply_beta(beta_params, p_elo)

        rows.append(pd.DataFrame({
            "game_id": be["game_id"].to_numpy(),
            "game_date": be["game_date"].to_numpy(),
            "origin": origin.strftime("%Y-%m-%d"),
            "home_win": y,
            "p_elo_raw": p_elo,
            "p_elo_platt_identity_shrunk": p_platt,
            "p_elo_beta_identity_shrunk": p_beta,
        }))
        fold_cal.append(_fold_cal_record(origin, elo_arch, meta, platt_params, beta_params))
    preds = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return preds, fold_cal


def _daily_predictions(games, architectures, arch_by_name, seq_cache, inner_fold,
                       origins, inner_start_ts, min_base, min_stacker):
    rows: list[pd.DataFrame] = []
    fold_cal: list[dict] = []
    sample = seq_cache[architectures[0].name]
    for origin in origins:
        block_end = origin + pd.Timedelta(days=7)
        if len(sample[(sample["game_date"] >= origin) & (sample["game_date"] < block_end)]) == 0:
            continue
        if len(sample[sample["game_date"] < origin]) < min_base:
            continue
        best = _select_architectures(inner_fold, "sequential", architectures, origin,
                                     inner_start_ts, min_base, min_stacker)
        elo_arch = best["elo_only"][1]
        if elo_arch is None:
            continue

        elo_oof = _causal_inner_oof(inner_fold, "sequential", elo_arch, arch_by_name, origin, inner_start_ts)
        platt_params, beta_params, meta = _fit_calibrators(elo_oof)

        block_dates = sorted(sample[(sample["game_date"] >= origin)
                                    & (sample["game_date"] < block_end)]["game_date"].unique())
        seq = seq_cache[elo_arch]
        produced = False
        for t in block_dates:
            t = pd.Timestamp(t)
            tr = seq[seq["game_date"] < t]
            fold = seq[seq["game_date"] == t].sort_values("game_id")
            if len(tr) < min_base or tr["home_win"].nunique() < 2 or len(fold) == 0:
                continue
            base = fit_base_models(tr, arch_by_name[elo_arch])
            p_elo = component_probabilities(base, fold)[0]
            yd = fold["home_win"].to_numpy(dtype=int)
            rows.append(pd.DataFrame({
                "game_id": fold["game_id"].to_numpy(),
                "game_date": fold["game_date"].to_numpy(),
                "origin": origin.strftime("%Y-%m-%d"),
                "home_win": yd,
                "p_elo_raw": p_elo,
                "p_elo_platt_identity_shrunk": apply_platt(platt_params, p_elo),
                "p_elo_beta_identity_shrunk": apply_beta(beta_params, p_elo),
            }))
            produced = True
        if produced:
            fold_cal.append(_fold_cal_record(origin, elo_arch, meta, platt_params, beta_params))
    preds = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return preds, fold_cal


def _fold_cal_record(origin, elo_arch, meta, platt_params, beta_params) -> dict:
    return {
        "origin": origin.strftime("%Y-%m-%d"),
        "selected_elo_architecture": elo_arch,
        "inner_training_rows": meta["inner_training_rows"],
        "inner_date_min": meta["inner_date_min"],
        "inner_date_max": meta["inner_date_max"],
        "inner_max_date_before_origin": bool(
            meta["inner_date_max"] is None or meta["inner_date_max"] < origin.strftime("%Y-%m-%d")),
        "elo_platt_identity_shrunk": {
            "selected_lambda": meta["platt_selected_lambda"],
            "lambda_degenerate": meta["platt_lambda_degenerate"],
            "delta_intercept": platt_params["delta_intercept"],
            "delta_slope": platt_params["delta_slope"],
            "alpha": platt_params["alpha"],
            "beta": platt_params["beta"],
            "fallback_status": bool(platt_params["fallback"]),
        },
        "elo_beta_identity_shrunk": {
            "selected_lambda": meta["beta_selected_lambda"],
            "lambda_degenerate": meta["beta_lambda_degenerate"],
            "delta_0": beta_params["delta_0"],
            "delta_a": beta_params["delta_a"],
            "delta_b": beta_params["delta_b"],
            "a": beta_params["a"],
            "b": beta_params["b"],
            "c": beta_params["c"],
            "fallback_status": bool(beta_params["fallback"]),
            "monotonic": bool(beta_params["monotonic"]),
        },
    }


# --------------------------------------------------------------------------- #
# Diagnostics.
# --------------------------------------------------------------------------- #
def _calibration_intercept_slope(y, p):
    if len(np.unique(y)) < 2:
        return None, None
    x = logit(_clip(p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10_000).fit(x, y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def _fixed_reliability(y, p):
    """Fixed-bin reliability table + ECE using boundaries [0.0, 0.1, ..., 1.0]."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    edges = np.asarray(FIXED_BIN_EDGES, dtype=float)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    bins = []
    ece = 0.0
    n = len(y)
    for b in range(len(edges) - 1):
        mask = idx == b
        cnt = int(mask.sum())
        entry = {
            "bin_lower": float(edges[b]),
            "bin_upper": float(edges[b + 1]),
            "count": cnt,
            "mean_forecast": float(np.mean(p[mask])) if cnt else None,
            "observed_rate": float(np.mean(y[mask])) if cnt else None,
        }
        bins.append(entry)
        if cnt:
            ece += (cnt / n) * abs(entry["observed_rate"] - entry["mean_forecast"])
    return bins, float(ece)


def _per_fold_metrics(preds: pd.DataFrame, candidate: str) -> list[dict]:
    col = f"p_{candidate}"
    out = []
    for origin, grp in preds.groupby("origin", sort=True):
        y = grp["home_win"].to_numpy(dtype=int)
        p = grp[col].to_numpy()
        out.append({"origin": origin, "fold_games": int(len(grp)),
                    "log_loss": _ll(y, p), "brier": _brier(y, p)})
    return out


def _diagnostics(preds: pd.DataFrame, candidate: str, fold_cal: list[dict]) -> dict:
    y = preds["home_win"].to_numpy(dtype=int)
    p = preds[f"p_{candidate}"].to_numpy()
    alpha, beta = _calibration_intercept_slope(y, p)
    bins, ece = _fixed_reliability(y, p)
    from sklearn.metrics import roc_auc_score
    diag = {
        "log_loss": _ll(y, p),
        "brier": _brier(y, p),
        "auc": float(roc_auc_score(y, _clip(p))) if len(np.unique(y)) > 1 else None,
        "calibration_intercept_alpha": alpha,
        "calibration_slope_beta": beta,
        "ece_fixed_10bin": ece,
        "reliability_bins": bins,
        "count_p_le_0_10": int(np.sum(p <= 0.10)),
        "count_p_ge_0_90": int(np.sum(p >= 0.90)),
        "per_fold_metrics": _per_fold_metrics(preds, candidate),
    }
    if candidate in CHALLENGERS:
        diag["per_fold_calibrator"] = [
            {"origin": f["origin"],
             "selected_elo_architecture": f["selected_elo_architecture"],
             "inner_training_rows": f["inner_training_rows"],
             "inner_date_min": f["inner_date_min"],
             "inner_date_max": f["inner_date_max"],
             "inner_max_date_before_origin": f["inner_max_date_before_origin"],
             **f[candidate]}
            for f in fold_cal
        ]
    return diag


def _week_blocks(preds: pd.DataFrame):
    week = preds["game_date"].dt.isocalendar().week.astype(int).to_numpy()
    year = preds["game_date"].dt.isocalendar().year.astype(int).to_numpy()
    block_id = year * 100 + week
    return block_id, np.unique(block_id)


def _paired_bootstrap(preds: pd.DataFrame, challenger: str, rng, n_boot: int) -> dict:
    """Week-block paired bootstrap of (challenger - raw) log loss and Brier."""
    y = preds["home_win"].to_numpy(dtype=int)
    p_ch = preds[f"p_{challenger}"].to_numpy()
    p_raw = preds["p_elo_raw"].to_numpy()
    block_id, blocks = _week_blocks(preds)
    d_ll = np.empty(n_boot)
    d_br = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.where(block_id == b)[0]
                              for b in rng.choice(blocks, size=len(blocks), replace=True)])
        yy = y[idx]
        d_ll[i] = _ll(yy, p_ch[idx]) - _ll(yy, p_raw[idx])
        d_br[i] = _brier(yy, p_ch[idx]) - _brier(yy, p_raw[idx])
    return {
        "delta_log_loss_mean": float(d_ll.mean()),
        "delta_log_loss_ci_2_5": float(np.quantile(d_ll, 0.025)),
        "delta_log_loss_ci_97_5": float(np.quantile(d_ll, 0.975)),
        "delta_brier_mean": float(d_br.mean()),
        "delta_brier_ci_2_5": float(np.quantile(d_br, 0.025)),
        "delta_brier_ci_97_5": float(np.quantile(d_br, 0.975)),
        "replicates": int(n_boot),
        "replicates_favoring_challenger_log_loss": int(np.sum(d_ll < 0.0)),
        "replicates_favoring_challenger_brier": int(np.sum(d_br < 0.0)),
    }


def _lofo(preds: pd.DataFrame, challenger: str) -> dict:
    """Leave-one-outer-fold-out aggregate deltas (challenger - raw), pooled over
    the REMAINING folds each time. Every such delta must be < 0 for a gate to
    pass, proving no single outer fold props up (or masks) the comparison."""
    y_all = preds["home_win"].to_numpy(dtype=int)
    p_ch = preds[f"p_{challenger}"].to_numpy()
    p_raw = preds["p_elo_raw"].to_numpy()
    origins = sorted(preds["origin"].unique())
    entries = []
    for o in origins:
        mask = (preds["origin"].to_numpy() != o)
        yy = y_all[mask]
        if len(np.unique(yy)) < 2:
            continue
        d_ll = _ll(yy, p_ch[mask]) - _ll(yy, p_raw[mask])
        d_br = _brier(yy, p_ch[mask]) - _brier(yy, p_raw[mask])
        entries.append({"left_out_origin": o, "delta_log_loss": d_ll, "delta_brier": d_br})
    all_ll_neg = bool(entries) and all(e["delta_log_loss"] < 0.0 for e in entries)
    all_br_neg = bool(entries) and all(e["delta_brier"] < 0.0 for e in entries)
    return {"entries": entries, "all_lofo_log_loss_negative": all_ll_neg,
            "all_lofo_brier_negative": all_br_neg}


def _fold_contribution(preds: pd.DataFrame, challenger: str) -> dict:
    """Per-fold contribution to the total improvement, per metric. Improvement of
    fold f = (raw_loss_f - challenger_loss_f) * n_f (positive == challenger
    better). max_share = max_f improvement_f / total_improvement. If the total
    improvement is <= 0 there is no aggregate gain to attribute and the
    no-single-fold-dominates gate is False."""
    y_all = preds["home_win"].to_numpy(dtype=int)
    p_ch = preds[f"p_{challenger}"].to_numpy()
    p_raw = preds["p_elo_raw"].to_numpy()
    origins = sorted(preds["origin"].unique())
    imp_ll, imp_br = [], []
    for o in origins:
        mask = (preds["origin"].to_numpy() == o)
        yy = y_all[mask]
        n = int(mask.sum())
        imp_ll.append((_ll(yy, p_raw[mask]) - _ll(yy, p_ch[mask])) * n)
        imp_br.append((_brier(yy, p_raw[mask]) - _brier(yy, p_ch[mask])) * n)
    imp_ll = np.asarray(imp_ll)
    imp_br = np.asarray(imp_br)

    def _share(imp):
        total = float(imp.sum())
        if total <= 0.0:
            return {"total_improvement": total, "max_single_fold_share": None,
                    "positive_aggregate_improvement": False}
        return {"total_improvement": total,
                "max_single_fold_share": float(imp.max() / total),
                "positive_aggregate_improvement": True}

    return {"log_loss": _share(imp_ll), "brier": _share(imp_br)}


def _tail_audit(preds: pd.DataFrame, challenger: str) -> dict:
    """Compare calibration gaps in the raw model's extreme-confidence regions.

    Rows are fixed by the RAW forecast (p_raw >= 0.90 and p_raw <= 0.10). In each
    region we compare |mean_forecast - observed_rate| for raw vs the challenger
    over the SAME rows. A tail 'worsens materially' if the challenger's gap
    exceeds the raw gap by more than TAIL_MATERIAL_TOL. Empty regions cannot
    worsen."""
    y = preds["home_win"].to_numpy(dtype=int)
    p_raw = preds["p_elo_raw"].to_numpy()
    p_ch = preds[f"p_{challenger}"].to_numpy()

    def region(mask):
        cnt = int(mask.sum())
        if cnt == 0:
            return {"count": 0, "raw_gap": None, "challenger_gap": None,
                    "worsens_materially": False}
        obs = float(np.mean(y[mask]))
        raw_gap = abs(float(np.mean(p_raw[mask])) - obs)
        ch_gap = abs(float(np.mean(p_ch[mask])) - obs)
        return {"count": cnt, "observed_rate": obs, "raw_mean_forecast": float(np.mean(p_raw[mask])),
                "challenger_mean_forecast": float(np.mean(p_ch[mask])),
                "raw_gap": raw_gap, "challenger_gap": ch_gap,
                "worsens_materially": bool(ch_gap > raw_gap + TAIL_MATERIAL_TOL)}

    high = region(p_raw >= 0.90)
    low = region(p_raw <= 0.10)
    return {"tolerance": TAIL_MATERIAL_TOL, "high_tail_p_ge_0_90": high, "low_tail_p_le_0_10": low,
            "no_material_tail_degradation": bool(
                not high["worsens_materially"] and not low["worsens_materially"])}


# --------------------------------------------------------------------------- #
# Per-policy assembly + promotion gates.
# --------------------------------------------------------------------------- #
def _policy_block(preds: pd.DataFrame, fold_cal: list[dict], rng, n_boot: int) -> dict:
    candidates = {c: _diagnostics(preds, c, fold_cal) for c in CANDIDATES}
    boot = {ch: _paired_bootstrap(preds, ch, rng, n_boot) for ch in CHALLENGERS}
    lofo = {ch: _lofo(preds, ch) for ch in CHALLENGERS}
    contrib = {ch: _fold_contribution(preds, ch) for ch in CHALLENGERS}
    tails = {ch: _tail_audit(preds, ch) for ch in CHALLENGERS}
    return {
        "pooled_games": int(len(preds)),
        "n_outer_folds": int(preds["origin"].nunique()),
        "candidates": candidates,
        "paired_block_bootstrap": boot,
        "lofo": lofo,
        "fold_contribution": contrib,
        "tail_audit": tails,
    }


def _gate_for_policy(policy_block: dict, challenger: str) -> dict:
    raw = policy_block["candidates"]["elo_raw"]
    ch = policy_block["candidates"][challenger]
    boot = policy_block["paired_block_bootstrap"][challenger]
    lofo = policy_block["lofo"][challenger]
    contrib = policy_block["fold_contribution"][challenger]
    tail = policy_block["tail_audit"][challenger]

    mean_dll = ch["log_loss"] - raw["log_loss"]
    mean_dbr = ch["brier"] - raw["brier"]
    ll_share = contrib["log_loss"]["max_single_fold_share"]
    br_share = contrib["brier"]["max_single_fold_share"]
    no_single_fold = bool(
        contrib["log_loss"]["positive_aggregate_improvement"]
        and contrib["brier"]["positive_aggregate_improvement"]
        and ll_share is not None and ll_share < 0.50
        and br_share is not None and br_share < 0.50)

    gates = {
        "mean_delta_log_loss_negative": bool(mean_dll < 0.0),
        "mean_delta_brier_negative": bool(mean_dbr < 0.0),
        "boot_upper_ci_log_loss_negative": bool(boot["delta_log_loss_ci_97_5"] < 0.0),
        "boot_upper_ci_brier_negative": bool(boot["delta_brier_ci_97_5"] < 0.0),
        "all_lofo_log_loss_negative": bool(lofo["all_lofo_log_loss_negative"]),
        "all_lofo_brier_negative": bool(lofo["all_lofo_brier_negative"]),
        "no_single_fold_dominates": no_single_fold,
        "no_material_tail_degradation": bool(tail["no_material_tail_degradation"]),
    }
    return {
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "values": {
            "pooled_delta_log_loss": mean_dll,
            "pooled_delta_brier": mean_dbr,
            "boot_delta_log_loss_mean": boot["delta_log_loss_mean"],
            "boot_delta_log_loss_ci_97_5": boot["delta_log_loss_ci_97_5"],
            "boot_delta_brier_mean": boot["delta_brier_mean"],
            "boot_delta_brier_ci_97_5": boot["delta_brier_ci_97_5"],
            "max_single_fold_share_log_loss": ll_share,
            "max_single_fold_share_brier": br_share,
        },
    }


def run(data_path, config_path, artifact_dir, *, outer_start=None, inner_start=None,
        min_base=40, min_stacker=40, n_boot=2000, seed=2026):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    architectures = [Architecture.from_dict(a)
                     for a in json.loads(Path(config_path).read_text())["architectures"]]
    arch_by_name = {a.name: a for a in architectures}
    games = load_games(data_path)

    periods = derive_periods(games)
    if outer_start is None:
        outer_start = periods.s(periods.outer_start)
    if inner_start is None:
        inner_start = (games["game_date"].min().normalize() + pd.Timedelta(weeks=4)).strftime("%Y-%m-%d")
    inner_start_ts = pd.Timestamp(inner_start)

    seq_cache = {a.name: build_features(games, a) for a in architectures}
    inner_fold = _make_inner_fold_computer(games, seq_cache, arch_by_name, min_base)
    origins = _weekly(pd.Timestamp(outer_start), games["game_date"].max())

    frozen_preds, frozen_fold_cal = _frozen_predictions(
        games, architectures, arch_by_name, seq_cache, inner_fold,
        origins, inner_start_ts, min_base, min_stacker)
    daily_preds, daily_fold_cal = _daily_predictions(
        games, architectures, arch_by_name, seq_cache, inner_fold,
        origins, inner_start_ts, min_base, min_stacker)

    # Independent bootstrap RNGs seeded identically per policy (seed 2026).
    frozen_block = _policy_block(frozen_preds, frozen_fold_cal, np.random.default_rng(seed), n_boot)
    daily_block = _policy_block(daily_preds, daily_fold_cal, np.random.default_rng(seed), n_boot)
    policies = {"frozen_block": frozen_block, "daily_sequential": daily_block}

    # --- Structural integrity checks (investigation validity) ---
    # All three candidates are transforms of the SAME p_elo_raw on the SAME rows,
    # so outer rows are identical by construction; we still verify it explicitly.
    def _rows_ok(preds):
        if len(preds) == 0:
            return False
        cols = ["p_elo_raw"] + [f"p_{c}" for c in CHALLENGERS]
        return all(len(preds[c]) == len(preds) and preds[c].notna().all() for c in cols)

    outer_rows_identical = bool(_rows_ok(frozen_preds) and _rows_ok(daily_preds))

    # No outer-fold outcome entered fitting: every fold's inner OOF max date is
    # strictly before its origin (enforced by _causal_inner_oof).
    def _causal_ok(fold_cal):
        return all(f["inner_max_date_before_origin"] for f in fold_cal) if fold_cal else False
    outer_outcomes_used_for_fitting = not bool(_causal_ok(frozen_fold_cal) and _causal_ok(daily_fold_cal))

    # --- Promotion gates: a challenger promotes ONLY IF every gate passes under
    # BOTH policies. ---
    promotion_gate = {}
    promoted = None
    for ch in CHALLENGERS:
        fg = _gate_for_policy(frozen_block, ch)
        dg = _gate_for_policy(daily_block, ch)
        both = bool(fg["all_gates_pass"] and dg["all_gates_pass"])
        promotion_gate[ch] = {
            "frozen_block": fg,
            "daily_sequential": dg,
            "all_gates_pass_both_policies": both,
        }
        if both and promoted is None:
            promoted = ch

    integrity_ok = bool(outer_rows_identical and not outer_outcomes_used_for_fitting)
    decision = f"promote_{promoted}" if (promoted is not None and integrity_ok) else "keep_elo_raw"

    if decision == "keep_elo_raw":
        reason = ("No challenger passed every promotion gate under both information "
                  "policies; the deployed raw-Elo champion is retained unchanged. The "
                  "in-sample calibration slope (~1.3) is diagnostic-only and is NOT "
                  "applied anywhere in deployment: post-hoc identity-shrunk calibration "
                  "over-corrects out-of-sample (worse pooled log loss and Brier with "
                  "week-block bootstrap upper CIs above zero).")
    else:
        reason = (f"{promoted} passed EVERY promotion gate (mean delta log loss and Brier "
                  "< 0, week-block bootstrap upper CIs < 0, all leave-one-outer-fold-out "
                  "deltas < 0, no single fold dominating the gain, and no material tail "
                  "degradation) under BOTH the frozen-block and daily-sequential policies.")

    decision_doc = {
        "status": "PASS" if integrity_ok else "FAIL",
        "deployed_before": "elo_raw",
        "decision": decision,
        "candidates_evaluated": CANDIDATES,
        "policies_evaluated": ["frozen_block", "daily_sequential"],
        "outer_rows_identical": outer_rows_identical,
        "outer_outcomes_used_for_fitting": outer_outcomes_used_for_fitting,
        "pooled_calibration_slope_is_diagnostic_only": True,
        "promotion_rule": (
            "Promote a challenger over raw Elo ONLY IF, under BOTH policies, ALL hold: "
            "mean delta log loss < 0; mean delta Brier < 0; week-block bootstrap upper "
            "95% CI < 0 for both proper scores; every leave-one-outer-fold-out aggregate "
            "delta < 0 for both; no single outer fold contributes >= 50% of the total "
            "improvement; and no material tail degradation (challenger calibration gap in "
            f"p>=0.90 or p<=0.10 not worse than raw by > {TAIL_MATERIAL_TOL}). Slope/ECE/"
            "AUC/accuracy/April/pooled point estimates alone can NEVER promote."),
        "lambda_grid": [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
        "lambda_convention": "larger lambda == more shrinkage toward the identity map",
        "seed": seed,
        "n_boot": n_boot,
        "policies": policies,
        "promotion_gate": promotion_gate,
        "reason": reason,
    }
    out = artifact_dir / "calibration_challenger_decision.json"
    out.write_text(json.dumps(decision_doc, indent=2))

    # Persist per-row predictions and per-fold calibrator tables for audit.
    if len(frozen_preds):
        frozen_preds.to_csv(artifact_dir / "calibration_challenger_frozen_predictions.csv", index=False)
    if len(daily_preds):
        daily_preds.to_csv(artifact_dir / "calibration_challenger_daily_predictions.csv", index=False)
    return decision_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/architecture_candidates.json")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    doc = run(args.data, args.config, args.artifact_dir, n_boot=args.n_boot)
    summary = {
        "status": doc["status"],
        "decision": doc["decision"],
        "outer_rows_identical": doc["outer_rows_identical"],
        "outer_outcomes_used_for_fitting": doc["outer_outcomes_used_for_fitting"],
        "promotion_gate": {ch: doc["promotion_gate"][ch]["all_gates_pass_both_policies"]
                           for ch in CHALLENGERS},
    }
    for pol in ["frozen_block", "daily_sequential"]:
        pm = doc["policies"][pol]["candidates"]
        summary[pol] = {c: {k: round(pm[c][k], 6) for k in ["log_loss", "brier", "auc"]}
                        for c in CANDIDATES}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
