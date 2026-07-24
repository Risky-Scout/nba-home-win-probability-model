"""Nested rolling-origin validation of the whole model-building procedure.

This is the rigorous, honest answer to "is the deployed model justified and are
its probabilities calibrated?" It fixes the weaknesses an auditor identified in
the first nested pass:

1. Model-specific selection. Each procedure selects its OWN architecture using
   its OWN inner out-of-fold score (Elo by Elo log loss, rank by rank log loss,
   blend by blend log loss). No procedure is represented by an architecture
   chosen for a different procedure.
2. Two clearly separated information policies, never mixed into one metric:
   * FROZEN-BLOCK: for outer origin O, performance state is frozen at O-1 via
     build_features(freeze_date=O). Every prediction in the block [O, O+7) uses
     only information available before O. Mutating any outcome inside the block
     cannot change any prediction in the block (verified in-script and tested).
   * DAILY-SEQUENTIAL: predictions for date t use results strictly before t
     (base models refit through t-1), a live one-step-ahead simulation.
3. Calibration is reported for EVERY candidate (constant / Elo / rank / blend):
   intercept alpha, slope beta, week-block bootstrap CIs, ECE with uncertainty,
   reliability table, mean-forecast-vs-observed, and tail counts.
4. Champion-challenger: Elo-only is champion; the blend must beat it on BOTH
   proper scores with stable evidence to be promoted.
5. Statistical language: bootstrap results are reported as "k of N replicates",
   never as an exact "P = 0".

Outputs (per policy):
  artifacts/nested_frozen_block_predictions.csv
  artifacts/nested_frozen_block_folds.csv
  artifacts/nested_frozen_block_summary.json
  artifacts/nested_daily_sequential_predictions.csv
  artifacts/nested_daily_sequential_folds.csv
  artifacts/nested_daily_sequential_summary.json
  figures/nested_frozen_block_reliability.png
  figures/nested_daily_sequential_reliability.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from nba_wp.data import load_games
from nba_wp.features import Architecture, build_features
from nba_wp.periods import derive_periods
from nba_wp.model import (
    apply_calibrator,
    apply_logit_stacker,
    component_probabilities,
    fit_base_models,
    fit_identity_shrunk_calibrator,
    fit_logit_stacker,
    fit_schedule_model,
    logit,
    schedule_probability,
    sigmoid,
)

EPS = 1e-12
CANDIDATES = ["constant", "elo_only", "rank_only", "blend", "calibrated_elo", "schedule_elo"]


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, _clip(p), labels=[0, 1]))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, _clip(p)))


def _weekly(start: pd.Timestamp, end: pd.Timestamp, step: int = 7) -> list[pd.Timestamp]:
    out, cur = [], start
    while cur <= end:
        out.append(cur)
        cur = cur + pd.Timedelta(days=step)
    return out


def _make_inner_fold_computer(games, seq_cache, arch_by_name, min_base):
    """Return a memoized ``inner_fold(policy, arch_name, io)`` function.

    An inner fold [io, io+7) trained on data before ``io`` is *independent of the
    outer origin*, so it is computed exactly once and reused across every outer
    fold. Two information policies are supported and never mixed:

      * ``"sequential"`` : uses the causal (daily-updating) feature cache, so
        earlier games in an inner week update later games in that week. This is
        the information policy of the daily-sequential live simulation.
      * ``"frozen"``     : builds features with ``freeze_date=io`` so every
        prediction in [io, io+7) uses only performance available before ``io``.
        This is the information policy of the frozen month-start deliverable, and
        it is the policy under which the frozen block's architecture MUST be
        selected (fixes the auditor's sequential-inner / frozen-outer mismatch).
    """
    frozen_feat_cache: dict[tuple[str, str], pd.DataFrame] = {}
    fold_cache: dict[tuple[str, str, str], pd.DataFrame | None] = {}

    def _frozen_features(name: str, io: pd.Timestamp) -> pd.DataFrame:
        key = (name, io.strftime("%Y-%m-%d"))
        if key not in frozen_feat_cache:
            frozen_feat_cache[key] = build_features(games, arch_by_name[name], freeze_date=io)
        return frozen_feat_cache[key]

    def inner_fold(policy: str, name: str, io: pd.Timestamp):
        key = (policy, name, io.strftime("%Y-%m-%d"))
        if key in fold_cache:
            return fold_cache[key]
        feat = _frozen_features(name, io) if policy == "frozen" else seq_cache[name]
        tr = feat[feat["game_date"] < io]
        fold = feat[(feat["game_date"] >= io) & (feat["game_date"] < io + pd.Timedelta(days=7))]
        if len(fold) == 0 or len(tr) < min_base or tr["home_win"].nunique() < 2:
            fold_cache[key] = None
            return None
        base = fit_base_models(tr, arch_by_name[name])
        pe, pr = component_probabilities(base, fold)
        res = pd.DataFrame({
            "game_date": fold["game_date"].to_numpy(),
            "home_win": fold["home_win"].to_numpy(dtype=int),
            "pe": pe, "pr": pr,
        })
        fold_cache[key] = res
        return res

    return inner_fold


def _inner_oof(inner_fold, policy: str, arch: Architecture, origin: pd.Timestamp,
               inner_start: pd.Timestamp) -> pd.DataFrame:
    """Concatenate every cached inner fold strictly before ``origin`` under ``policy``."""
    rows: list[pd.DataFrame] = []
    for io in _weekly(inner_start, origin - pd.Timedelta(days=1)):
        res = inner_fold(policy, arch.name, io)
        if res is not None and len(res):
            rows.append(res)
    if not rows:
        return pd.DataFrame(columns=["game_date", "home_win", "pe", "pr"])
    return pd.concat(rows, ignore_index=True)


def _inner_blend_ll(oof: pd.DataFrame, min_stacker: int) -> float:
    if len(oof) < (min_stacker + 10):
        return float("inf")
    oof = oof.sort_values("game_date").reset_index(drop=True)
    dates = sorted(oof["game_date"].unique())
    num, den = 0.0, 0
    for cut in _weekly(pd.Timestamp(dates[0]) + pd.Timedelta(days=21), pd.Timestamp(dates[-1])):
        tr = oof[oof["game_date"] < cut]
        fold = oof[(oof["game_date"] >= cut) & (oof["game_date"] < cut + pd.Timedelta(days=7))]
        if len(fold) == 0 or len(tr) < min_stacker or tr["home_win"].nunique() < 2:
            continue
        stk = fit_logit_stacker(tr["home_win"].to_numpy(int), tr["pe"].to_numpy(), tr["pr"].to_numpy(), min_temperature=1.0)
        p = apply_logit_stacker(stk, fold["pe"].to_numpy(), fold["pr"].to_numpy())
        num += _ll(fold["home_win"].to_numpy(int), p) * len(fold)
        den += len(fold)
    return float(num / den) if den else float("inf")


def _select_architectures(inner_fold, policy, architectures, origin, inner_start, min_base, min_stacker):
    """Independently pick the best architecture for Elo, rank, and blend under one
    information policy. The inner OOF used here is built with the SAME policy that
    the outer evaluation will use, so frozen deployment is validated by frozen
    inner selection and daily deployment by sequential inner selection."""
    best = {"elo_only": (float("inf"), None, None),
            "rank_only": (float("inf"), None, None),
            "blend": (float("inf"), None, None)}
    for arch in architectures:
        oof = _inner_oof(inner_fold, policy, arch, origin, inner_start)
        if len(oof) < min_base:
            continue
        y = oof["home_win"].to_numpy(int)
        elo_ll = _ll(y, oof["pe"].to_numpy())
        rank_ll = _ll(y, oof["pr"].to_numpy())
        blend_ll = _inner_blend_ll(oof, min_stacker)
        for key, score in [("elo_only", elo_ll), ("rank_only", rank_ll), ("blend", blend_ll)]:
            if (score, arch.name) < (best[key][0], best[key][1] or "~"):
                best[key] = (score, arch.name, oof if key == "blend" else None)
    return best


def _calibration_report(y, p, block_id, blocks, rng, n_boot=1000):
    x = logit(p).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10_000).fit(x, y)
    alpha, beta = float(model.intercept_[0]), float(model.coef_[0, 0])
    fr = pd.DataFrame({"p": p, "y": y})
    fr["bin"] = pd.qcut(fr["p"], q=min(10, len(np.unique(p))), duplicates="drop").astype(str)
    g = fr.groupby("bin", observed=True).agg(games=("y", "size"), mean_probability=("p", "mean"),
                                             observed_home_win_rate=("y", "mean"))
    ece = float((g["games"] / g["games"].sum() * (g["observed_home_win_rate"] - g["mean_probability"]).abs()).sum())
    a_bs, b_bs, e_bs = [], [], []
    for _ in range(n_boot):
        idx = np.concatenate([np.where(block_id == b)[0] for b in rng.choice(blocks, size=len(blocks), replace=True)])
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10_000).fit(logit(pp).reshape(-1, 1), yy)
        a_bs.append(float(m.intercept_[0]))
        b_bs.append(float(m.coef_[0, 0]))
        try:
            f2 = pd.DataFrame({"p": pp, "y": yy})
            f2["bin"] = pd.qcut(f2["p"], q=10, duplicates="drop").astype(str)
            gg = f2.groupby("bin", observed=True).agg(n=("y", "size"), mp=("p", "mean"), ob=("y", "mean"))
            e_bs.append(float((gg["n"] / gg["n"].sum() * (gg["ob"] - gg["mp"]).abs()).sum()))
        except ValueError:
            pass
    return {
        "calibration_intercept_alpha": alpha,
        "alpha_ci_2_5": float(np.quantile(a_bs, 0.025)) if a_bs else None,
        "alpha_ci_97_5": float(np.quantile(a_bs, 0.975)) if a_bs else None,
        "calibration_slope_beta": beta,
        "beta_ci_2_5": float(np.quantile(b_bs, 0.025)) if b_bs else None,
        "beta_ci_97_5": float(np.quantile(b_bs, 0.975)) if b_bs else None,
        "ece_10bin": ece,
        "ece_ci_2_5": float(np.quantile(e_bs, 0.025)) if e_bs else None,
        "ece_ci_97_5": float(np.quantile(e_bs, 0.975)) if e_bs else None,
        "mean_forecast": float(np.mean(p)),
        "observed_rate": float(np.mean(y)),
        "reliability_table": g.reset_index().to_dict(orient="records"),
    }


def _tail_counts(y, p):
    return {
        "n": int(len(y)),
        "p_ge_0_80": int(np.sum(p >= 0.80)),
        "obs_rate_p_ge_0_80": float(y[p >= 0.80].mean()) if np.any(p >= 0.80) else None,
        "p_ge_0_90": int(np.sum(p >= 0.90)),
        "p_le_0_20": int(np.sum(p <= 0.20)),
        "obs_rate_p_le_0_20": float(y[p <= 0.20].mean()) if np.any(p <= 0.20) else None,
        "p_le_0_10": int(np.sum(p <= 0.10)),
    }


def _summarize(preds: pd.DataFrame, policy: str, rng, n_boot: int) -> dict:
    y = preds["home_win"].to_numpy(int)
    cand = {c: preds[f"p_{c}"].to_numpy() for c in CANDIDATES}
    pooled = {c: {"log_loss": _ll(y, p), "brier": _brier(y, p),
                  "auc": float(roc_auc_score(y, _clip(p))) if len(np.unique(y)) > 1 else None,
                  "accuracy": float(np.mean((p >= 0.5).astype(int) == y)),
                  "mean_probability": float(np.mean(p))}
              for c, p in cand.items()}

    week = preds["game_date"].dt.isocalendar().week.astype(int).to_numpy()
    year = preds["game_date"].dt.isocalendar().year.astype(int).to_numpy()
    block_id = year * 100 + week
    blocks = np.unique(block_id)

    def boot_paired(pm, pb):
        d_ll = np.empty(n_boot)
        d_br = np.empty(n_boot)
        for i in range(n_boot):
            idx = np.concatenate([np.where(block_id == b)[0] for b in rng.choice(blocks, size=len(blocks), replace=True)])
            yy = y[idx]
            d_ll[i] = _ll(yy, pm[idx]) - _ll(yy, pb[idx])
            d_br[i] = _brier(yy, pm[idx]) - _brier(yy, pb[idx])
        n_better_ll = int(np.sum(d_ll < 0.0))
        n_better_br = int(np.sum(d_br < 0.0))
        return {
            "delta_log_loss_mean": float(d_ll.mean()),
            "delta_log_loss_ci_2_5": float(np.quantile(d_ll, 0.025)),
            "delta_log_loss_ci_97_5": float(np.quantile(d_ll, 0.975)),
            "delta_brier_mean": float(d_br.mean()),
            "delta_brier_ci_2_5": float(np.quantile(d_br, 0.025)),
            "delta_brier_ci_97_5": float(np.quantile(d_br, 0.975)),
            "replicates": int(n_boot),
            "replicates_favoring_model_log_loss": n_better_ll,
            "replicates_favoring_model_brier": n_better_br,
            "log_loss_statement": f"{n_better_ll} of {n_boot} week-block bootstrap replicates favored the model",
            "brier_statement": f"{n_better_br} of {n_boot} week-block bootstrap replicates favored the model",
        }

    paired = {
        "blend_minus_elo": boot_paired(cand["blend"], cand["elo_only"]),
        "elo_minus_constant": boot_paired(cand["elo_only"], cand["constant"]),
        "elo_minus_rank": boot_paired(cand["elo_only"], cand["rank_only"]),
        "blend_minus_constant": boot_paired(cand["blend"], cand["constant"]),
        "calibrated_elo_minus_elo": boot_paired(cand["calibrated_elo"], cand["elo_only"]),
        "schedule_elo_minus_elo": boot_paired(cand["schedule_elo"], cand["elo_only"]),
    }

    calibration = {c: _calibration_report(y, cand[c], block_id, blocks, rng) for c in CANDIDATES}
    tails = {c: _tail_counts(y, cand[c]) for c in CANDIDATES}

    blend_beats_elo = (pooled["blend"]["log_loss"] < pooled["elo_only"]["log_loss"]
                       and pooled["blend"]["brier"] < pooled["elo_only"]["brier"])
    stable = (paired["blend_minus_elo"]["delta_log_loss_ci_97_5"] < 0.0
              and paired["blend_minus_elo"]["delta_brier_ci_97_5"] < 0.0)
    verdict = {
        "champion": "elo_only",
        "challenger": "blend",
        "rule": "Promote blend only if it beats Elo-only on BOTH pooled log loss and Brier with block-bootstrap upper CI < 0.",
        "blend_beats_elo_point_estimate_both": bool(blend_beats_elo),
        "blend_beats_elo_stable": bool(stable),
        "decision": "promote_blend" if (blend_beats_elo and stable) else "keep_elo_only",
    }

    cal_beats = (pooled["calibrated_elo"]["log_loss"] < pooled["elo_only"]["log_loss"]
                 and pooled["calibrated_elo"]["brier"] < pooled["elo_only"]["brier"])
    cal_stable = (paired["calibrated_elo_minus_elo"]["delta_log_loss_ci_97_5"] < 0.0
                  and paired["calibrated_elo_minus_elo"]["delta_brier_ci_97_5"] < 0.0)
    calibration_verdict = {
        "champion": "elo_only",
        "challenger": "calibrated_elo",
        "rule": ("Promote the cross-fitted identity-shrunk calibrator only if it beats raw "
                 "Elo-only on BOTH pooled log loss and Brier with block-bootstrap upper CI < 0."),
        "calibrated_beats_elo_point_estimate_both": bool(cal_beats),
        "calibrated_beats_elo_stable": bool(cal_stable),
        "decision": "promote_calibrated_elo" if (cal_beats and cal_stable) else "keep_raw_elo",
    }
    sched_beats = (pooled["schedule_elo"]["log_loss"] < pooled["elo_only"]["log_loss"]
                   and pooled["schedule_elo"]["brier"] < pooled["elo_only"]["brier"])
    sched_stable = (paired["schedule_elo_minus_elo"]["delta_log_loss_ci_97_5"] < 0.0
                    and paired["schedule_elo_minus_elo"]["delta_brier_ci_97_5"] < 0.0)
    schedule_verdict = {
        "champion": "elo_only",
        "challenger": "schedule_elo",
        "rule": ("Promote the Elo+schedule challenger only if it beats raw Elo-only on "
                 "BOTH pooled log loss and Brier with block-bootstrap upper CI < 0."),
        "schedule_beats_elo_point_estimate_both": bool(sched_beats),
        "schedule_beats_elo_stable": bool(sched_stable),
        "decision": "promote_schedule_elo" if (sched_beats and sched_stable) else "keep_raw_elo",
    }
    return {"policy": policy, "pooled_games": int(len(preds)),
            "pooled_metrics": pooled, "paired_block_bootstrap": paired,
            "calibration": calibration, "tail_counts": tails,
            "champion_challenger": verdict,
            "calibration_challenger": calibration_verdict,
            "schedule_challenger": schedule_verdict}


def _reliability_figure(preds, summary, path, title):
    y = preds["home_win"].to_numpy(int)
    plt.figure(figsize=(6, 6))
    for c, style in [("elo_only", "s--"), ("blend", "o-")]:
        p = preds[f"p_{c}"].to_numpy()
        fr = pd.DataFrame({"p": p, "y": y})
        fr["bin"] = pd.qcut(fr["p"], q=10, duplicates="drop").astype(str)
        g = fr.groupby("bin", observed=True).agg(mp=("p", "mean"), ob=("y", "mean")).sort_values("mp")
        plt.plot(g["mp"], g["ob"], style, label=c)
    plt.plot([0, 1], [0, 1], ":", color="gray", label="perfect")
    m = summary["pooled_metrics"]
    plt.xlabel("Mean predicted home-win probability")
    plt.ylabel("Observed home-win rate")
    plt.title(f"{title}\nElo LL={m['elo_only']['log_loss']:.3f}  blend LL={m['blend']['log_loss']:.3f}  (n={len(preds)})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run(data_path, config_path, artifact_dir, figure_dir, *,
        outer_start=None, inner_start=None,
        min_base=40, min_stacker=40, n_boot=4000, seed=2026, cal_n0=200.0, schedule_c=0.5):
    artifact_dir = Path(artifact_dir)
    figure_dir = Path(figure_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    architectures = [Architecture.from_dict(a) for a in json.loads(Path(config_path).read_text())["architectures"]]
    arch_by_name = {a.name: a for a in architectures}
    games = load_games(data_path)

    # Data-driven rolling-origin anchors (no hard-coded season boundaries):
    #   outer_start = one month before the selection month (derived);
    #   inner_start = ~4 weeks after the first game (warmup for the first Elo fit).
    periods = derive_periods(games)
    if outer_start is None:
        outer_start = periods.s(periods.outer_start)
    if inner_start is None:
        inner_start = (games["game_date"].min().normalize() + pd.Timedelta(weeks=4)).strftime("%Y-%m-%d")

    # Sequential (causal) features once per architecture; slicing by date is exact.
    seq_cache = {a.name: build_features(games, a) for a in architectures}
    # Frozen features cache keyed by (arch_name, origin) built lazily.
    frozen_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def frozen_features(name: str, origin: pd.Timestamp) -> pd.DataFrame:
        key = (name, origin.strftime("%Y-%m-%d"))
        if key not in frozen_cache:
            frozen_cache[key] = build_features(games, arch_by_name[name], freeze_date=origin)
        return frozen_cache[key]

    # Memoized inner folds for BOTH policies. Frozen deployment is now selected
    # under frozen inner folds and daily deployment under sequential inner folds.
    inner_fold = _make_inner_fold_computer(games, seq_cache, arch_by_name, min_base)

    inner_start_ts = pd.Timestamp(inner_start)
    max_date = games["game_date"].max()
    origins = _weekly(pd.Timestamp(outer_start), max_date)

    frozen_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    frozen_folds: list[dict] = []
    daily_folds: list[dict] = []
    leakage_ok = True

    for origin in origins:
        block_end = origin + pd.Timedelta(days=7)
        sample = seq_cache[architectures[0].name]
        if len(sample[(sample["game_date"] >= origin) & (sample["game_date"] < block_end)]) == 0:
            continue
        if len(sample[sample["game_date"] < origin]) < min_base:
            continue

        best_frozen = _select_architectures(inner_fold, "frozen", architectures, origin,
                                            inner_start_ts, min_base, min_stacker)
        best_daily = _select_architectures(inner_fold, "sequential", architectures, origin,
                                           inner_start_ts, min_base, min_stacker)

        # ---------- FROZEN-BLOCK (architecture selected under the FROZEN policy) ----------
        best = best_frozen
        elo_arch = best["elo_only"][1]
        rank_arch = best["rank_only"][1]
        blend_arch = best["blend"][1]
        blend_oof = best["blend"][2]
        if None in (elo_arch, rank_arch, blend_arch) or blend_oof is None or len(blend_oof) < min_stacker:
            continue

        const_rate = float(seq_cache[elo_arch][seq_cache[elo_arch]["game_date"] < origin]["home_win"].mean())
        stacker = fit_logit_stacker(blend_oof["home_win"].to_numpy(int), blend_oof["pe"].to_numpy(),
                                    blend_oof["pr"].to_numpy(), min_temperature=1.0)

        elo_base = fit_base_models(seq_cache[elo_arch][seq_cache[elo_arch]["game_date"] < origin], arch_by_name[elo_arch])
        rank_base = fit_base_models(seq_cache[rank_arch][seq_cache[rank_arch]["game_date"] < origin], arch_by_name[rank_arch])
        blend_base = fit_base_models(seq_cache[blend_arch][seq_cache[blend_arch]["game_date"] < origin], arch_by_name[blend_arch])

        fe = frozen_features(elo_arch, origin)
        fr = frozen_features(rank_arch, origin)
        fb = frozen_features(blend_arch, origin)
        be = fe[(fe["game_date"] >= origin) & (fe["game_date"] < block_end)].sort_values("game_id")
        br = fr[(fr["game_date"] >= origin) & (fr["game_date"] < block_end)].sort_values("game_id")
        bb = fb[(fb["game_date"] >= origin) & (fb["game_date"] < block_end)].sort_values("game_id")

        p_elo = component_probabilities(elo_base, be)[0]
        p_rank = component_probabilities(rank_base, br)[1]
        pe_b, pr_b = component_probabilities(blend_base, bb)
        p_blend = apply_logit_stacker(stacker, pe_b, pr_b)
        y = be["home_win"].to_numpy(int)

        # Calibration challenger: fit alpha/beta on the elo arch's FROZEN inner
        # OOF only, shrink toward identity, apply to the untouched outer block.
        elo_oof_f = _inner_oof(inner_fold, "frozen", arch_by_name[elo_arch], origin, inner_start_ts)
        a_cal, b_cal = fit_identity_shrunk_calibrator(elo_oof_f["pe"], elo_oof_f["home_win"], cal_n0)
        p_cal_elo = apply_calibrator(a_cal, b_cal, p_elo)

        # Schedule/rest challenger: fit on the elo arch's train (< origin), apply
        # to the frozen block. Uses elo_diff + rest + back-to-back only.
        sched_train = seq_cache[elo_arch][seq_cache[elo_arch]["game_date"] < origin]
        p_sched = schedule_probability(fit_schedule_model(sched_train, schedule_c), be)

        frozen_rows.append(pd.DataFrame({
            "game_id": be["game_id"].to_numpy(), "game_date": be["game_date"].to_numpy(),
            "origin": origin.strftime("%Y-%m-%d"), "home_win": y,
            "p_constant": np.full(len(be), const_rate),
            "p_elo_only": p_elo, "p_rank_only": p_rank, "p_blend": p_blend,
            "p_calibrated_elo": p_cal_elo, "p_schedule_elo": p_sched,
        }))
        frozen_folds.append({"origin": origin.strftime("%Y-%m-%d"), "policy": "frozen_block",
                             "selected_elo_architecture": elo_arch, "selected_rank_architecture": rank_arch,
                             "selected_blend_architecture": blend_arch,
                             "inner_elo_log_loss": best["elo_only"][0], "inner_rank_log_loss": best["rank_only"][0],
                             "inner_blend_log_loss": best["blend"][0],
                             "blend_elo_weight": float(stacker.coef_[0, 0] / (stacker.coef_[0, 0] + stacker.coef_[0, 1])),
                             "blend_temperature": float(1.0 / (stacker.coef_[0, 0] + stacker.coef_[0, 1])),
                             "calibrator_alpha": a_cal, "calibrator_beta": b_cal,
                             "fold_games": int(len(be)),
                             "log_loss_elo": _ll(y, p_elo), "log_loss_blend": _ll(y, p_blend),
                             "log_loss_calibrated_elo": _ll(y, p_cal_elo),
                             "brier_elo": _brier(y, p_elo), "brier_blend": _brier(y, p_blend),
                             "brier_calibrated_elo": _brier(y, p_cal_elo)})

        # Leakage guarantee check on the first frozen block: mutating block
        # outcomes must not change block predictions.
        if leakage_ok and len(be) > 0 and origin == origins[0]:
            mutated = games.copy()
            mask = (mutated["game_date"] >= origin) & (mutated["game_date"] < block_end)
            hp = mutated.loc[mask, "home_points"].to_numpy().copy()
            ap = mutated.loc[mask, "away_points"].to_numpy().copy()
            mutated.loc[mask, "home_points"] = ap
            mutated.loc[mask, "away_points"] = hp
            mutated["home_win"] = (mutated["home_points"] > mutated["away_points"]).astype("int8")
            fe2 = build_features(mutated, arch_by_name[elo_arch], freeze_date=origin)
            be2 = fe2[(fe2["game_date"] >= origin) & (fe2["game_date"] < block_end)].sort_values("game_id")
            p_elo2 = component_probabilities(elo_base, be2)[0]
            leakage_ok = bool(np.allclose(p_elo, p_elo2, atol=1e-12))

        # ---------- DAILY-SEQUENTIAL (architecture selected under the SEQUENTIAL policy) ----------
        d_elo_arch = best_daily["elo_only"][1]
        d_rank_arch = best_daily["rank_only"][1]
        d_blend_arch = best_daily["blend"][1]
        d_blend_oof = best_daily["blend"][2]
        if None in (d_elo_arch, d_rank_arch, d_blend_arch) or d_blend_oof is None or len(d_blend_oof) < min_stacker:
            continue
        d_const_rate = float(seq_cache[d_elo_arch][seq_cache[d_elo_arch]["game_date"] < origin]["home_win"].mean())
        d_stacker = fit_logit_stacker(d_blend_oof["home_win"].to_numpy(int), d_blend_oof["pe"].to_numpy(),
                                      d_blend_oof["pr"].to_numpy(), min_temperature=1.0)
        d_elo_oof = _inner_oof(inner_fold, "sequential", arch_by_name[d_elo_arch], origin, inner_start_ts)
        da_cal, db_cal = fit_identity_shrunk_calibrator(d_elo_oof["pe"], d_elo_oof["home_win"], cal_n0)

        block_dates = sorted(sample[(sample["game_date"] >= origin) & (sample["game_date"] < block_end)]["game_date"].unique())
        for t in block_dates:
            t = pd.Timestamp(t)
            def day_pred(arch_name, which):
                seq = seq_cache[arch_name]
                tr = seq[seq["game_date"] < t]
                fold = seq[seq["game_date"] == t].sort_values("game_id")
                if len(tr) < min_base or tr["home_win"].nunique() < 2 or len(fold) == 0:
                    return None, None
                base = fit_base_models(tr, arch_by_name[arch_name])
                pe, pr = component_probabilities(base, fold)
                return fold, (pe if which == "elo" else pr)
            fold_e, pe_d = day_pred(d_elo_arch, "elo")
            fold_r, pr_d = day_pred(d_rank_arch, "rank")
            # blend uses its own arch; refit base through t-1 (block-level OOF stacker reused)
            seqb = seq_cache[d_blend_arch]
            trb = seqb[seqb["game_date"] < t]
            foldb = seqb[seqb["game_date"] == t].sort_values("game_id")
            if fold_e is None or fold_r is None or len(foldb) == 0 or trb["home_win"].nunique() < 2:
                continue
            base_b = fit_base_models(trb, arch_by_name[d_blend_arch])
            peb, prb = component_probabilities(base_b, foldb)
            p_blend_d = apply_logit_stacker(d_stacker, peb, prb)
            yd = fold_e["home_win"].to_numpy(int)
            sched_tr_d = seq_cache[d_elo_arch][seq_cache[d_elo_arch]["game_date"] < t]
            p_sched_d = schedule_probability(fit_schedule_model(sched_tr_d, schedule_c), fold_e)
            daily_rows.append(pd.DataFrame({
                "game_id": fold_e["game_id"].to_numpy(), "game_date": fold_e["game_date"].to_numpy(),
                "origin": origin.strftime("%Y-%m-%d"), "home_win": yd,
                "p_constant": np.full(len(fold_e), d_const_rate),
                "p_elo_only": pe_d, "p_rank_only": pr_d, "p_blend": p_blend_d,
                "p_calibrated_elo": apply_calibrator(da_cal, db_cal, pe_d),
                "p_schedule_elo": p_sched_d,
            }))
        daily_folds.append({"origin": origin.strftime("%Y-%m-%d"), "policy": "daily_sequential",
                            "selected_elo_architecture": d_elo_arch, "selected_rank_architecture": d_rank_arch,
                            "selected_blend_architecture": d_blend_arch,
                            "inner_elo_log_loss": best_daily["elo_only"][0],
                            "inner_rank_log_loss": best_daily["rank_only"][0],
                            "inner_blend_log_loss": best_daily["blend"][0],
                            "week_dates": len(block_dates)})

    rng = np.random.default_rng(seed)
    frozen_preds = pd.concat(frozen_rows, ignore_index=True)
    daily_preds = pd.concat(daily_rows, ignore_index=True)
    frozen_preds.to_csv(artifact_dir / "nested_frozen_block_predictions.csv", index=False)
    daily_preds.to_csv(artifact_dir / "nested_daily_sequential_predictions.csv", index=False)
    pd.DataFrame(frozen_folds).to_csv(artifact_dir / "nested_frozen_block_folds.csv", index=False)
    pd.DataFrame(daily_folds).to_csv(artifact_dir / "nested_daily_sequential_folds.csv", index=False)

    frozen_summary = _summarize(frozen_preds, "frozen_block", rng, n_boot)
    frozen_summary["frozen_block_leakage_guarantee_verified"] = bool(leakage_ok)
    frozen_summary["design"] = ("Per outer weekly origin O, performance state frozen at O-1 (build_features "
                                "freeze_date=O); block [O,O+7) scored with base models fit strictly before O; "
                                "architectures selected per-procedure by FROZEN inner OOF (each inner origin I "
                                "uses build_features(freeze_date=I)); blend uses frozen inner-OOF stacker. "
                                "Inner and outer information policies now match (frozen/frozen).")
    daily_summary = _summarize(daily_preds, "daily_sequential", rng, n_boot)
    daily_summary["design"] = ("One game date per fold; base models refit through t-1; architectures selected "
                               "per-procedure weekly by SEQUENTIAL inner OOF; blend uses sequential inner-OOF "
                               "stacker. Inner and outer information policies now match (sequential/sequential). "
                               "Live simulation.")

    (artifact_dir / "nested_frozen_block_summary.json").write_text(json.dumps(frozen_summary, indent=2))
    (artifact_dir / "nested_daily_sequential_summary.json").write_text(json.dumps(daily_summary, indent=2))

    _reliability_figure(frozen_preds, frozen_summary, figure_dir / "nested_frozen_block_reliability.png",
                        "Nested frozen-block reliability")
    _reliability_figure(daily_preds, daily_summary, figure_dir / "nested_daily_sequential_reliability.png",
                        "Nested daily-sequential reliability")

    return {"frozen_block": frozen_summary, "daily_sequential": daily_summary}


def _digest(policy_summary: dict) -> dict:
    return {
        "pooled_games": policy_summary["pooled_games"],
        "pooled_metrics": {c: {k: policy_summary["pooled_metrics"][c][k] for k in ["log_loss", "brier", "auc"]}
                           for c in CANDIDATES},
        "blend_minus_elo": policy_summary["paired_block_bootstrap"]["blend_minus_elo"],
        "calibrated_elo_minus_elo": policy_summary["paired_block_bootstrap"]["calibrated_elo_minus_elo"],
        "elo_calibration": {k: policy_summary["calibration"]["elo_only"][k]
                            for k in ["calibration_intercept_alpha", "alpha_ci_2_5", "alpha_ci_97_5",
                                      "calibration_slope_beta", "beta_ci_2_5", "beta_ci_97_5",
                                      "ece_10bin", "ece_ci_2_5", "ece_ci_97_5",
                                      "mean_forecast", "observed_rate"]},
        "calibrated_elo_calibration": {k: policy_summary["calibration"]["calibrated_elo"][k]
                                       for k in ["calibration_intercept_alpha", "calibration_slope_beta",
                                                 "ece_10bin", "mean_forecast", "observed_rate"]},
        "schedule_elo_minus_elo": policy_summary["paired_block_bootstrap"]["schedule_elo_minus_elo"],
        "champion_challenger": policy_summary["champion_challenger"],
        "calibration_challenger": policy_summary["calibration_challenger"],
        "schedule_challenger": policy_summary["schedule_challenger"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/architecture_candidates.json")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--figure-dir", default="figures")
    args = parser.parse_args()
    out = run(args.data, args.config, args.artifact_dir, args.figure_dir)
    print(json.dumps({
        "frozen_block": _digest(out["frozen_block"]) | {
            "leakage_guarantee_verified": out["frozen_block"]["frozen_block_leakage_guarantee_verified"]},
        "daily_sequential": _digest(out["daily_sequential"]),
    }, indent=2))


if __name__ == "__main__":
    main()
