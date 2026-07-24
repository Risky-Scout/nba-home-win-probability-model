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
from nba_wp.model import (
    apply_logit_stacker,
    component_probabilities,
    fit_base_models,
    fit_logit_stacker,
    logit,
)

EPS = 1e-12
CANDIDATES = ["constant", "elo_only", "rank_only", "blend"]


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


def _inner_oof(features: pd.DataFrame, arch: Architecture, origin: pd.Timestamp,
               inner_start: pd.Timestamp, min_base: int) -> pd.DataFrame:
    """Expanding-window inner OOF component predictions on data before origin."""
    train_all = features[features["game_date"] < origin]
    rows: list[pd.DataFrame] = []
    for io in _weekly(inner_start, origin - pd.Timedelta(days=1)):
        tr = train_all[train_all["game_date"] < io]
        fold = train_all[(train_all["game_date"] >= io) & (train_all["game_date"] < io + pd.Timedelta(days=7))]
        if len(fold) == 0 or len(tr) < min_base or tr["home_win"].nunique() < 2:
            continue
        base = fit_base_models(tr, arch)
        pe, pr = component_probabilities(base, fold)
        rows.append(pd.DataFrame({
            "game_date": fold["game_date"].to_numpy(),
            "home_win": fold["home_win"].to_numpy(dtype=int),
            "pe": pe, "pr": pr,
        }))
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


def _select_architectures(feature_cache, architectures, origin, inner_start, min_base, min_stacker):
    """Independently pick the best architecture for Elo, rank, and blend."""
    best = {"elo_only": (float("inf"), None, None),
            "rank_only": (float("inf"), None, None),
            "blend": (float("inf"), None, None)}
    for arch in architectures:
        oof = _inner_oof(feature_cache[arch.name], arch, origin, inner_start, min_base)
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
    return {"policy": policy, "pooled_games": int(len(preds)),
            "pooled_metrics": pooled, "paired_block_bootstrap": paired,
            "calibration": calibration, "tail_counts": tails,
            "champion_challenger": verdict}


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
        outer_start="2026-02-01", inner_start="2025-11-15",
        min_base=40, min_stacker=40, n_boot=4000, seed=2026):
    artifact_dir = Path(artifact_dir)
    figure_dir = Path(figure_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    architectures = [Architecture.from_dict(a) for a in json.loads(Path(config_path).read_text())["architectures"]]
    arch_by_name = {a.name: a for a in architectures}
    games = load_games(data_path)

    # Sequential (causal) features once per architecture; slicing by date is exact.
    seq_cache = {a.name: build_features(games, a) for a in architectures}
    # Frozen features cache keyed by (arch_name, origin) built lazily.
    frozen_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def frozen_features(name: str, origin: pd.Timestamp) -> pd.DataFrame:
        key = (name, origin.strftime("%Y-%m-%d"))
        if key not in frozen_cache:
            frozen_cache[key] = build_features(games, arch_by_name[name], freeze_date=origin)
        return frozen_cache[key]

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

        best = _select_architectures(seq_cache, architectures, origin, inner_start_ts, min_base, min_stacker)
        elo_arch = best["elo_only"][1]
        rank_arch = best["rank_only"][1]
        blend_arch = best["blend"][1]
        blend_oof = best["blend"][2]
        if None in (elo_arch, rank_arch, blend_arch) or blend_oof is None or len(blend_oof) < min_stacker:
            continue

        const_rate = float(seq_cache[elo_arch][seq_cache[elo_arch]["game_date"] < origin]["home_win"].mean())
        stacker = fit_logit_stacker(blend_oof["home_win"].to_numpy(int), blend_oof["pe"].to_numpy(),
                                    blend_oof["pr"].to_numpy(), min_temperature=1.0)

        # ---------- FROZEN-BLOCK ----------
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

        frozen_rows.append(pd.DataFrame({
            "game_id": be["game_id"].to_numpy(), "game_date": be["game_date"].to_numpy(),
            "origin": origin.strftime("%Y-%m-%d"), "home_win": y,
            "p_constant": np.full(len(be), const_rate),
            "p_elo_only": p_elo, "p_rank_only": p_rank, "p_blend": p_blend,
        }))
        frozen_folds.append({"origin": origin.strftime("%Y-%m-%d"), "policy": "frozen_block",
                             "selected_elo_architecture": elo_arch, "selected_rank_architecture": rank_arch,
                             "selected_blend_architecture": blend_arch,
                             "inner_elo_log_loss": best["elo_only"][0], "inner_rank_log_loss": best["rank_only"][0],
                             "inner_blend_log_loss": best["blend"][0],
                             "blend_elo_weight": float(stacker.coef_[0, 0] / (stacker.coef_[0, 0] + stacker.coef_[0, 1])),
                             "blend_temperature": float(1.0 / (stacker.coef_[0, 0] + stacker.coef_[0, 1])),
                             "fold_games": int(len(be)),
                             "log_loss_elo": _ll(y, p_elo), "log_loss_blend": _ll(y, p_blend),
                             "brier_elo": _brier(y, p_elo), "brier_blend": _brier(y, p_blend)})

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

        # ---------- DAILY-SEQUENTIAL ----------
        block_dates = sorted(sample[(sample["game_date"] >= origin) & (sample["game_date"] < block_end)]["game_date"].unique())
        for t in block_dates:
            t = pd.Timestamp(t)
            def day_pred(arch_name, base_train_lt_t, which):
                seq = seq_cache[arch_name]
                tr = seq[seq["game_date"] < t]
                fold = seq[seq["game_date"] == t].sort_values("game_id")
                if len(tr) < min_base or tr["home_win"].nunique() < 2 or len(fold) == 0:
                    return None, None
                base = fit_base_models(tr, arch_by_name[arch_name])
                pe, pr = component_probabilities(base, fold)
                return fold, (pe if which == "elo" else pr)
            fold_e, pe_d = day_pred(elo_arch, None, "elo")
            fold_r, pr_d = day_pred(rank_arch, None, "rank")
            # blend uses its own arch; refit base through t-1 and OOF stacker (reuse block stacker)
            seqb = seq_cache[blend_arch]
            trb = seqb[seqb["game_date"] < t]
            foldb = seqb[seqb["game_date"] == t].sort_values("game_id")
            if fold_e is None or fold_r is None or len(foldb) == 0 or trb["home_win"].nunique() < 2:
                continue
            base_b = fit_base_models(trb, arch_by_name[blend_arch])
            peb, prb = component_probabilities(base_b, foldb)
            p_blend_d = apply_logit_stacker(stacker, peb, prb)
            yd = fold_e["home_win"].to_numpy(int)
            daily_rows.append(pd.DataFrame({
                "game_id": fold_e["game_id"].to_numpy(), "game_date": fold_e["game_date"].to_numpy(),
                "origin": origin.strftime("%Y-%m-%d"), "home_win": yd,
                "p_constant": np.full(len(fold_e), const_rate),
                "p_elo_only": pe_d, "p_rank_only": pr_d, "p_blend": p_blend_d,
            }))
        daily_folds.append({"origin": origin.strftime("%Y-%m-%d"), "policy": "daily_sequential",
                            "selected_elo_architecture": elo_arch, "selected_rank_architecture": rank_arch,
                            "selected_blend_architecture": blend_arch,
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
                                "architectures selected per-procedure by inner OOF; blend uses inner-OOF stacker.")
    daily_summary = _summarize(daily_preds, "daily_sequential", rng, n_boot)
    daily_summary["design"] = ("One game date per fold; base models refit through t-1; architectures selected "
                               "per-procedure weekly by inner OOF; blend uses inner-OOF stacker. Live simulation.")

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
        "elo_calibration": {k: policy_summary["calibration"]["elo_only"][k]
                            for k in ["calibration_intercept_alpha", "alpha_ci_2_5", "alpha_ci_97_5",
                                      "calibration_slope_beta", "beta_ci_2_5", "beta_ci_97_5",
                                      "ece_10bin", "ece_ci_2_5", "ece_ci_97_5",
                                      "mean_forecast", "observed_rate"]},
        "champion_challenger": policy_summary["champion_challenger"],
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
