"""Nested rolling-origin validation of the whole model-building procedure.

This is the rigorous answer to "is the final blend justified and are the
probabilities calibrated?" It removes the two weaknesses of the simpler rolling
audit:

  1. Architecture selection is nested. Every outer fold selects its own
     architecture using ONLY data before that fold (via an inner rolling-origin
     search). No architecture chosen on full March is ever scored on March.
  2. The stacker is trained on chronological OUT-OF-FOLD component predictions
     (not in-sample base-model predictions), matching how it is deployed.

Procedure (per outer weekly fold with origin O):
    train_all = games with game_date < O
    for each candidate architecture A:
        inner rolling-origin over train_all -> pooled inner OOF component preds;
        score A by inner one-step-ahead blend log loss (constrained stacker fit
        on inner OOF strictly before each inner fold).
    A* = argmin inner blend log loss.
    Deploy A*:
        fit constrained stacker (T>=1) on ALL inner OOF component preds of A*;
        refit base Elo/rank models on all of train_all;
        score the untouched outer fold [O, O+7).
    Record, on identical outer-fold rows, four candidate prices:
        constant (train home-win rate), Elo-only, rank-only, constrained blend.

We then pool every outer fold and report:
  * candidate comparison (log loss + Brier);
  * week-block bootstrap paired intervals (model - baseline);
  * calibration regression logit(y) = a + b logit(p_hat) with block CIs;
  * ECE (10-bin) with block-bootstrap CI, plus tail counts;
  * a champion-challenger verdict (Elo champion; blend must beat Elo on BOTH
    log loss and Brier to be promoted).

Outputs:
  artifacts/nested_backtest_folds.csv
  artifacts/nested_backtest_predictions.csv
  artifacts/nested_backtest_summary.json
  figures/nested_backtest_reliability.png
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


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, _clip(p), labels=[0, 1]))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, _clip(p)))


def _weekly_origins(start: pd.Timestamp, end: pd.Timestamp, step_days: int = 7) -> list[pd.Timestamp]:
    origins, cursor = [], start
    while cursor <= end:
        origins.append(cursor)
        cursor = cursor + pd.Timedelta(days=step_days)
    return origins


def _inner_oof(
    features: pd.DataFrame,
    architecture: Architecture,
    origin: pd.Timestamp,
    *,
    inner_start: pd.Timestamp,
    min_base_games: int,
) -> pd.DataFrame:
    """Expanding-window inner OOF component predictions for train_all (< origin)."""
    train_all = features[features["game_date"] < origin]
    rows: list[pd.DataFrame] = []
    for io in _weekly_origins(inner_start, origin - pd.Timedelta(days=1)):
        inner_train = train_all[train_all["game_date"] < io]
        inner_fold = train_all[
            (train_all["game_date"] >= io) & (train_all["game_date"] < io + pd.Timedelta(days=7))
        ]
        if len(inner_fold) == 0 or len(inner_train) < min_base_games:
            continue
        if inner_train["home_win"].nunique() < 2:
            continue
        base = fit_base_models(inner_train, architecture)
        pe, pr = component_probabilities(base, inner_fold)
        rows.append(
            pd.DataFrame(
                {
                    "game_date": inner_fold["game_date"].to_numpy(),
                    "home_win": inner_fold["home_win"].to_numpy(dtype=int),
                    "pe": pe,
                    "pr": pr,
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["game_date", "home_win", "pe", "pr"])
    return pd.concat(rows, ignore_index=True)


def _inner_blend_logloss(oof: pd.DataFrame, *, min_stacker: int) -> float:
    """Score an architecture by inner one-step-ahead blend log loss.

    The constrained stacker for each inner fold is fit only on inner OOF rows
    strictly earlier than that fold, so this is a genuine inner OOS score.
    """
    if len(oof) < (min_stacker + 10):
        return float("inf")
    oof = oof.sort_values("game_date").reset_index(drop=True)
    dates = sorted(oof["game_date"].unique())
    losses: list[float] = []
    counts: list[int] = []
    for cut in _weekly_origins(pd.Timestamp(dates[0]) + pd.Timedelta(days=21), pd.Timestamp(dates[-1])):
        train = oof[oof["game_date"] < cut]
        fold = oof[(oof["game_date"] >= cut) & (oof["game_date"] < cut + pd.Timedelta(days=7))]
        if len(fold) == 0 or len(train) < min_stacker or train["home_win"].nunique() < 2:
            continue
        stacker = fit_logit_stacker(
            train["home_win"].to_numpy(dtype=int),
            train["pe"].to_numpy(),
            train["pr"].to_numpy(),
            min_temperature=1.0,
        )
        p = apply_logit_stacker(stacker, fold["pe"].to_numpy(), fold["pr"].to_numpy())
        losses.append(_ll(fold["home_win"].to_numpy(dtype=int), p) * len(fold))
        counts.append(len(fold))
    if not counts:
        return float("inf")
    return float(np.sum(losses) / np.sum(counts))


def run(
    data_path: str | Path,
    config_path: str | Path,
    artifact_dir: str | Path,
    figure_dir: str | Path,
    *,
    outer_start: str = "2026-02-01",
    inner_start: str = "2025-11-15",
    min_base_games: int = 40,
    min_stacker: int = 40,
    n_boot: int = 4000,
    seed: int = 2026,
) -> dict:
    artifact_dir = Path(artifact_dir)
    figure_dir = Path(figure_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    architectures = [
        Architecture.from_dict(a)
        for a in json.loads(Path(config_path).read_text())["architectures"]
    ]
    games = load_games(data_path)

    # Features are causal (game t uses only games < t), so building once per
    # architecture on the full season and slicing by date is exact.
    feature_cache = {arch.name: build_features(games, arch) for arch in architectures}

    max_date = games["game_date"].max()
    outer_origins = _weekly_origins(pd.Timestamp(outer_start), max_date)
    inner_start_ts = pd.Timestamp(inner_start)

    fold_records: list[dict] = []
    pooled: list[pd.DataFrame] = []

    for origin in outer_origins:
        any_feat = feature_cache[architectures[0].name]
        outer_fold_index = any_feat[
            (any_feat["game_date"] >= origin) & (any_feat["game_date"] < origin + pd.Timedelta(days=7))
        ]
        train_all_index = any_feat[any_feat["game_date"] < origin]
        if len(outer_fold_index) == 0 or len(train_all_index) < min_base_games:
            continue

        # --- Inner architecture selection (only data < origin) ---
        scored: list[tuple[float, str, pd.DataFrame]] = []
        for arch in architectures:
            oof = _inner_oof(
                feature_cache[arch.name],
                arch,
                origin,
                inner_start=inner_start_ts,
                min_base_games=min_base_games,
            )
            score = _inner_blend_logloss(oof, min_stacker=min_stacker)
            scored.append((score, arch.name, oof))
        scored.sort(key=lambda t: (t[0], t[1]))
        best_score, best_name, best_oof = scored[0]
        best_arch = next(a for a in architectures if a.name == best_name)

        # --- Deploy selected architecture on the untouched outer fold ---
        features = feature_cache[best_name]
        train_all = features[features["game_date"] < origin]
        outer_fold = features[
            (features["game_date"] >= origin) & (features["game_date"] < origin + pd.Timedelta(days=7))
        ]
        # Constrained stacker on OUT-OF-FOLD component preds (deployment match).
        stacker = fit_logit_stacker(
            best_oof["home_win"].to_numpy(dtype=int),
            best_oof["pe"].to_numpy(),
            best_oof["pr"].to_numpy(),
            min_temperature=1.0,
        )
        base = fit_base_models(train_all, best_arch)
        pe, pr = component_probabilities(base, outer_fold)
        p_blend = apply_logit_stacker(stacker, pe, pr)
        p_const = np.full(len(outer_fold), float(train_all["home_win"].mean()))

        y = outer_fold["home_win"].to_numpy(dtype=int)
        block = pd.DataFrame(
            {
                "game_id": outer_fold["game_id"].to_numpy(),
                "game_date": outer_fold["game_date"].to_numpy(),
                "origin": origin.strftime("%Y-%m-%d"),
                "selected_architecture": best_name,
                "home_win": y,
                "p_constant": p_const,
                "p_elo": pe,
                "p_rank": pr,
                "p_blend": p_blend,
            }
        )
        pooled.append(block)
        fold_records.append(
            {
                "origin": origin.strftime("%Y-%m-%d"),
                "selected_architecture": best_name,
                "inner_blend_log_loss": best_score,
                "train_games": int(len(train_all)),
                "fold_games": int(len(outer_fold)),
                "log_loss_constant": _ll(y, p_const),
                "log_loss_elo": _ll(y, pe),
                "log_loss_rank": _ll(y, pr),
                "log_loss_blend": _ll(y, p_blend),
                "brier_constant": _brier(y, p_const),
                "brier_elo": _brier(y, pe),
                "brier_rank": _brier(y, pr),
                "brier_blend": _brier(y, p_blend),
            }
        )

    preds = pd.concat(pooled, ignore_index=True)
    preds.to_csv(artifact_dir / "nested_backtest_predictions.csv", index=False)
    pd.DataFrame(fold_records).to_csv(artifact_dir / "nested_backtest_folds.csv", index=False)

    y = preds["home_win"].to_numpy(dtype=int)
    candidates = {
        "constant": preds["p_constant"].to_numpy(),
        "elo_only": preds["p_elo"].to_numpy(),
        "rank_only": preds["p_rank"].to_numpy(),
        "blend": preds["p_blend"].to_numpy(),
    }
    pooled_metrics = {
        name: {
            "log_loss": _ll(y, p),
            "brier": _brier(y, p),
            "auc": float(roc_auc_score(y, _clip(p))),
            "accuracy": float(np.mean((p >= 0.5).astype(int) == y)),
            "mean_probability": float(np.mean(p)),
        }
        for name, p in candidates.items()
    }

    # --- Week-block bootstrap of paired differences (model - baseline) ---
    week = preds["game_date"].dt.isocalendar().week.astype(int).to_numpy()
    year = preds["game_date"].dt.isocalendar().year.astype(int).to_numpy()
    block_id = year * 100 + week
    blocks = np.unique(block_id)
    rng = np.random.default_rng(seed)

    def _boot_paired(p_model: np.ndarray, p_base: np.ndarray) -> dict:
        d_ll = np.empty(n_boot)
        d_br = np.empty(n_boot)
        for i in range(n_boot):
            chosen = rng.choice(blocks, size=len(blocks), replace=True)
            idx = np.concatenate([np.where(block_id == b)[0] for b in chosen])
            yy = y[idx]
            d_ll[i] = _ll(yy, p_model[idx]) - _ll(yy, p_base[idx])
            d_br[i] = _brier(yy, p_model[idx]) - _brier(yy, p_base[idx])
        return {
            "delta_log_loss_mean": float(d_ll.mean()),
            "delta_log_loss_ci_2_5": float(np.quantile(d_ll, 0.025)),
            "delta_log_loss_ci_97_5": float(np.quantile(d_ll, 0.975)),
            "delta_brier_mean": float(d_br.mean()),
            "delta_brier_ci_2_5": float(np.quantile(d_br, 0.025)),
            "delta_brier_ci_97_5": float(np.quantile(d_br, 0.975)),
            "prob_model_better_log_loss": float(np.mean(d_ll < 0.0)),
            "prob_model_better_brier": float(np.mean(d_br < 0.0)),
        }

    paired = {
        "blend_minus_elo": _boot_paired(candidates["blend"], candidates["elo_only"]),
        "blend_minus_constant": _boot_paired(candidates["blend"], candidates["constant"]),
        "elo_minus_constant": _boot_paired(candidates["elo_only"], candidates["constant"]),
        "blend_minus_rank": _boot_paired(candidates["blend"], candidates["rank_only"]),
    }

    # --- Calibration regression logit(y) = a + b logit(p_hat) for the blend ---
    def _calibration(p: np.ndarray) -> dict:
        x = logit(p).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10_000)
        model.fit(x, y)
        alpha = float(model.intercept_[0])
        beta = float(model.coef_[0, 0])
        # 10-bin ECE
        order = pd.qcut(pd.Series(p), q=10, duplicates="drop")
        frame = pd.DataFrame({"p": p, "y": y, "bin": order.astype(str)})
        grp = frame.groupby("bin", observed=True).agg(
            n=("y", "size"), mp=("p", "mean"), obs=("y", "mean")
        )
        ece = float((grp["n"] / grp["n"].sum() * (grp["obs"] - grp["mp"]).abs()).sum())
        # Block bootstrap for alpha, beta, ece
        a_bs, b_bs, e_bs = [], [], []
        for _ in range(1000):
            chosen = rng.choice(blocks, size=len(blocks), replace=True)
            idx = np.concatenate([np.where(block_id == b)[0] for b in chosen])
            yy, pp = y[idx], p[idx]
            if len(np.unique(yy)) < 2:
                continue
            m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10_000)
            m.fit(logit(pp).reshape(-1, 1), yy)
            a_bs.append(float(m.intercept_[0]))
            b_bs.append(float(m.coef_[0, 0]))
            fr = pd.DataFrame({"p": pp, "y": yy})
            try:
                fr["bin"] = pd.qcut(fr["p"], q=10, duplicates="drop").astype(str)
                g = fr.groupby("bin", observed=True).agg(n=("y", "size"), mp=("p", "mean"), obs=("y", "mean"))
                e_bs.append(float((g["n"] / g["n"].sum() * (g["obs"] - g["mp"]).abs()).sum()))
            except ValueError:
                pass
        return {
            "calibration_intercept_alpha": alpha,
            "calibration_slope_beta": beta,
            "alpha_ci_2_5": float(np.quantile(a_bs, 0.025)),
            "alpha_ci_97_5": float(np.quantile(a_bs, 0.975)),
            "beta_ci_2_5": float(np.quantile(b_bs, 0.025)),
            "beta_ci_97_5": float(np.quantile(b_bs, 0.975)),
            "ece_10bin": ece,
            "ece_ci_2_5": float(np.quantile(e_bs, 0.025)) if e_bs else None,
            "ece_ci_97_5": float(np.quantile(e_bs, 0.975)) if e_bs else None,
            "reliability_table": grp.reset_index().rename(
                columns={"mp": "mean_probability", "obs": "observed_home_win_rate", "n": "games"}
            ).to_dict(orient="records"),
        }

    calibration = _calibration(candidates["blend"])

    tail_counts = {
        "n": int(len(y)),
        "p_ge_0_80": int(np.sum(candidates["blend"] >= 0.80)),
        "p_ge_0_90": int(np.sum(candidates["blend"] >= 0.90)),
        "p_le_0_20": int(np.sum(candidates["blend"] <= 0.20)),
        "p_le_0_10": int(np.sum(candidates["blend"] <= 0.10)),
        "observed_home_win_rate": float(y.mean()),
    }

    # --- Champion-challenger verdict ---
    blend_beats_elo = (
        pooled_metrics["blend"]["log_loss"] < pooled_metrics["elo_only"]["log_loss"]
        and pooled_metrics["blend"]["brier"] < pooled_metrics["elo_only"]["brier"]
    )
    stable = (
        paired["blend_minus_elo"]["delta_log_loss_ci_97_5"] < 0.0
        and paired["blend_minus_elo"]["delta_brier_ci_97_5"] < 0.0
    )
    verdict = {
        "champion": "elo_only",
        "challenger": "blend",
        "rule": "Promote blend only if it beats Elo-only on BOTH pooled log loss and Brier, with block-bootstrap upper CI < 0.",
        "blend_beats_elo_point_estimate_both": bool(blend_beats_elo),
        "blend_beats_elo_stable": bool(stable),
        "decision": "promote_blend" if (blend_beats_elo and stable) else "keep_elo_only",
    }

    summary = {
        "design": (
            "Nested rolling-origin: per outer weekly fold, architecture selected "
            "by inner rolling-origin blend log loss on data before the fold; "
            "constrained stacker trained on inner out-of-fold component preds; "
            "base models refit on all pre-fold data; outer fold scored untouched."
        ),
        "outer_folds": len(fold_records),
        "pooled_games": int(len(preds)),
        "architecture_selection_counts": (
            preds.drop_duplicates("origin")["selected_architecture"].value_counts().to_dict()
        ),
        "pooled_metrics": pooled_metrics,
        "paired_block_bootstrap": paired,
        "calibration": calibration,
        "tail_counts": tail_counts,
        "champion_challenger": verdict,
    }
    (artifact_dir / "nested_backtest_summary.json").write_text(json.dumps(summary, indent=2))

    # --- Reliability figure: blend vs Elo-only ---
    plt.figure(figsize=(6, 6))
    for name, style in [("blend", "o-"), ("elo_only", "s--")]:
        p = candidates[name]
        fr = pd.DataFrame({"p": p, "y": y})
        fr["bin"] = pd.qcut(fr["p"], q=10, duplicates="drop").astype(str)
        g = fr.groupby("bin", observed=True).agg(mp=("p", "mean"), obs=("y", "mean")).sort_values("mp")
        plt.plot(g["mp"], g["obs"], style, label=name)
    plt.plot([0, 1], [0, 1], ":", color="gray", label="perfect")
    plt.xlabel("Mean predicted home-win probability")
    plt.ylabel("Observed home-win rate")
    plt.title(
        f"Nested rolling-origin reliability (n={len(preds)})\n"
        f"blend LL={pooled_metrics['blend']['log_loss']:.3f} vs "
        f"Elo LL={pooled_metrics['elo_only']['log_loss']:.3f}"
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(figure_dir / "nested_backtest_reliability.png", dpi=160)
    plt.close()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/architecture_candidates.json")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--figure-dir", default="figures")
    args = parser.parse_args()
    summary = run(args.data, args.config, args.artifact_dir, args.figure_dir)
    # Print a compact digest.
    print(json.dumps({
        "outer_folds": summary["outer_folds"],
        "pooled_games": summary["pooled_games"],
        "architecture_selection_counts": summary["architecture_selection_counts"],
        "pooled_metrics": summary["pooled_metrics"],
        "blend_minus_elo": summary["paired_block_bootstrap"]["blend_minus_elo"],
        "calibration": {k: v for k, v in summary["calibration"].items() if k != "reliability_table"},
        "tail_counts": summary["tail_counts"],
        "champion_challenger": summary["champion_challenger"],
    }, indent=2))


if __name__ == "__main__":
    main()
