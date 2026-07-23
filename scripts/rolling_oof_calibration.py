"""Rolling-origin out-of-sample calibration.

The primary deliverable freezes state at a single origin (pre-April). That is
the right operational contract, but it produces only one holdout window. To
assess calibration more robustly this script runs an *expanding-window,
one-step-ahead* backtest over the March-April period:

  For each weekly fold W (starting 2026-03-01):
    * base Elo/rank logistic models are fit on games strictly before W;
    * the deploy stacker (temperature floor T>=1) is fit on component
      probabilities of games strictly before W;
    * the games inside W are scored and stored.

No fold ever contributes to the model that predicts it, so the pooled
predictions are genuinely out-of-sample. We then pool every fold and report a
single consolidated reliability table (deciles), an expected calibration error
(ECE), and a reliability diagram.

Outputs:
  artifacts/rolling_oof_predictions.csv   one row per scored game
  artifacts/rolling_oof_calibration.csv   decile reliability table
  artifacts/rolling_oof_metrics.json      pooled metrics + ECE/MCE + folds
  figures/rolling_oof_calibration.png     reliability diagram
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nba_wp.data import load_games
from nba_wp.features import Architecture, build_features
from nba_wp.model import (
    apply_logit_stacker,
    component_probabilities,
    evaluate,
    fit_base_models,
    fit_logit_stacker,
)

FEATURE_COLS = ["elo_diff", "bt_logit", "trend_diff"]


def _weekly_origins(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    origins = []
    cursor = start
    while cursor <= end:
        origins.append(cursor)
        cursor = cursor + pd.Timedelta(days=7)
    return origins


def run(
    data_path: str | Path,
    spec_path: str | Path,
    artifact_dir: str | Path,
    figure_dir: str | Path,
    *,
    min_train_games: int = 200,
    min_stacker_games: int = 60,
) -> dict:
    artifact_dir = Path(artifact_dir)
    figure_dir = Path(figure_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    spec = json.loads(Path(spec_path).read_text())
    architecture = Architecture.from_dict(spec["architecture"])
    games = load_games(data_path)

    # Causal pregame features for every game (no look-ahead by construction).
    features = build_features(games, architecture)

    holdout_start = pd.Timestamp("2026-03-01")
    max_date = features["game_date"].max()
    origins = _weekly_origins(holdout_start, max_date)

    collected: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    for origin in origins:
        fold_end = origin + pd.Timedelta(days=7)
        train = features[features["game_date"] < origin]
        fold = features[
            (features["game_date"] >= origin) & (features["game_date"] < fold_end)
        ]
        if len(fold) == 0 or len(train) < min_train_games:
            continue

        # Base models on everything strictly before the fold; stacker on the
        # most recent pre-fold window (component probs from those base models).
        base_models = fit_base_models(train, architecture)
        stacker_train = train[train["game_date"] >= (origin - pd.Timedelta(days=45))]
        if len(stacker_train) < min_stacker_games:
            stacker_train = train
        pe_tr, pr_tr = component_probabilities(base_models, stacker_train)
        stacker = fit_logit_stacker(
            stacker_train["home_win"].to_numpy(dtype=int),
            pe_tr,
            pr_tr,
            min_temperature=1.0,
        )

        pe, pr = component_probabilities(base_models, fold)
        probability = apply_logit_stacker(stacker, pe, pr)

        block = fold[["game_id", "game_date", "away", "home", "home_win"]].copy()
        block["origin"] = origin.strftime("%Y-%m-%d")
        block["home_win_probability"] = probability
        collected.append(block)

        fold_metrics = evaluate(fold["home_win"], probability)
        fold_rows.append(
            {
                "origin": origin.strftime("%Y-%m-%d"),
                "train_games": int(len(train)),
                "fold_games": int(len(fold)),
                **{k: fold_metrics[k] for k in ["log_loss", "brier", "auc", "accuracy"]},
            }
        )

    pooled = pd.concat(collected, ignore_index=True)
    pooled.to_csv(artifact_dir / "rolling_oof_predictions.csv", index=False)

    y = pooled["home_win"].to_numpy(dtype=int)
    p = pooled["home_win_probability"].to_numpy(dtype=float)

    # Decile reliability table on pooled OOF predictions.
    labels = pd.qcut(pooled["home_win_probability"], q=10, duplicates="drop")
    table = (
        pooled.assign(bin=labels.astype(str))
        .groupby("bin", observed=True)
        .agg(
            games=("game_id", "size"),
            mean_probability=("home_win_probability", "mean"),
            observed_home_win_rate=("home_win", "mean"),
            min_probability=("home_win_probability", "min"),
            max_probability=("home_win_probability", "max"),
        )
        .reset_index()
        .sort_values("mean_probability")
        .reset_index(drop=True)
    )
    table["gap"] = table["observed_home_win_rate"] - table["mean_probability"]
    table.to_csv(artifact_dir / "rolling_oof_calibration.csv", index=False)

    weights = table["games"] / table["games"].sum()
    ece = float((weights * table["gap"].abs()).sum())
    mce = float(table["gap"].abs().max())

    pooled_metrics = evaluate(y, p)
    metrics = {
        "design": (
            "expanding-window one-step-ahead; weekly folds from 2026-03-01; "
            "base models and deploy stacker (T>=1) fit strictly before each fold"
        ),
        "folds": len(fold_rows),
        "pooled_games": int(len(pooled)),
        "pooled": pooled_metrics,
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
        "per_fold": fold_rows,
    }
    (artifact_dir / "rolling_oof_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    plt.figure(figsize=(6, 6))
    plt.plot(
        table["mean_probability"],
        table["observed_home_win_rate"],
        marker="o",
        label="Rolling OOF (pooled deciles)",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    plt.xlabel("Mean predicted home-win probability")
    plt.ylabel("Observed home-win rate")
    plt.title(
        f"Rolling-origin OOS reliability\n"
        f"n={len(pooled)}, ECE={ece:.3f}, log loss={pooled_metrics['log_loss']:.3f}"
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(figure_dir / "rolling_oof_calibration.png", dpi=160)
    plt.close()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--selected-spec", default="artifacts/selected_spec.json")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--figure-dir", default="figures")
    args = parser.parse_args()

    metrics = run(
        args.data,
        args.selected_spec,
        args.artifact_dir,
        args.figure_dir,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
