"""Clean April prediction charts for interview presentation.

Reads:
  outputs/april_predictions.csv
  artifacts/final_metrics.json

Writes:
  figures/april_predictions_reliability_clean.png
  figures/april_predictions_by_game_clean.png
  figures/april_predictions_summary_clean.png

Usage (from repo root):
  python -m scripts.plot_april_predictions_clean
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "figures"
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(root / "outputs" / "april_predictions.csv")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    # 1) Reliability / calibration
    bins = np.linspace(0, 1, 11)
    df["bin"] = pd.cut(df["home_win_probability"], bins=bins, include_lowest=True)
    cal = (
        df.groupby("bin", observed=False)
        .agg(
            n=("home_win", "size"),
            mean_pred=("home_win_probability", "mean"),
            observed=("home_win", "mean"),
        )
        .dropna(subset=["mean_pred"])
    )
    cal = cal[cal["n"] > 0]

    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=160)
    ax.plot([0, 1], [0, 1], color="#9AA0A6", linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax.plot(
        cal["mean_pred"],
        cal["observed"],
        color="#1A73E8",
        marker="o",
        markersize=7,
        linewidth=2,
        label="April model",
    )
    for _, row in cal.iterrows():
        ax.annotate(
            f"n={int(row.n)}",
            (row.mean_pred, row.observed),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="#5F6368",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted home-win probability")
    ax.set_ylabel("Observed home-win rate")
    ax.set_title("April home-win predictions — reliability")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / "april_predictions_reliability_clean.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 2) Predictions by game
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    x = np.arange(len(df))
    wins = df["home_win"] == 1
    ax.scatter(
        x[wins],
        df.loc[wins, "home_win_probability"],
        c="#1A73E8",
        s=28,
        label="Home won",
        zorder=3,
    )
    ax.scatter(
        x[~wins],
        df.loc[~wins, "home_win_probability"],
        c="#D93025",
        s=28,
        label="Home lost",
        zorder=3,
    )
    ax.axhline(0.5, color="#9AA0A6", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("April games (chronological)")
    ax.set_ylabel("Predicted P(home win)")
    ax.set_title("April predictions by game")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, axis="y", alpha=0.25)
    dates = df["game_date"].dt.strftime("%m-%d")
    step = max(1, len(df) // 8)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(dates[::step], rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir / "april_predictions_by_game_clean.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 3) Summary card
    apr = json.loads((root / "artifacts" / "final_metrics.json").read_text())["sequential_daily"]["april"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
    ax = axes[0]
    ax.hist(df["home_win_probability"], bins=12, color="#1A73E8", edgecolor="white", alpha=0.9)
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Number of April games")
    ax.set_title("Distribution of April prices")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    ax.axis("off")
    ax.set_title("April sequential results", loc="left", fontsize=12, pad=12)
    lines = [
        f"Games: {int(apr['games'])}",
        f"Correct @ 0.5: {int(round(apr['accuracy'] * apr['games']))}/{int(apr['games'])}",
        f"Log loss: {apr['log_loss']:.4f}",
        f"Brier:    {apr['brier']:.4f}",
        f"AUC:      {apr['auc']:.3f}",
        f"Accuracy: {apr['accuracy']:.1%}",
        "",
        "Source: outputs/april_predictions.csv",
        "Policy: frozen_pre_april",
    ]
    ax.text(
        0.02,
        0.92,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "april_predictions_summary_clean.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Wrote clean April charts to figures/")


if __name__ == "__main__":
    main()
