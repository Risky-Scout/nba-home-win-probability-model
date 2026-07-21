
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .features import Architecture, build_features
from .model import (
    apply_logit_stacker,
    component_probabilities,
    evaluate,
    fit_base_models,
    fit_logit_stacker,
    stacker_calibration_dict,
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def run_selection(
    games: pd.DataFrame,
    architecture_config: dict[str, Any],
    selection_policy: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Select the architecture using March only.

    For each architecture: fit base models on games before 2026-03-01, fit a
    logistic stacker on the March component logits, and score March. The
    selection rule is: minimize March log loss. Nothing else.

    The caller must provide no rows dated April or later. The function checks
    that contract and writes proof fields into the selected specification.
    """
    del selection_policy  # policy carries no tunable state after Task 2
    if games["game_date"].max() >= pd.Timestamp("2026-04-01"):
        raise ValueError(
            "Selection data contain April rows. Truncate at 2026-04-01 first."
        )

    architecture_rows: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []

    for architecture_values in architecture_config["architectures"]:
        architecture = Architecture.from_dict(architecture_values)
        features = build_features(games, architecture)
        train = features[features["game_date"] < "2026-03-01"].copy()
        march = features[
            (features["game_date"] >= "2026-03-01")
            & (features["game_date"] < "2026-04-01")
        ].copy()

        models = fit_base_models(train, architecture)
        elo_probability, rank_probability = component_probabilities(models, march)
        march_home_win = march["home_win"].to_numpy(dtype=int)
        stacker = fit_logit_stacker(
            march_home_win,
            elo_probability,
            rank_probability,
        )
        march_probability = apply_logit_stacker(
            stacker,
            elo_probability,
            rank_probability,
        )
        metrics = evaluate(march_home_win, march_probability)
        calibration = stacker_calibration_dict(stacker)

        architecture_rows.append(
            {
                "architecture": architecture.name,
                "elo_k": architecture.elo_k,
                "elo_hfa": architecture.elo_hfa,
                "elo_mov": architecture.elo_mov,
                "bt_c": architecture.bt_c,
                "trend_half_life_days": architecture.trend_half_life_days,
                "trend_short_games": architecture.trend_short_games,
                "elo_model_c": architecture.elo_model_c,
                "rank_model_c": architecture.rank_model_c,
                "coef_elo_logit": calibration["coef_elo_logit"],
                "coef_rank_logit": calibration["coef_rank_logit"],
                "intercept": calibration["intercept"],
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
            }
        )
        selected_candidates.append(
            {
                "architecture": architecture,
                "calibration": calibration,
                "metrics": {
                    key: metrics[key]
                    for key in ["log_loss", "brier", "auc", "accuracy"]
                },
            }
        )

    # Selection rule: minimize March log loss. Architecture name only breaks
    # exact ties deterministically.
    selected_candidates.sort(
        key=lambda item: (
            item["metrics"]["log_loss"],
            item["architecture"].name,
        )
    )
    winner = selected_candidates[0]
    selected_spec = {
        "model_family": "logistic-stacked blend: Elo + Bradley-Terry/recent-trend",
        "selected_using": "March 2026 operational one-step-ahead validation",
        "selection_data_max_date": games["game_date"].max().strftime("%Y-%m-%d"),
        "april_rows_loaded_during_selection": int(
            (games["game_date"] >= "2026-04-01").sum()
        ),
        "selection_rule": "Minimize March log loss.",
        "primary_metric": "log_loss",
        "secondary_metrics": ["brier", "auc", "accuracy"],
        "information_policy": "sequential_daily",
        "architecture": winner["architecture"].to_dict(),
        "calibration": winner["calibration"],
        "march_validation_metrics": winner["metrics"],
        "architecture_count": len(architecture_config["architectures"]),
        "notes": [
            "Base model coefficients use games through February.",
            "March state features update only after each completed game date.",
            "No April row is loaded by model selection.",
            "The final April model refits base coefficients through March.",
        ],
    }

    architecture_table = pd.DataFrame(architecture_rows).sort_values(
        "log_loss",
        ascending=True,
    ).reset_index(drop=True)
    top_table = architecture_table.copy()

    return selected_spec, architecture_table, top_table
