
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .features import Architecture, build_features
from .model import (
    Calibration,
    component_probabilities,
    evaluate,
    fit_base_models,
    search_calibration,
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _float_grid(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(count + 1)]


def run_selection(
    games: pd.DataFrame,
    architecture_config: dict[str, Any],
    selection_policy: dict[str, Any],
    benchmarks: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Select the architecture and calibration using March only.

    The caller must provide no rows dated April or later. The function checks
    that contract and writes proof fields into the selected specification.
    """
    if games["game_date"].max() >= pd.Timestamp("2026-04-01"):
        raise ValueError(
            "Selection data contain April rows. Truncate at 2026-04-01 first."
        )

    march_benchmark = benchmarks["march"]
    grid = selection_policy["calibration_grid"]
    weights = _float_grid(**grid["elo_weight"])
    temperatures = _float_grid(**grid["temperature"])
    shifts = _float_grid(**grid["shift"])

    architecture_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
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
        search = search_calibration(
            march["home_win"].to_numpy(),
            elo_probability,
            rank_probability,
            weights=weights,
            temperatures=temperatures,
            shifts=shifts,
            benchmarks=march_benchmark,
            top_n=int(selection_policy.get("top_candidates_per_architecture", 100)),
        )

        best_any = search["best_any"]
        best_eligible = search["best_eligible"]
        row = {
            "architecture": architecture.name,
            "elo_k": architecture.elo_k,
            "elo_hfa": architecture.elo_hfa,
            "elo_mov": architecture.elo_mov,
            "bt_c": architecture.bt_c,
            "trend_half_life_days": architecture.trend_half_life_days,
            "trend_short_games": architecture.trend_short_games,
            "elo_model_c": architecture.elo_model_c,
            "rank_model_c": architecture.rank_model_c,
            "candidate_count": search["candidate_count"],
            "eligible_count": search["eligible_count"],
            "best_any_log_loss": best_any["log_loss"],
            "best_any_brier": best_any["brier"],
            "best_any_auc": best_any["auc"],
            "best_any_accuracy": best_any["accuracy"],
        }
        if best_eligible is not None:
            row.update(
                {
                    "eligible": True,
                    "best_eligible_log_loss": best_eligible["log_loss"],
                    "best_eligible_brier": best_eligible["brier"],
                    "best_eligible_auc": best_eligible["auc"],
                    "best_eligible_accuracy": best_eligible["accuracy"],
                    "best_eligible_elo_weight": best_eligible["elo_weight"],
                    "best_eligible_temperature": best_eligible["temperature"],
                    "best_eligible_shift": best_eligible["shift"],
                }
            )
            selected_candidates.append(
                {
                    "architecture": architecture,
                    "calibration": Calibration.from_dict(best_eligible),
                    "metrics": {
                        key: best_eligible[key]
                        for key in ["log_loss", "brier", "auc", "accuracy"]
                    },
                }
            )
        else:
            row["eligible"] = False
        architecture_rows.append(row)

        for rank, candidate in enumerate(search["top_eligible"], start=1):
            top_rows.append(
                {
                    "architecture": architecture.name,
                    "architecture_rank": rank,
                    **candidate,
                }
            )

    if not selected_candidates:
        raise RuntimeError(
            "No candidate beat all four March targets under the declared grid."
        )

    # Predeclared lexicographic rule: proper score first, then ranking and
    # threshold diagnostics. No April value appears in this key.
    selected_candidates.sort(
        key=lambda item: (
            item["metrics"]["log_loss"],
            item["metrics"]["brier"],
            -item["metrics"]["auc"],
            -item["metrics"]["accuracy"],
            item["architecture"].name,
        )
    )
    winner = selected_candidates[0]
    selected_spec = {
        "model_family": "calibrated logit blend: Elo + Bradley-Terry/recent-trend",
        "selected_using": "March 2026 operational one-step-ahead validation",
        "selection_data_max_date": games["game_date"].max().strftime("%Y-%m-%d"),
        "april_rows_loaded_during_selection": int(
            (games["game_date"] >= "2026-04-01").sum()
        ),
        "selection_rule": (
            "Filter to candidates beating all four March targets; minimize "
            "March log loss; tie-break by Brier, AUC, accuracy, architecture name."
        ),
        "primary_metric": "log_loss",
        "secondary_metrics": ["brier", "auc", "accuracy"],
        "information_policy": "sequential_daily",
        "architecture": winner["architecture"].to_dict(),
        "calibration": winner["calibration"].to_dict(),
        "march_validation_metrics": winner["metrics"],
        "march_benchmarks": march_benchmark,
        "calibration_grid": grid,
        "architecture_count": len(architecture_config["architectures"]),
        "notes": [
            "Base model coefficients use games through February.",
            "March state features update only after each completed game date.",
            "No April row is loaded by model selection.",
            "The final April model refits base coefficients through March.",
        ],
    }

    architecture_table = pd.DataFrame(architecture_rows).sort_values(
        ["eligible", "best_eligible_log_loss", "best_any_log_loss"],
        ascending=[False, True, True],
        na_position="last",
    )
    top_table = pd.DataFrame(top_rows)
    if not top_table.empty:
        top_table = top_table.sort_values(
            ["log_loss", "brier", "auc", "accuracy"],
            ascending=[True, True, False, False],
        ).reset_index(drop=True)

    return selected_spec, architecture_table, top_table
