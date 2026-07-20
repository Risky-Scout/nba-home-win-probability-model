
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import Architecture, build_features
from .model import (
    Calibration,
    evaluate,
    fit_base_models,
    fit_direct_logistic,
    fit_platt_calibration,
    predict_direct_logistic,
    raw_blend_probability,
    component_probabilities,
)


SELECTION_CUTOFF = pd.Timestamp("2026-03-01")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def assert_pre_march_selection_frame(games: pd.DataFrame) -> None:
    """Hard fail if March or later rows enter architecture/hyperparameter selection."""
    if games.empty:
        raise ValueError("Selection frame is empty.")
    max_date = pd.Timestamp(games["game_date"].max())
    if max_date >= SELECTION_CUTOFF:
        raise ValueError(
            "Selection data contain March or later rows. "
            "Truncate strictly before 2026-03-01 first. "
            f"Found max date {max_date.date()}."
        )
    march_or_later = int((games["game_date"] >= SELECTION_CUTOFF).sum())
    if march_or_later != 0:
        raise ValueError(
            f"Selection data contain {march_or_later} March-or-later rows."
        )


def _make_architecture(
    elo_k: float,
    trend_half_life_days: float,
    defaults: dict[str, Any],
) -> Architecture:
    name = f"k{int(elo_k)}_hl{int(trend_half_life_days)}"
    return Architecture(
        name=name,
        elo_k=float(elo_k),
        elo_hfa=float(defaults["elo_hfa"]),
        elo_mov=str(defaults["elo_mov"]),
        bt_c=float(defaults["bt_c"]),
        trend_half_life_days=float(trend_half_life_days),
        trend_short_games=int(defaults["trend_short_games"]),
        elo_model_c=float(defaults["elo_model_c"]),
        rank_model_c=float(defaults["rank_model_c"]),
    )


def _fold_frames(
    features: pd.DataFrame,
    fold: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = features[features["game_date"] < fold["train_end"]].copy()
    validation = features[
        (features["game_date"] >= fold["validation_start"])
        & (features["game_date"] < fold["validation_end"])
    ].copy()
    if train.empty or validation.empty:
        raise RuntimeError(f"Fold {fold['name']} produced an empty split.")
    return train, validation


def run_selection(
    games: pd.DataFrame,
    architecture_config: dict[str, Any],
    selection_policy: dict[str, Any],
    benchmarks: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Select model using expanding-window folds that end before March.

    Primary metric: mean validation log loss across January and February folds.
    External benchmark values are ignored for selection decisions.
    """
    del benchmarks  # intentionally unused for selection governance
    assert_pre_march_selection_frame(games)

    budget = architecture_config["search_budget"]
    defaults = architecture_config["feature_defaults"]
    folds = selection_policy["folds"]
    blend_cfg = architecture_config.get("blend_challenger", {"enabled": False})

    fold_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []

    for elo_k in budget["elo_k"]:
        for half_life in budget["trend_half_life_days"]:
            architecture = _make_architecture(elo_k, half_life, defaults)
            features = build_features(games, architecture)

            for logistic_c in budget["logistic_c"]:
                fold_metrics: list[dict[str, float]] = []
                for fold in folds:
                    train, validation = _fold_frames(features, fold)
                    model = fit_direct_logistic(train, float(logistic_c))
                    probability = predict_direct_logistic(model, validation)
                    metrics = evaluate(validation["home_win"], probability)
                    fold_metrics.append(metrics)
                    fold_rows.append(
                        {
                            "model_type": "direct_logistic",
                            "architecture": architecture.name,
                            "elo_k": architecture.elo_k,
                            "trend_half_life_days": architecture.trend_half_life_days,
                            "logistic_c": float(logistic_c),
                            "fold": fold["name"],
                            **{f"validation_{k}": v for k, v in metrics.items()},
                        }
                    )
                summary = {
                    "model_type": "direct_logistic",
                    "architecture": architecture,
                    "logistic_c": float(logistic_c),
                    "mean_validation_log_loss": float(
                        np.mean([m["log_loss"] for m in fold_metrics])
                    ),
                    "mean_validation_brier": float(
                        np.mean([m["brier"] for m in fold_metrics])
                    ),
                    "mean_validation_auc": float(
                        np.nanmean([m["auc"] for m in fold_metrics])
                    ),
                    "mean_validation_accuracy": float(
                        np.mean([m["accuracy"] for m in fold_metrics])
                    ),
                    "calibration": Calibration(
                        method="none",
                        elo_weight=0.0,
                        temperature=1.0,
                        shift=0.0,
                    ),
                    "platt_calibration": None,
                }
                candidate_rows.append(
                    {
                        "model_type": summary["model_type"],
                        "architecture": architecture.name,
                        "elo_k": architecture.elo_k,
                        "trend_half_life_days": architecture.trend_half_life_days,
                        "logistic_c": summary["logistic_c"],
                        "mean_validation_log_loss": summary["mean_validation_log_loss"],
                        "mean_validation_brier": summary["mean_validation_brier"],
                        "mean_validation_auc": summary["mean_validation_auc"],
                        "mean_validation_accuracy": summary["mean_validation_accuracy"],
                    }
                )
                selected_candidates.append(summary)

            if blend_cfg.get("enabled", False):
                elo_weight = float(blend_cfg.get("elo_weight", 0.2))
                fold_metrics = []
                for fold in folds:
                    train, validation = _fold_frames(features, fold)
                    models = fit_base_models(train, architecture)
                    elo_p, rank_p = component_probabilities(models, validation)
                    raw = raw_blend_probability(elo_p, rank_p, elo_weight)
                    # Platt is fit only on training predictions/labels, then
                    # applied to validation. Do not fit on validation labels.
                    elo_tr, rank_tr = component_probabilities(models, train)
                    raw_tr = raw_blend_probability(elo_tr, rank_tr, elo_weight)
                    platt = fit_platt_calibration(
                        raw_tr,
                        train["home_win"].to_numpy(dtype=int),
                    )
                    probability = platt.apply(raw)
                    metrics = evaluate(validation["home_win"], probability)
                    fold_metrics.append(metrics)
                    fold_rows.append(
                        {
                            "model_type": "blend_platt_challenger",
                            "architecture": architecture.name,
                            "elo_k": architecture.elo_k,
                            "trend_half_life_days": architecture.trend_half_life_days,
                            "logistic_c": np.nan,
                            "elo_weight": elo_weight,
                            "fold": fold["name"],
                            **{f"validation_{k}": v for k, v in metrics.items()},
                        }
                    )
                summary = {
                    "model_type": "blend_platt_challenger",
                    "architecture": architecture,
                    "logistic_c": None,
                    "mean_validation_log_loss": float(
                        np.mean([m["log_loss"] for m in fold_metrics])
                    ),
                    "mean_validation_brier": float(
                        np.mean([m["brier"] for m in fold_metrics])
                    ),
                    "mean_validation_auc": float(
                        np.nanmean([m["auc"] for m in fold_metrics])
                    ),
                    "mean_validation_accuracy": float(
                        np.mean([m["accuracy"] for m in fold_metrics])
                    ),
                    "calibration": Calibration(
                        method="blend_platt",
                        elo_weight=elo_weight,
                        temperature=1.0,
                        shift=0.0,
                    ),
                    "platt_calibration": None,
                }
                candidate_rows.append(
                    {
                        "model_type": summary["model_type"],
                        "architecture": architecture.name,
                        "elo_k": architecture.elo_k,
                        "trend_half_life_days": architecture.trend_half_life_days,
                        "logistic_c": np.nan,
                        "mean_validation_log_loss": summary["mean_validation_log_loss"],
                        "mean_validation_brier": summary["mean_validation_brier"],
                        "mean_validation_auc": summary["mean_validation_auc"],
                        "mean_validation_accuracy": summary["mean_validation_accuracy"],
                    }
                )
                selected_candidates.append(summary)

    if not selected_candidates:
        raise RuntimeError("No pre-March candidates were evaluated.")

    selected_candidates.sort(
        key=lambda item: (
            item["mean_validation_log_loss"],
            item["mean_validation_brier"],
            -item["mean_validation_auc"]
            if item["mean_validation_auc"] == item["mean_validation_auc"]
            else 0.0,
            -item["mean_validation_accuracy"],
            item["model_type"],
            item["architecture"].name,
        )
    )
    winner = selected_candidates[0]

    # Fit final Platt on all pre-March data for blend challenger if selected.
    platt_dict = None
    if winner["model_type"] == "blend_platt_challenger":
        features = build_features(games, winner["architecture"])
        models = fit_base_models(features, winner["architecture"])
        elo_p, rank_p = component_probabilities(models, features)
        raw = raw_blend_probability(
            elo_p,
            rank_p,
            winner["calibration"].elo_weight,
        )
        platt = fit_platt_calibration(raw, features["home_win"].to_numpy(dtype=int))
        platt_dict = platt.to_dict()

    selected_spec = {
        "model_family": (
            "direct L2 logistic on elo_diff + bt_logit + trend_diff"
            if winner["model_type"] == "direct_logistic"
            else "Platt-calibrated Elo + Bradley-Terry/trend blend challenger"
        ),
        "model_type": winner["model_type"],
        "selected_using": (
            "Expanding-window pre-March validation "
            "(train through Dec -> Jan; train through Jan -> Feb)"
        ),
        "selection_data_end": games["game_date"].max().strftime("%Y-%m-%d"),
        "selection_data_max_date": games["game_date"].max().strftime("%Y-%m-%d"),
        "march_rows_used_in_selection": 0,
        "april_rows_used_in_selection": 0,
        "april_rows_loaded_during_selection": 0,
        "selection_metric": "mean_validation_log_loss",
        "selection_rule": (
            "Minimize mean January/February validation log loss; "
            "tie-break by mean Brier, then AUC, accuracy, model type, architecture name. "
            "External benchmark values do not gate selection."
        ),
        "primary_metric": "mean_validation_log_loss",
        "secondary_metrics": ["mean_validation_brier"],
        "descriptive_metrics": [
            "mean_validation_auc",
            "mean_validation_accuracy",
        ],
        "information_policy_primary_april": "frozen_snapshot",
        "architecture": winner["architecture"].to_dict(),
        "logistic_c": winner["logistic_c"],
        "calibration": winner["calibration"].to_dict(),
        "platt_calibration": platt_dict,
        "pre_march_validation_metrics": {
            "mean_validation_log_loss": winner["mean_validation_log_loss"],
            "mean_validation_brier": winner["mean_validation_brier"],
            "mean_validation_auc": winner["mean_validation_auc"],
            "mean_validation_accuracy": winner["mean_validation_accuracy"],
        },
        "search_budget": budget,
        "fold_definitions": folds,
        "notes": [
            "No March or April row is loaded by model selection.",
            "March is a locked pre-final test after the specification is frozen.",
            "Frozen March 31 state is the primary April assignment result.",
            "Sequential April scoring is retained only as operational sensitivity.",
            "Original v1 submission is preserved at git tag v1-original-submission.",
        ],
    }

    fold_table = pd.DataFrame(fold_rows)
    selection_table = pd.DataFrame(candidate_rows).sort_values(
        [
            "mean_validation_log_loss",
            "mean_validation_brier",
            "mean_validation_auc",
            "mean_validation_accuracy",
        ],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)

    return selected_spec, fold_table, selection_table
