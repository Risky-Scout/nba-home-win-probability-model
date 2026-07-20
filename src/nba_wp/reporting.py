
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import Architecture, CANDIDATE_FEATURES, build_features, feature_dictionary
from .model import (
    DIRECT_FEATURES,
    BaseModels,
    Calibration,
    calibration_diagnostics,
    component_probabilities,
    date_block_bootstrap,
    direct_coefficient_rows,
    evaluate,
    extreme_probability_audit,
    fit_base_models,
    fit_direct_logistic,
    predict,
    predict_direct_logistic,
    predict_from_spec,
    standardized_coefficient_rows,
)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=False))


def prediction_frame(
    frame: pd.DataFrame,
    final_probability: np.ndarray,
    elo_probability: np.ndarray,
    rank_probability: np.ndarray,
) -> pd.DataFrame:
    output = frame[
        [
            "game_id",
            "game_date",
            "away",
            "home",
            "home_win",
            "state_policy",
            "performance_cutoff",
            "elo_diff",
            "bt_logit",
            "trend_diff",
        ]
    ].copy()
    output["elo_component_probability"] = elo_probability
    output["rank_component_probability"] = rank_probability
    output["home_win_probability"] = final_probability
    output["predicted_home_win"] = (final_probability >= 0.5).astype(int)
    p = np.clip(final_probability, 1e-12, 1.0 - 1e-12)
    y = output["home_win"].to_numpy(dtype=int)
    output["log_loss_contribution"] = -(
        y * np.log(p) + (1 - y) * np.log(1.0 - p)
    )
    output["brier_contribution"] = (p - y) ** 2
    output["fair_home_decimal_odds"] = 1.0 / p
    output["fair_away_decimal_odds"] = 1.0 / (1.0 - p)
    return output


def calibration_bins(
    predictions: pd.DataFrame,
    bins: int = 8,
) -> pd.DataFrame:
    frame = predictions.copy()
    labels = pd.qcut(
        frame["home_win_probability"],
        q=min(bins, len(frame)),
        duplicates="drop",
    )
    grouped = (
        frame.assign(calibration_bin=labels.astype(str))
        .groupby("calibration_bin", observed=True)
        .agg(
            games=("game_id", "size"),
            mean_probability=("home_win_probability", "mean"),
            observed_home_win_rate=("home_win", "mean"),
            minimum_probability=("home_win_probability", "min"),
            maximum_probability=("home_win_probability", "max"),
        )
        .reset_index()
    )
    return grouped


def _simple_logistic(c_value: float = 0.1) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    max_iter=10_000,
                    tol=1e-12,
                    random_state=0,
                ),
            ),
        ]
    )


def ablation_table(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    final_models: BaseModels,
    calibration: Calibration,
) -> pd.DataFrame:
    y = validation["home_win"].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []

    baseline_probability = np.full(
        len(validation),
        train["home_win"].mean(),
        dtype=float,
    )
    rows.append(
        {
            "stage": "B0",
            "model": "Training-period constant home prior",
            "features": "(none)",
            **evaluate(y, baseline_probability),
        }
    )

    candidates = [
        ("B1", "Record-only logistic", ["record_logit_diff"]),
        ("B2", "Record + cumulative margin", ["record_logit_diff", "cumulative_margin_diff"]),
        ("B3", "Elo component", ["elo_diff"]),
        ("B4", "Bradley-Terry component", ["bt_logit"]),
        ("B5", "Bradley-Terry + trend", ["bt_logit", "trend_diff"]),
        (
            "B6",
            "Rich linear challenger",
            [
                "record_logit_diff",
                "cumulative_margin_diff",
                "trend_diff",
                "rest_advantage",
                "turnover_advantage_diff",
                "rebound_advantage_diff",
            ],
        ),
    ]
    for stage, name, columns in candidates:
        model = _simple_logistic(0.1)
        model.fit(train[columns], train["home_win"].astype(int))
        probability = model.predict_proba(validation[columns])[:, 1]
        rows.append(
            {
                "stage": stage,
                "model": name,
                "features": ", ".join(columns),
                **evaluate(y, probability),
            }
        )

    direct = fit_direct_logistic(train, 0.3)
    direct_probability = predict_direct_logistic(direct, validation)
    rows.append(
        {
            "stage": "B7",
            "model": "Direct logistic on elo_diff + bt_logit + trend_diff",
            "features": ", ".join(DIRECT_FEATURES),
            **evaluate(y, direct_probability),
        }
    )
    if final_models is not None and calibration is not None:
        final_probability, _, _ = predict(final_models, validation, calibration)
        rows.append(
            {
                "stage": "B8",
                "model": "Blend challenger (if available)",
                "features": "elo_diff | bt_logit + trend_diff",
                **evaluate(y, final_probability),
            }
        )

    result = pd.DataFrame(rows)
    result["delta_log_loss_vs_constant"] = (
        result["log_loss"] - result.loc[result["stage"] == "B0", "log_loss"].iloc[0]
    )
    result["delta_brier_vs_constant"] = (
        result["brier"] - result.loc[result["stage"] == "B0", "brier"].iloc[0]
    )
    return result


def permutation_importance(
    models: BaseModels,
    validation: pd.DataFrame,
    calibration: Calibration,
    *,
    repeats: int = 100,
    seed: int = 365,
) -> pd.DataFrame:
    baseline, _, _ = predict(models, validation, calibration)
    baseline_metrics = evaluate(validation["home_win"], baseline)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for feature in ["elo_diff", "bt_logit", "trend_diff"]:
        changes = []
        for repeat in range(repeats):
            shuffled = validation.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            probability, _, _ = predict(models, shuffled, calibration)
            metric = evaluate(shuffled["home_win"], probability)
            changes.append(metric["log_loss"] - baseline_metrics["log_loss"])
        changes_array = np.asarray(changes)
        rows.append(
            {
                "feature": feature,
                "repeats": repeats,
                "baseline_log_loss": baseline_metrics["log_loss"],
                "mean_log_loss_increase": float(changes_array.mean()),
                "std_log_loss_increase": float(changes_array.std(ddof=1)),
                "p05_log_loss_increase": float(np.quantile(changes_array, 0.05)),
                "p95_log_loss_increase": float(np.quantile(changes_array, 0.95)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_log_loss_increase", ascending=False
    )


def bootstrap_metric_difference(
    y: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    *,
    repeats: int = 2_000,
    seed: int = 2026,
) -> dict[str, float]:
    """Paired bootstrap for log-loss difference A minus B."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    p_a = np.clip(np.asarray(probability_a), 1e-12, 1 - 1e-12)
    p_b = np.clip(np.asarray(probability_b), 1e-12, 1 - 1e-12)
    loss_a = -(y * np.log(p_a) + (1 - y) * np.log(1 - p_a))
    loss_b = -(y * np.log(p_b) + (1 - y) * np.log(1 - p_b))
    differences = loss_a - loss_b
    samples = np.empty(repeats, dtype=float)
    n = len(y)
    for i in range(repeats):
        index = rng.integers(0, n, size=n)
        samples[i] = differences[index].mean()
    return {
        "observed_difference_a_minus_b": float(differences.mean()),
        "bootstrap_mean": float(samples.mean()),
        "ci_2_5": float(np.quantile(samples, 0.025)),
        "ci_97_5": float(np.quantile(samples, 0.975)),
        "probability_a_better_than_b": float(np.mean(samples < 0.0)),
        "repeats": repeats,
    }


def save_figures(
    output_dir: str | Path,
    march_predictions: pd.DataFrame,
    april_predictions: pd.DataFrame,
    ablation: pd.DataFrame,
    importance: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.bar(ablation["stage"], ablation["log_loss"])
    plt.xlabel("Ablation stage")
    plt.ylabel("March log loss")
    plt.title("Chronological March feature ablation")
    plt.tight_layout()
    plt.savefig(out / "ablation_log_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(
        importance["feature"],
        importance["mean_log_loss_increase"],
        yerr=importance["std_log_loss_increase"],
    )
    plt.ylabel("Increase in March log loss after permutation")
    plt.title("Permutation importance of selected features")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out / "permutation_importance.png", dpi=160)
    plt.close()

    for name, predictions in [
        ("march", march_predictions),
        ("april", april_predictions),
    ]:
        bins = calibration_bins(predictions)
        plt.figure(figsize=(6, 6))
        plt.plot(
            bins["mean_probability"],
            bins["observed_home_win_rate"],
            marker="o",
        )
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Mean predicted home-win probability")
        plt.ylabel("Observed home-win rate")
        plt.title(f"{name.title()} reliability diagram")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(out / f"{name}_calibration.png", dpi=160)
        plt.close()

    plt.figure(figsize=(10, 8))
    image = plt.imshow(correlations.to_numpy(), vmin=-1, vmax=1)
    plt.colorbar(image, label="Correlation")
    plt.xticks(
        range(len(correlations.columns)),
        correlations.columns,
        rotation=90,
    )
    plt.yticks(range(len(correlations.index)), correlations.index)
    plt.title("March engineered-feature correlation matrix")
    plt.tight_layout()
    plt.savefig(out / "feature_correlation_matrix.png", dpi=160)
    plt.close()


def _component_placeholders(
    frame: pd.DataFrame,
    elo_p: np.ndarray | None,
    rank_p: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)
    if elo_p is None:
        elo_p = np.full(n, np.nan)
    if rank_p is None:
        rank_p = np.full(n, np.nan)
    return elo_p, rank_p


def permutation_importance_direct(
    model: Any,
    validation: pd.DataFrame,
    *,
    repeats: int = 100,
    seed: int = 365,
) -> pd.DataFrame:
    baseline = predict_direct_logistic(model, validation)
    baseline_metrics = evaluate(validation["home_win"], baseline)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    feature_names = list(getattr(model, "feature_names", DIRECT_FEATURES))
    for feature in feature_names:
        changes = []
        for _ in range(repeats):
            shuffled = validation.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            probability = predict_direct_logistic(model, shuffled)
            metric = evaluate(shuffled["home_win"], probability)
            changes.append(metric["log_loss"] - baseline_metrics["log_loss"])
        changes_array = np.asarray(changes)
        rows.append(
            {
                "feature": feature,
                "repeats": repeats,
                "baseline_log_loss": baseline_metrics["log_loss"],
                "mean_log_loss_increase": float(changes_array.mean()),
                "std_log_loss_increase": float(changes_array.std(ddof=1)),
                "p05_log_loss_increase": float(np.quantile(changes_array, 0.05)),
                "p95_log_loss_increase": float(np.quantile(changes_array, 0.95)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_log_loss_increase", ascending=False
    )


def score_and_write(
    games: pd.DataFrame,
    selected_spec: dict[str, Any],
    output_dir: str | Path,
    artifact_dir: str | Path,
    figure_dir: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    artifact_path = Path(artifact_dir)
    figure_path = Path(figure_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    architecture = Architecture.from_dict(selected_spec["architecture"])
    sequential_features = build_features(games, architecture)
    frozen_march_features = build_features(
        games[games["game_date"] < "2026-04-01"].copy(),
        architecture,
        freeze_date="2026-03-01",
    )
    frozen_april_features = build_features(
        games,
        architecture,
        freeze_date="2026-04-01",
    )

    train_feb = sequential_features[
        sequential_features["game_date"] < "2026-03-01"
    ].copy()
    march = sequential_features[
        (sequential_features["game_date"] >= "2026-03-01")
        & (sequential_features["game_date"] < "2026-04-01")
    ].copy()
    through_march = sequential_features[
        sequential_features["game_date"] < "2026-04-01"
    ].copy()
    april = sequential_features[
        sequential_features["game_date"] >= "2026-04-01"
    ].copy()
    frozen_march = frozen_march_features[
        frozen_march_features["game_date"] >= "2026-03-01"
    ].copy()
    frozen_april = frozen_april_features[
        frozen_april_features["game_date"] >= "2026-04-01"
    ].copy()

    march_probability, march_elo, march_rank, march_fit = predict_from_spec(
        selected_spec, train_feb, march
    )
    april_probability, april_elo, april_rank, final_fit = predict_from_spec(
        selected_spec, through_march, april
    )
    frozen_march_probability, frozen_march_elo, frozen_march_rank, _ = predict_from_spec(
        selected_spec, train_feb, frozen_march
    )
    frozen_april_probability, frozen_april_elo, frozen_april_rank, _ = predict_from_spec(
        selected_spec, through_march, frozen_april
    )

    # When the champion is direct logistic, still fit Elo / BT+trend components
    # so date-block bootstrap can report paired differences versus baselines.
    if march_elo is None or march_rank is None:
        component_models = fit_base_models(train_feb, architecture)
        march_elo, march_rank = component_probabilities(component_models, march)
        frozen_march_elo, frozen_march_rank = component_probabilities(
            component_models, frozen_march
        )
        april_components = fit_base_models(through_march, architecture)
        april_elo, april_rank = component_probabilities(april_components, april)
        frozen_april_elo, frozen_april_rank = component_probabilities(
            april_components, frozen_april
        )
    else:
        march_elo, march_rank = _component_placeholders(march, march_elo, march_rank)
        april_elo, april_rank = _component_placeholders(april, april_elo, april_rank)
        frozen_march_elo, frozen_march_rank = _component_placeholders(
            frozen_march, frozen_march_elo, frozen_march_rank
        )
        frozen_april_elo, frozen_april_rank = _component_placeholders(
            frozen_april, frozen_april_elo, frozen_april_rank
        )

    march_output = prediction_frame(march, march_probability, march_elo, march_rank)
    april_output = prediction_frame(april, april_probability, april_elo, april_rank)
    frozen_march_output = prediction_frame(
        frozen_march, frozen_march_probability, frozen_march_elo, frozen_march_rank
    )
    frozen_april_output = prediction_frame(
        frozen_april, frozen_april_probability, frozen_april_elo, frozen_april_rank
    )

    # Primary assignment artifact first.
    frozen_april_output.to_csv(
        output_path / "april_predictions_frozen_snapshot.csv",
        index=False,
    )
    april_output.to_csv(output_path / "april_predictions.csv", index=False)
    march_output.to_csv(output_path / "march_predictions.csv", index=False)
    frozen_march_output.to_csv(
        output_path / "march_predictions_frozen_snapshot.csv",
        index=False,
    )
    sequential_features.to_csv(output_path / "engineered_features.csv", index=False)

    march_metrics = evaluate(march["home_win"], march_probability)
    frozen_april_metrics = evaluate(
        frozen_april["home_win"], frozen_april_probability
    )
    sequential_april_metrics = evaluate(april["home_win"], april_probability)

    metrics = {
        "model": selected_spec["model_family"],
        "model_type": selected_spec.get("model_type"),
        "selected_spec": {
            "architecture": selected_spec["architecture"],
            "logistic_c": selected_spec.get("logistic_c"),
            "calibration": selected_spec.get("calibration"),
            "platt_calibration": selected_spec.get("platt_calibration"),
        },
        "primary_april_result": {
            "information_policy": "frozen_snapshot",
            "cutoff": "2026-03-31",
            "metrics": frozen_april_metrics,
            "note": (
                "Primary assignment result: April games scored from the "
                "March 31 performance-state snapshot."
            ),
        },
        "locked_march_test": {
            "information_policy": "sequential_daily",
            "metrics": march_metrics,
            "correct_games": march_metrics["correct_games"],
            "note": (
                "March is a locked pre-final test after pre-March selection. "
                "It is not used for architecture or hyperparameter selection."
            ),
        },
        "sequential_daily": {
            "march": march_metrics,
            "april": sequential_april_metrics,
        },
        "frozen_snapshot": {
            "march": evaluate(frozen_march["home_win"], frozen_march_probability),
            "april": frozen_april_metrics,
        },
        "information_policy_note": (
            "Frozen March 31 April scoring is the primary assignment result. "
            "Sequential daily April scoring is an operational sensitivity analysis."
        ),
        "april_status": (
            "April is the assignment's retrospective scoring period. The executable "
            "selection pipeline uses zero April rows, but April had previously been "
            "viewed during the broader project, so it is not claimed as a pristine "
            "untouched holdout."
        ),
    }
    write_json(artifact_path / "final_metrics.json", metrics)

    if selected_spec.get("model_type") == "direct_logistic":
        coefficients = pd.DataFrame(direct_coefficient_rows(final_fit))
        importance = permutation_importance_direct(march_fit, march)
        blend_models = None
        calibration = Calibration(method="none")
    else:
        blend_models = final_fit
        calibration = Calibration.from_dict(selected_spec["calibration"])
        coefficients = pd.DataFrame(standardized_coefficient_rows(blend_models))
        importance = permutation_importance(march_fit, march, calibration)

    coefficients.to_csv(artifact_path / "coefficient_table.csv", index=False)
    write_json(
        artifact_path / "model_coefficients.json",
        {"rows": coefficients.replace({np.nan: None}).to_dict(orient="records")},
    )
    write_json(
        artifact_path / "model_schema.json",
        {
            "features": DIRECT_FEATURES,
            "target": "home_win",
            "model_type": selected_spec.get("model_type"),
        },
    )

    ablation = ablation_table(
        train_feb,
        march,
        fit_base_models(train_feb, architecture),
        Calibration(elo_weight=0.2, temperature=1.0, shift=0.0),
    )
    ablation.to_csv(artifact_path / "feature_group_ablation.csv", index=False)
    importance.to_csv(artifact_path / "permutation_importance.csv", index=False)

    selected_columns = [
        column for column in CANDIDATE_FEATURES if column in march.columns
    ]
    correlations = march[selected_columns].corr()
    correlations.to_csv(artifact_path / "feature_correlations.csv")

    calibration_bins(march_output).to_csv(
        artifact_path / "march_calibration_bins.csv", index=False
    )
    calibration_bins(frozen_april_output).to_csv(
        artifact_path / "april_calibration_bins.csv", index=False
    )
    pd.DataFrame(feature_dictionary()).to_csv(
        artifact_path / "feature_dictionary.csv", index=False
    )
    march.iloc[[0, len(march) // 2, -1]].to_csv(
        artifact_path / "feature_examples.csv", index=False
    )

    # Calibration diagnostics on locked March and primary frozen April.
    write_json(
        artifact_path / "calibration_diagnostics.json",
        {
            "march_locked_test": calibration_diagnostics(
                march["home_win"].to_numpy(), march_probability
            ),
            "april_frozen_primary": calibration_diagnostics(
                frozen_april["home_win"].to_numpy(), frozen_april_probability
            ),
        },
    )
    extreme_probability_audit(frozen_april_output).to_csv(
        artifact_path / "extreme_probability_audit.csv", index=False
    )

    comparator_cols = {
        "elo": "elo_component_probability",
        "rank": "rank_component_probability",
    }
    bootstrap_frame, bootstrap_summary = date_block_bootstrap(
        frozen_april_output,
        comparator_cols=comparator_cols,
    )
    bootstrap_frame.to_csv(
        artifact_path / "date_block_bootstrap_results.csv", index=False
    )
    write_json(
        artifact_path / "date_block_bootstrap_summary.json", bootstrap_summary
    )

    constant_probability = np.full(len(march), train_feb["home_win"].mean())
    write_json(
        artifact_path / "paired_bootstrap_vs_constant.json",
        bootstrap_metric_difference(
            march["home_win"].to_numpy(),
            march_probability,
            constant_probability,
        ),
    )

    save_figures(
        figure_path,
        march_output,
        frozen_april_output,
        ablation,
        importance,
        correlations,
    )

    joblib.dump(
        {
            "selected_spec": selected_spec,
            "architecture": architecture.to_dict(),
            "fitted_model": final_fit,
        },
        artifact_path / "trained_model.joblib",
    )
    return metrics
