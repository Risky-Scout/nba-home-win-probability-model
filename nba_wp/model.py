
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import Architecture


@dataclass(frozen=True)
class Calibration:
    elo_weight: float
    temperature: float
    shift: float

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Calibration":
        return cls(
            elo_weight=float(values["elo_weight"]),
            temperature=float(values["temperature"]),
            shift=float(values["shift"]),
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class BaseModels:
    elo_model: Pipeline
    rank_model: Pipeline


def sigmoid(value: np.ndarray | float) -> np.ndarray:
    z = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def logit(probability: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def evaluate(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    result = {
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "games": int(len(y)),
        "home_win_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
    }
    return result


def _logistic(c_value: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(c_value),
                    solver="lbfgs",
                    fit_intercept=True,
                    max_iter=10_000,
                    tol=1e-12,
                    random_state=0,
                ),
            ),
        ]
    )


def fit_base_models(train: pd.DataFrame, architecture: Architecture) -> BaseModels:
    elo_model = _logistic(architecture.elo_model_c)
    rank_model = _logistic(architecture.rank_model_c)
    elo_model.fit(train[["elo_diff"]], train["home_win"].astype(int))
    rank_model.fit(
        train[["bt_logit", "trend_diff"]],
        train["home_win"].astype(int),
    )
    return BaseModels(elo_model=elo_model, rank_model=rank_model)


def component_probabilities(
    models: BaseModels,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    elo_probability = models.elo_model.predict_proba(frame[["elo_diff"]])[:, 1]
    rank_probability = models.rank_model.predict_proba(
        frame[["bt_logit", "trend_diff"]]
    )[:, 1]
    return elo_probability, rank_probability


def blend_probabilities(
    elo_probability: np.ndarray,
    rank_probability: np.ndarray,
    calibration: Calibration,
) -> np.ndarray:
    blended_logit = (
        calibration.elo_weight * logit(elo_probability)
        + (1.0 - calibration.elo_weight) * logit(rank_probability)
    )
    return sigmoid(blended_logit / calibration.temperature + calibration.shift)


def predict(
    models: BaseModels,
    frame: pd.DataFrame,
    calibration: Calibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elo_probability, rank_probability = component_probabilities(models, frame)
    final_probability = blend_probabilities(
        elo_probability,
        rank_probability,
        calibration,
    )
    return final_probability, elo_probability, rank_probability


def candidate_beats_benchmarks(
    metrics: dict[str, float],
    benchmarks: dict[str, float],
) -> bool:
    return (
        metrics["log_loss"] < benchmarks["log_loss"]
        and metrics["brier"] < benchmarks["brier"]
        and metrics["auc"] > benchmarks["auc"]
        and metrics["accuracy"] > benchmarks["accuracy"]
    )


def search_calibration(
    y_true: np.ndarray,
    elo_probability: np.ndarray,
    rank_probability: np.ndarray,
    *,
    weights: list[float],
    temperatures: list[float],
    shifts: list[float],
    benchmarks: dict[str, float],
    top_n: int = 100,
) -> dict[str, Any]:
    """Exhaustively search a transparent calibration grid.

    The loop is vectorized over temperature and shift for speed. AUC is
    calculated once per blend weight because positive affine calibration does
    not change ranking.
    """
    y = np.asarray(y_true, dtype=int)
    elo_logit = logit(elo_probability)
    rank_logit = logit(rank_probability)
    temperatures_array = np.asarray(temperatures, dtype=float)
    shifts_array = np.asarray(shifts, dtype=float)

    best_any: dict[str, Any] | None = None
    best_eligible: dict[str, Any] | None = None
    eligible_rows: list[dict[str, Any]] = []
    candidate_count = 0
    eligible_count = 0

    for weight in weights:
        base = float(weight) * elo_logit + (1.0 - float(weight)) * rank_logit
        auc = float(roc_auc_score(y, base))

        z = base[None, :] / temperatures_array[:, None]
        probabilities = sigmoid(
            z[None, :, :] + shifts_array[:, None, None]
        )
        clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        losses = -(
            y[None, None, :] * np.log(clipped)
            + (1 - y[None, None, :]) * np.log(1.0 - clipped)
        ).mean(axis=2)
        briers = ((probabilities - y[None, None, :]) ** 2).mean(axis=2)
        accuracies = ((probabilities >= 0.5) == y[None, None, :]).mean(axis=2)
        candidate_count += int(losses.size)

        best_index = np.unravel_index(int(np.argmin(losses)), losses.shape)
        shift_index, temperature_index = best_index
        row_any = {
            "elo_weight": float(weight),
            "temperature": float(temperatures_array[temperature_index]),
            "shift": float(shifts_array[shift_index]),
            "log_loss": float(losses[best_index]),
            "brier": float(briers[best_index]),
            "auc": auc,
            "accuracy": float(accuracies[best_index]),
        }
        if best_any is None or (
            row_any["log_loss"],
            row_any["brier"],
            -row_any["auc"],
            -row_any["accuracy"],
        ) < (
            best_any["log_loss"],
            best_any["brier"],
            -best_any["auc"],
            -best_any["accuracy"],
        ):
            best_any = row_any

        eligible_mask = (
            (losses < benchmarks["log_loss"])
            & (briers < benchmarks["brier"])
            & (auc > benchmarks["auc"])
            & (accuracies > benchmarks["accuracy"])
        )
        eligible_count += int(eligible_mask.sum())
        if not eligible_mask.any():
            continue

        masked_losses = np.where(eligible_mask, losses, np.inf)
        eligible_index = np.unravel_index(
            int(np.argmin(masked_losses)),
            masked_losses.shape,
        )
        shift_index, temperature_index = eligible_index
        row_eligible = {
            "elo_weight": float(weight),
            "temperature": float(temperatures_array[temperature_index]),
            "shift": float(shifts_array[shift_index]),
            "log_loss": float(losses[eligible_index]),
            "brier": float(briers[eligible_index]),
            "auc": auc,
            "accuracy": float(accuracies[eligible_index]),
        }
        if best_eligible is None or (
            row_eligible["log_loss"],
            row_eligible["brier"],
            -row_eligible["auc"],
            -row_eligible["accuracy"],
        ) < (
            best_eligible["log_loss"],
            best_eligible["brier"],
            -best_eligible["auc"],
            -best_eligible["accuracy"],
        ):
            best_eligible = row_eligible

        # Keep only the best bounded subset for this weight. Enumerating every
        # eligible cell is unnecessary and makes the audit run slower.
        flattened = np.where(eligible_mask, losses, np.inf).ravel()
        finite_count = int(np.isfinite(flattened).sum())
        keep_count = min(top_n, finite_count)
        if keep_count:
            candidate_indices = np.argpartition(
                flattened,
                keep_count - 1,
            )[:keep_count]
            local_rows: list[dict[str, Any]] = []
            for flat_index in candidate_indices:
                shift_index, temperature_index = np.unravel_index(
                    int(flat_index),
                    losses.shape,
                )
                local_rows.append(
                    {
                        "elo_weight": float(weight),
                        "temperature": float(
                            temperatures_array[temperature_index]
                        ),
                        "shift": float(shifts_array[shift_index]),
                        "log_loss": float(
                            losses[shift_index, temperature_index]
                        ),
                        "brier": float(
                            briers[shift_index, temperature_index]
                        ),
                        "auc": auc,
                        "accuracy": float(
                            accuracies[shift_index, temperature_index]
                        ),
                    }
                )
            eligible_rows.extend(local_rows)
        eligible_rows.sort(
            key=lambda row: (
                row["log_loss"],
                row["brier"],
                -row["auc"],
                -row["accuracy"],
            )
        )
        eligible_rows = eligible_rows[:top_n]

    return {
        "candidate_count": candidate_count,
        "eligible_count": eligible_count,
        "best_any": best_any,
        "best_eligible": best_eligible,
        "top_eligible": eligible_rows,
    }


def standardized_coefficient_rows(
    models: BaseModels,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component, pipeline, feature_names in [
        ("elo", models.elo_model, ["elo_diff"]),
        ("rank", models.rank_model, ["bt_logit", "trend_diff"]),
    ]:
        scaler: StandardScaler = pipeline.named_steps["scale"]
        estimator: LogisticRegression = pipeline.named_steps["model"]
        for feature, coefficient, mean, scale in zip(
            feature_names,
            estimator.coef_[0],
            scaler.mean_,
            scaler.scale_,
        ):
            rows.append(
                {
                    "component": component,
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                    "training_mean": float(mean),
                    "training_scale": float(scale),
                    "raw_unit_coefficient": float(coefficient / scale),
                }
            )
        rows.append(
            {
                "component": component,
                "feature": "(intercept)",
                "standardized_coefficient": float(estimator.intercept_[0]),
                "training_mean": np.nan,
                "training_scale": np.nan,
                "raw_unit_coefficient": np.nan,
            }
        )
    return rows
