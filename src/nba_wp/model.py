
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

DIRECT_FEATURES = ["elo_diff", "bt_logit", "trend_diff"]


@dataclass(frozen=True)
class Calibration:
    """Legacy blend calibration (v1) or identity marker for direct logistic."""

    elo_weight: float = 0.0
    temperature: float = 1.0
    shift: float = 0.0
    method: str = "blend_temperature_shift"

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Calibration":
        return cls(
            elo_weight=float(values.get("elo_weight", 0.0)),
            temperature=float(values.get("temperature", 1.0)),
            shift=float(values.get("shift", 0.0)),
            method=str(values.get("method", "blend_temperature_shift")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlattCalibration:
    intercept: float
    slope: float

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PlattCalibration":
        return cls(intercept=float(values["intercept"]), slope=float(values["slope"]))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def apply(self, raw_probability: np.ndarray) -> np.ndarray:
        return sigmoid(self.intercept + self.slope * logit(raw_probability))


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
    if len(y) == 0:
        raise ValueError("Cannot evaluate empty prediction set.")
    # labels=[0,1] keeps metrics defined on tiny synthetic slices with one class.
    result = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "games": int(len(y)),
        "home_win_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "correct_games": int(((p >= 0.5).astype(int) == y).sum()),
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


def fit_direct_logistic(
    train: pd.DataFrame,
    c_value: float,
    features: list[str] | None = None,
) -> Pipeline:
    cols = list(features) if features else list(DIRECT_FEATURES)
    model = _logistic(c_value)
    model.fit(train[cols], train["home_win"].astype(int))
    # Record fitted schema so prediction always uses the same columns.
    model.feature_names = cols
    return model


def predict_direct_logistic(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    cols = list(getattr(model, "feature_names", DIRECT_FEATURES))
    return model.predict_proba(frame[cols])[:, 1]


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


def raw_blend_probability(
    elo_probability: np.ndarray,
    rank_probability: np.ndarray,
    elo_weight: float,
) -> np.ndarray:
    return blend_probabilities(
        elo_probability,
        rank_probability,
        Calibration(elo_weight=elo_weight, temperature=1.0, shift=0.0),
    )


def fit_platt_calibration(
    raw_probability: np.ndarray,
    y_true: np.ndarray,
) -> PlattCalibration:
    x = logit(raw_probability).reshape(-1, 1)
    y = np.asarray(y_true, dtype=int)
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=10_000,
        tol=1e-12,
        random_state=0,
    )
    model.fit(x, y)
    return PlattCalibration(
        intercept=float(model.intercept_[0]),
        slope=float(model.coef_[0, 0]),
    )


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


def predict_from_spec(
    selected_spec: dict[str, Any],
    train: pd.DataFrame,
    score: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, Any]:
    """Fit on train and score `score` according to the locked specification."""
    architecture = Architecture.from_dict(selected_spec["architecture"])
    model_type = selected_spec.get("model_type", "blend")

    if model_type == "direct_logistic":
        model = fit_direct_logistic(
            train,
            float(selected_spec["logistic_c"]),
            features=selected_spec.get("features"),
        )
        probability = predict_direct_logistic(model, score)
        platt_values = selected_spec.get("platt_calibration")
        if platt_values:
            probability = PlattCalibration.from_dict(platt_values).apply(probability)
        return probability, None, None, model

    models = fit_base_models(train, architecture)
    calibration = Calibration.from_dict(selected_spec["calibration"])
    probability, elo_p, rank_p = predict(models, score, calibration)
    platt_values = selected_spec.get("platt_calibration")
    if platt_values:
        probability = PlattCalibration.from_dict(platt_values).apply(probability)
    return probability, elo_p, rank_p, models


def calibration_diagnostics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    """Fit logit(P(Y=1)) = alpha + gamma * logit(p_hat)."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    # Expected calibration error with fixed bins on probability.
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        if not np.any(mask):
            continue
        ece += (mask.mean()) * abs(y[mask].mean() - p[mask].mean())
    out: dict[str, float] = {
        "expected_calibration_error": float(ece),
        "min_predicted_probability": float(p.min()),
        "max_predicted_probability": float(p.max()),
        "mean_predicted_probability": float(p.mean()),
        "observed_home_win_rate": float(y.mean()),
    }
    if len(np.unique(y)) < 2:
        out["calibration_intercept_alpha"] = float("nan")
        out["calibration_slope_gamma"] = float("nan")
        return out

    x = logit(p).reshape(-1, 1)
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=10_000,
        tol=1e-12,
        random_state=0,
    )
    model.fit(x, y)
    out["calibration_intercept_alpha"] = float(model.intercept_[0])
    out["calibration_slope_gamma"] = float(model.coef_[0, 0])
    return out


def extreme_probability_audit(
    predictions: pd.DataFrame,
    probability_col: str = "home_win_probability",
) -> pd.DataFrame:
    frame = predictions.copy()
    bins = [0.0, 0.10, 0.25, 0.75, 0.90, 1.0000001]
    labels = ["0.00-0.10", "0.10-0.25", "0.25-0.75", "0.75-0.90", "0.90-1.00"]
    frame["probability_range"] = pd.cut(
        frame[probability_col],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    return (
        frame.groupby("probability_range", observed=False)
        .agg(
            games=("game_id", "size"),
            mean_prediction=(probability_col, "mean"),
            actual_home_win_rate=("home_win", "mean"),
        )
        .reset_index()
    )


def date_block_bootstrap(
    predictions: pd.DataFrame,
    *,
    repeats: int = 1000,
    seed: int = 2026,
    probability_col: str = "home_win_probability",
    comparator_cols: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Paired date-block bootstrap of forecasting metrics.

    Optional `comparator_cols` maps a short label to a probability column in
    `predictions` (for example ``{"elo": "elo_probability"}``). For each
    replicate, metric differences are champion minus comparator.
    """
    frame = predictions.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    dates = np.array(sorted(frame["game_date"].unique()))
    rng = np.random.default_rng(seed)
    comparators = comparator_cols or {}
    for label, column in comparators.items():
        if column not in frame.columns:
            raise KeyError(f"Comparator column missing: {column} ({label})")
    # Tiny fixtures (CI synthetic) use a smaller bootstrap budget.
    effective_repeats = repeats if len(dates) >= 8 else min(repeats, 200)
    rows: list[dict[str, Any]] = []
    for i in range(effective_repeats):
        sampled_dates = rng.choice(dates, size=len(dates), replace=True)
        pieces = [frame[frame["game_date"] == date] for date in sampled_dates]
        sample = pd.concat(pieces, ignore_index=True)
        y_sample = sample["home_win"].to_numpy(dtype=int)
        # Date-block samples can collapse to one class on tiny fixtures.
        if len(np.unique(y_sample)) < 2:
            continue
        p_champ = sample[probability_col].to_numpy()
        metrics = evaluate(y_sample, p_champ)
        calib = calibration_diagnostics(y_sample, p_champ)
        row: dict[str, Any] = {"replicate": i, **metrics, **calib}
        for label, column in comparators.items():
            other = evaluate(y_sample, sample[column].to_numpy())
            for metric_name in ("log_loss", "brier", "auc", "accuracy"):
                row[f"delta_{metric_name}_vs_{label}"] = (
                    metrics[metric_name] - other[metric_name]
                )
        rows.append(row)
    if not rows:
        raise ValueError(
            "Date-block bootstrap produced no two-class replicates; "
            "need more outcome diversity across dates."
        )
    result = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "method": "paired_date_block_bootstrap",
        "repeats_requested": repeats,
        "repeats_used": effective_repeats,
        "successful_replicates": int(len(result)),
        "seed": seed,
        "n_dates": int(len(dates)),
        "comparator_cols": comparators,
        "conditioning_note": (
            "Intervals condition on the locked selected specification; "
            "model selection is not re-run inside each bootstrap sample."
        ),
        "metrics": {},
    }
    summary_columns = [
        "log_loss",
        "brier",
        "auc",
        "accuracy",
        "calibration_intercept_alpha",
        "calibration_slope_gamma",
    ]
    for label in comparators:
        summary_columns.extend(
            [
                f"delta_log_loss_vs_{label}",
                f"delta_brier_vs_{label}",
                f"delta_auc_vs_{label}",
                f"delta_accuracy_vs_{label}",
            ]
        )
    for column in summary_columns:
        if column not in result.columns:
            continue
        values = result[column].dropna().to_numpy()
        if len(values) == 0:
            summary["metrics"][column] = {
                "mean": float("nan"),
                "p05": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
            }
            continue
        summary["metrics"][column] = {
            "mean": float(values.mean()),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        }
    return result, summary


def standardized_coefficient_rows(models: BaseModels) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, model, features in [
        ("elo", models.elo_model, ["elo_diff"]),
        ("rank", models.rank_model, ["bt_logit", "trend_diff"]),
    ]:
        scaler: StandardScaler = model.named_steps["scale"]
        logistic: LogisticRegression = model.named_steps["model"]
        for feature, coef, mean, scale in zip(
            features,
            logistic.coef_[0],
            scaler.mean_,
            scaler.scale_,
        ):
            rows.append(
                {
                    "component": name,
                    "feature": feature,
                    "standardized_coefficient": float(coef),
                    "training_mean": float(mean),
                    "training_scale": float(scale),
                    "raw_unit_coefficient": float(coef / scale),
                }
            )
        rows.append(
            {
                "component": name,
                "feature": "(intercept)",
                "standardized_coefficient": float(logistic.intercept_[0]),
                "training_mean": np.nan,
                "training_scale": np.nan,
                "raw_unit_coefficient": np.nan,
            }
        )
    return rows


def direct_coefficient_rows(model: Pipeline) -> list[dict[str, Any]]:
    scaler: StandardScaler = model.named_steps["scale"]
    logistic: LogisticRegression = model.named_steps["model"]
    feature_names = list(getattr(model, "feature_names", DIRECT_FEATURES))
    rows: list[dict[str, Any]] = []
    for feature, coef, mean, scale in zip(
        feature_names,
        logistic.coef_[0],
        scaler.mean_,
        scaler.scale_,
    ):
        rows.append(
            {
                "component": "direct_logistic",
                "feature": feature,
                "standardized_coefficient": float(coef),
                "training_mean": float(mean),
                "training_scale": float(scale),
                "raw_unit_coefficient": float(coef / scale),
            }
        )
    rows.append(
        {
            "component": "direct_logistic",
            "feature": "(intercept)",
            "standardized_coefficient": float(logistic.intercept_[0]),
            "training_mean": np.nan,
            "training_scale": np.nan,
            "raw_unit_coefficient": np.nan,
        }
    )
    return rows


# Kept for v1 validator/scripts that still import the name.
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
    """Legacy v1 dense grid retained only for historical comparison utilities."""
    del benchmarks  # no longer used as a selection gate
    y = np.asarray(y_true, dtype=int)
    elo_logit = logit(elo_probability)
    rank_logit = logit(rank_probability)
    temperatures_array = np.asarray(temperatures, dtype=float)
    shifts_array = np.asarray(shifts, dtype=float)

    best_any: dict[str, Any] | None = None
    eligible_rows: list[dict[str, Any]] = []
    candidate_count = 0

    for weight in weights:
        base = weight * elo_logit + (1.0 - weight) * rank_logit
        z = base[None, :] / temperatures_array[:, None]
        probabilities = sigmoid(z[None, :, :] + shifts_array[:, None, None])
        clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        losses = -(
            y[None, None, :] * np.log(clipped)
            + (1 - y[None, None, :]) * np.log(1.0 - clipped)
        ).mean(axis=2)
        briers = ((clipped - y[None, None, :]) ** 2).mean(axis=2)
        accuracies = (clipped >= 0.5).astype(float).mean(axis=2)
        auc = float(roc_auc_score(y, sigmoid(base))) if len(np.unique(y)) > 1 else float("nan")
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
        if best_any is None or row_any["log_loss"] < best_any["log_loss"]:
            best_any = row_any
        order = np.argsort(losses, axis=None)[:top_n]
        for flat_index in order:
            shift_index, temperature_index = np.unravel_index(int(flat_index), losses.shape)
            eligible_rows.append(
                {
                    "elo_weight": float(weight),
                    "temperature": float(temperatures_array[temperature_index]),
                    "shift": float(shifts_array[shift_index]),
                    "log_loss": float(losses[shift_index, temperature_index]),
                    "brier": float(briers[shift_index, temperature_index]),
                    "auc": auc,
                    "accuracy": float(accuracies[shift_index, temperature_index]),
                }
            )

    return {
        "candidate_count": candidate_count,
        "eligible_count": len(eligible_rows),
        "best_any": best_any,
        "best_eligible": best_any,
        "top_eligible": eligible_rows[:top_n],
    }
