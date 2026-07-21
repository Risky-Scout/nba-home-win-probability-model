
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


def fit_logit_stacker(
    y_true: np.ndarray,
    elo_probability: np.ndarray,
    rank_probability: np.ndarray,
) -> LogisticRegression:
    """Fit the blend by penalized maximum likelihood (logistic stacking)."""
    x = np.column_stack([logit(elo_probability), logit(rank_probability)])
    stacker = LogisticRegression(
        C=1.0,
        max_iter=10_000,
        tol=1e-12,
        random_state=0,
    )
    stacker.fit(x, np.asarray(y_true, dtype=int))
    return stacker


def apply_logit_stacker(
    stacker: LogisticRegression,
    elo_probability: np.ndarray,
    rank_probability: np.ndarray,
) -> np.ndarray:
    x = np.column_stack([logit(elo_probability), logit(rank_probability)])
    return stacker.predict_proba(x)[:, 1]


def stacker_calibration_dict(stacker: LogisticRegression) -> dict[str, Any]:
    """Serializable calibration block for selected_spec.json."""
    a = float(stacker.coef_[0, 0])
    b = float(stacker.coef_[0, 1])
    c = float(stacker.intercept_[0])
    return {
        "method": "logistic_stack",
        "coef_elo_logit": a,
        "coef_rank_logit": b,
        "intercept": c,
        "note": (
            "blend fitted by penalized maximum likelihood (logistic stacking); "
            "equivalent to the (w, T, s) parameterization via w=a/(a+b), "
            "T=1/(a+b), s=c"
        ),
    }


def stacker_from_calibration_dict(values: dict[str, Any]) -> LogisticRegression:
    """Rebuild a fitted stacker from its persisted coefficients."""
    stacker = LogisticRegression(
        C=1.0,
        max_iter=10_000,
        tol=1e-12,
        random_state=0,
    )
    stacker.classes_ = np.array([0, 1])
    stacker.coef_ = np.array(
        [[float(values["coef_elo_logit"]), float(values["coef_rank_logit"])]]
    )
    stacker.intercept_ = np.array([float(values["intercept"])])
    stacker.n_features_in_ = 2
    return stacker


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
