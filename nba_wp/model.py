
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import Architecture

# Minimum inner-fold sample size below which every identity-shrunk calibrator
# refuses to fit and returns the numerical identity (no-op) map instead.
MIN_CALIB_N = 50
# Grid of L2 penalty strengths for the identity-shrunk calibrators. Larger
# lambda == MORE shrinkage toward the identity map (delta parameters -> 0);
# lambda == 0 would be the unpenalized MLE (not in the grid). Selection over
# this grid is done ONLY by chronological inner validation, never on outer folds.
LAM_GRID = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


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


def fit_elo_model(train: pd.DataFrame, architecture: Architecture) -> Pipeline:
    """Fit the deployed Elo-only champion: a logistic map on the Elo rating
    differential (elo_diff). This is the model promoted after the nested audit
    rejected the Elo + rank blend."""
    model = _logistic(architecture.elo_model_c)
    model.fit(train[["elo_diff"]], train["home_win"].astype(int))
    return model


def elo_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[["elo_diff"]])[:, 1]


SCHEDULE_FEATURES = ["elo_diff", "rest_advantage", "back_to_back_advantage"]


def fit_schedule_model(train: pd.DataFrame, c_value: float = 0.5) -> Pipeline:
    """Elo + schedule/rest challenger: a strongly-regularized logistic on
    [elo_diff, rest_advantage, back_to_back_advantage]. Small C forces the
    schedule terms to earn their weight; evaluated as a nested challenger only."""
    model = _logistic(c_value)
    model.fit(train[SCHEDULE_FEATURES], train["home_win"].astype(int))
    return model


def schedule_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[SCHEDULE_FEATURES])[:, 1]


def elo_model_rows(model: Pipeline) -> list[dict[str, Any]]:
    """Standardized + raw coefficients for the deployed Elo-only model."""
    scaler: StandardScaler = model.named_steps["scale"]
    estimator: LogisticRegression = model.named_steps["model"]
    coefficient = float(estimator.coef_[0][0])
    scale = float(scaler.scale_[0])
    return [
        {
            "component": "elo",
            "feature": "elo_diff",
            "standardized_coefficient": coefficient,
            "training_mean": float(scaler.mean_[0]),
            "training_scale": scale,
            "raw_unit_coefficient": coefficient / scale,
        },
        {
            "component": "elo",
            "feature": "(intercept)",
            "standardized_coefficient": float(estimator.intercept_[0]),
            "training_mean": float("nan"),
            "training_scale": float("nan"),
            "raw_unit_coefficient": float("nan"),
        },
    ]


def elo_calibration_dict(model: Pipeline, architecture_c: float) -> dict[str, Any]:
    """Serializable deployed Elo-only probability map for selected_spec.json."""
    scaler: StandardScaler = model.named_steps["scale"]
    estimator: LogisticRegression = model.named_steps["model"]
    coefficient = float(estimator.coef_[0][0])
    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])
    intercept = float(estimator.intercept_[0])
    # Raw-unit reconstruction. The pipeline standardizes first, so
    #   p = sigmoid(intercept + coef * (elo_diff - mean) / scale)
    #     = sigmoid(raw_intercept + raw_unit_coefficient * elo_diff)
    # where raw_intercept ABSORBS the -coef*mean/scale shift. Using
    # raw_unit_coefficient with the *standardized* intercept is WRONG; use
    # raw_intercept with it. Both fields are provided so downstream consumers
    # (e.g. the Excel workbook) cannot mis-reconstruct the price.
    raw_unit_coefficient = coefficient / scale
    raw_intercept = intercept - coefficient * mean / scale
    return {
        "method": "logistic_on_elo_diff",
        "feature": "elo_diff",
        "standardized_coefficient": coefficient,
        "intercept": intercept,
        "training_mean": mean,
        "training_scale": scale,
        "raw_unit_coefficient": raw_unit_coefficient,
        "raw_intercept": raw_intercept,
        "C": float(architecture_c),
        "note": (
            "Deployed champion is Elo-only. Two EQUIVALENT forms: "
            "(standardized) p = sigmoid(intercept + standardized_coefficient * "
            "(elo_diff - training_mean)/training_scale); "
            "(raw) p = sigmoid(raw_intercept + raw_unit_coefficient * elo_diff). "
            "Do NOT combine raw_unit_coefficient with the standardized intercept: "
            "raw_intercept = intercept - standardized_coefficient*training_mean/"
            "training_scale already absorbs the centering shift. "
            "The Elo + rank blend was rejected by the nested rolling-origin audit "
            "(worse out-of-sample log loss and Brier)."
        ),
    }


def component_probabilities(
    models: BaseModels,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    elo_probability = models.elo_model.predict_proba(frame[["elo_diff"]])[:, 1]
    rank_probability = models.rank_model.predict_proba(
        frame[["bt_logit", "trend_diff"]]
    )[:, 1]
    return elo_probability, rank_probability


def _refit_stacker_intercept(
    y_true: np.ndarray,
    blended_logit: np.ndarray,
) -> float:
    """MLE intercept for p = sigmoid(blended_logit + c), other coeffs fixed."""
    y = np.asarray(y_true, dtype=float)
    z = np.asarray(blended_logit, dtype=float)

    def nll(c: float) -> float:
        p = np.clip(sigmoid(z + c), 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    # Golden-section style bounded search is enough for a 1-D concave NLL.
    lo, hi = -5.0, 5.0
    for _ in range(60):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2.0)


def fit_logit_stacker(
    y_true: np.ndarray,
    elo_probability: np.ndarray,
    rank_probability: np.ndarray,
    *,
    min_temperature: float = 1.0,
) -> LogisticRegression:
    """Fit the blend by penalized maximum likelihood (logistic stacking).

    An unconstrained stacker can learn temperature T = 1/(a+b) < 1, which
    sharpens component probabilities toward 0/1 and produces many extreme
    prices (e.g. p >= 0.90). By default we enforce T >= ``min_temperature``
    (1.0): keep the unconstrained blend weight w = a/(a+b), set
    a' = w/T_min, b' = (1-w)/T_min, and refit only the intercept on the same
    labels. That removes sharpening while preserving the Elo vs rank mix.
    """
    y = np.asarray(y_true, dtype=int)
    x = np.column_stack([logit(elo_probability), logit(rank_probability)])
    stacker = LogisticRegression(
        C=1.0,
        max_iter=10_000,
        tol=1e-12,
        random_state=0,
    )
    stacker.fit(x, y)

    a = float(stacker.coef_[0, 0])
    b = float(stacker.coef_[0, 1])
    c = float(stacker.intercept_[0])
    stacker.unconstrained_coef_ = np.array([a, b], dtype=float)
    stacker.unconstrained_intercept_ = c
    stacker.temperature_floor_applied_ = False
    stacker.temperature_floor_ = float(min_temperature) if min_temperature is not None else None

    if min_temperature is None:
        # Pure MLE (research / architecture-selection surface only).
        return stacker

    # Deployment contract: a genuine convex logit blend with a temperature
    # floor. That requires BOTH non-negative weights (0 <= w <= 1, so
    # a, b >= 0) AND temperature T = 1/(a+b) >= min_temperature. Enforce both.
    t_floor = float(min_temperature)
    a_pos = max(a, 0.0)
    b_pos = max(b, 0.0)
    total_pos = a_pos + b_pos
    if total_pos <= 0.0:
        # Degenerate MLE (both weights non-positive): fall back to an equal
        # convex blend at the temperature floor rather than emitting a
        # non-convex or negative-temperature map.
        weight = 0.5
        base_temperature = 1.0
    else:
        weight = a_pos / total_pos
        base_temperature = 1.0 / total_pos

    temperature = max(base_temperature, t_floor)
    a_proj = weight / temperature
    b_proj = (1.0 - weight) / temperature

    # No-op fast path: already convex and non-sharpening.
    already_convex = (a >= 0.0) and (b >= 0.0)
    if already_convex and abs(a_proj - a) < 1e-12 and abs(b_proj - b) < 1e-12:
        return stacker

    blended = a_proj * x[:, 0] + b_proj * x[:, 1]
    c_proj = _refit_stacker_intercept(y, blended)
    stacker.coef_ = np.array([[a_proj, b_proj]], dtype=float)
    stacker.intercept_ = np.array([c_proj], dtype=float)
    stacker.temperature_floor_applied_ = True
    return stacker


def apply_logit_stacker(
    stacker: LogisticRegression,
    elo_probability: np.ndarray,
    rank_probability: np.ndarray,
) -> np.ndarray:
    x = np.column_stack([logit(elo_probability), logit(rank_probability)])
    return stacker.predict_proba(x)[:, 1]


def fit_identity_shrunk_calibrator(
    p_inner: np.ndarray,
    y_inner: np.ndarray,
    n0: float = 200.0,
) -> tuple[float, float]:
    """Fit logit(p_cal) = alpha + beta*logit(p) by MLE, then shrink toward the
    IDENTITY map (alpha=0, beta=1) by an empirical-Bayes weight w = n/(n+n0).

    Shrinking toward (0, 1) rather than (0, 0) is deliberate: an underconfident
    base (MLE beta > 1) is sharpened toward calibration, while small samples fall
    back to the identity (no-op) instead of to a constant. Returns (alpha, beta).
    """
    y = np.asarray(y_inner, dtype=int)
    z = logit(np.asarray(p_inner, dtype=float))
    if len(y) < 20 or len(np.unique(y)) < 2:
        return 0.0, 1.0
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10_000).fit(z.reshape(-1, 1), y)
    a_mle, b_mle = float(model.intercept_[0]), float(model.coef_[0, 0])
    w = len(y) / (len(y) + float(n0))
    return a_mle * w, 1.0 + w * (b_mle - 1.0)


def apply_calibrator(alpha: float, beta: float, p: np.ndarray) -> np.ndarray:
    """Apply a logit-linear calibrator: p_cal = sigmoid(alpha + beta*logit(p))."""
    return sigmoid(alpha + beta * logit(np.asarray(p, dtype=float)))


# --------------------------------------------------------------------------- #
# Identity-shrunk calibration challengers (Platt-on-logit and Beta).
#
# Both are penalized toward the IDENTITY (no-op) map: their free "delta"
# parameters are shrunk toward 0 by an L2 penalty of strength ``lam`` (larger
# lam => more shrinkage). They are fit ONLY on inner out-of-fold rows, the
# penalty strength is chosen ONLY by chronological inner validation, and any
# degeneracy (too few rows, one class, optimizer failure, non-finite params, or
# a non-monotonic Beta transform) falls back to the numerical identity so the
# challenger can never do worse than raw Elo by construction on that fold.
# --------------------------------------------------------------------------- #
_EPS = 1e-12


def _clip01(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def _softplus(z: np.ndarray) -> np.ndarray:
    # log(1 + exp(z)), numerically stable.
    return np.logaddexp(0.0, z)


def _mean_logloss_from_logit(lp: np.ndarray, y: np.ndarray) -> float:
    # -mean[ y*log(sigmoid(lp)) + (1-y)*log(1-sigmoid(lp)) ] = mean[ softplus(lp) - y*lp ].
    return float(np.mean(_softplus(lp) - y * lp))


def platt_identity_params() -> dict[str, Any]:
    """The Platt identity (no-op) map: alpha=0, beta=1 (deltas 0)."""
    return {"delta_intercept": 0.0, "delta_slope": 0.0, "alpha": 0.0, "beta": 1.0, "fallback": True}


def fit_platt_identity_shrunk(p_inner: np.ndarray, y_inner: np.ndarray, lam: float) -> dict[str, Any]:
    """Fit an identity-shrunk Platt-on-logit calibrator.

    With z = logit(p_raw):

        logit(p_cal) = z + delta_intercept + delta_slope * z
                     = alpha + beta * z,   alpha = delta_intercept, beta = 1 + delta_slope.

    The map is fit by minimizing mean log loss + ``lam`` * (delta_intercept^2 +
    delta_slope^2), i.e. a MAP / penalized-logistic fit that shrinks toward the
    identity map (delta = 0). Returns delta_intercept, delta_slope, alpha, beta,
    and a ``fallback`` flag (True when the identity was returned unchanged).
    """
    y = np.asarray(y_inner, dtype=int)
    p = _clip01(p_inner)
    if len(y) < MIN_CALIB_N or len(np.unique(y)) < 2:
        return platt_identity_params()
    z = logit(p)
    lam = float(lam)

    def objective(theta: np.ndarray) -> float:
        di, ds = float(theta[0]), float(theta[1])
        lp = di + (1.0 + ds) * z
        return _mean_logloss_from_logit(lp, y) + lam * (di * di + ds * ds)

    try:
        res = minimize(objective, x0=np.zeros(2), method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-12})
    except Exception:
        return platt_identity_params()
    if not res.success and not np.all(np.isfinite(res.x)):
        return platt_identity_params()
    di, ds = float(res.x[0]), float(res.x[1])
    alpha, beta = di, 1.0 + ds
    if not all(np.isfinite(v) for v in (di, ds, alpha, beta)):
        return platt_identity_params()
    return {"delta_intercept": di, "delta_slope": ds, "alpha": alpha, "beta": beta, "fallback": False}


def apply_platt(params: dict[str, Any], p: np.ndarray) -> np.ndarray:
    """Apply an identity-shrunk Platt calibrator: p_cal = sigmoid(alpha + beta*logit(p))."""
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return sigmoid(alpha + beta * logit(_clip01(p)))


def beta_identity_params() -> dict[str, Any]:
    """The Beta identity (no-op) map: a=1, b=1, c=0 (deltas 0)."""
    return {"delta_0": 0.0, "delta_a": 0.0, "delta_b": 0.0,
            "a": 1.0, "b": 1.0, "c": 0.0, "fallback": True, "monotonic": True}


def fit_beta_identity_shrunk(p_inner: np.ndarray, y_inner: np.ndarray, lam: float) -> dict[str, Any]:
    """Fit an identity-shrunk Beta calibrator.

    With l1 = log(p_raw), l0 = log(1 - p_raw):

        logit(p_cal) = logit(p_raw) + delta_0 + delta_a * l1 - delta_b * l0
                     = c + a * l1 - b * l0,
        c = delta_0, a = 1 + delta_a, b = 1 + delta_b.

    (Because logit(p_raw) = l1 - l0.) The deltas are shrunk toward 0 with an L2
    penalty ``lam``. Monotonicity of p_cal in p_raw requires a >= 0 and b >= 0,
    enforced as box constraints (delta_a >= -1, delta_b >= -1). If the fit is
    degenerate or the resulting transform is non-monotonic, the identity map is
    returned. Returns delta_0/delta_a/delta_b, a/b/c, ``fallback`` and
    ``monotonic`` flags.
    """
    y = np.asarray(y_inner, dtype=int)
    p = _clip01(p_inner)
    if len(y) < MIN_CALIB_N or len(np.unique(y)) < 2:
        return beta_identity_params()
    l1 = np.log(p)
    l0 = np.log(1.0 - p)
    lam = float(lam)

    def objective(theta: np.ndarray) -> float:
        d0, da, db = float(theta[0]), float(theta[1]), float(theta[2])
        a, b = 1.0 + da, 1.0 + db
        lp = d0 + a * l1 - b * l0
        return _mean_logloss_from_logit(lp, y) + lam * (d0 * d0 + da * da + db * db)

    try:
        res = minimize(objective, x0=np.zeros(3), method="L-BFGS-B",
                       bounds=[(None, None), (-1.0, None), (-1.0, None)],
                       options={"maxiter": 500, "ftol": 1e-12})
    except Exception:
        return beta_identity_params()
    if not np.all(np.isfinite(res.x)):
        return beta_identity_params()
    d0, da, db = float(res.x[0]), float(res.x[1]), float(res.x[2])
    a, b, c = 1.0 + da, 1.0 + db, d0
    if not all(np.isfinite(v) for v in (d0, da, db, a, b, c)):
        return beta_identity_params()
    monotonic = bool(a >= 0.0 and b >= 0.0)
    if not monotonic:
        # Enforcing monotonicity failed; fall back to identity rather than
        # emit a non-monotonic price map.
        return beta_identity_params()
    return {"delta_0": d0, "delta_a": da, "delta_b": db,
            "a": a, "b": b, "c": c, "fallback": False, "monotonic": True}


def apply_beta(params: dict[str, Any], p: np.ndarray) -> np.ndarray:
    """Apply an identity-shrunk Beta calibrator: p_cal = sigmoid(c + a*log(p) - b*log(1-p))."""
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
    p = _clip01(p)
    lp = c + a * np.log(p) - b * np.log(1.0 - p)
    return sigmoid(lp)


def select_calibrator_lambda(
    fit_fn,
    apply_fn,
    p_inner: np.ndarray,
    y_inner: np.ndarray,
    dates_inner: np.ndarray,
    lam_grid: list[float] | None = None,
    val_fraction: float = 0.30,
) -> dict[str, Any]:
    """Choose the L2 penalty strength by CHRONOLOGICAL inner validation only.

    ``p_inner``/``y_inner``/``dates_inner`` are inner out-of-fold rows (never
    outer-fold rows). Rows are ordered by date, split into an earlier fit
    portion and the last ``val_fraction`` (by date) validation portion; each
    lambda is fit on the fit portion and scored by validation log loss. The
    lambda with the lowest validation log loss is returned (ties broken toward
    MORE shrinkage, i.e. the larger lambda). No outer-fold outcome is ever seen.

    Returns {selected_lambda, val_log_loss_by_lambda, degenerate:bool}. On a
    degenerate split the largest (most shrinking, safest) lambda is returned.
    """
    grid = list(LAM_GRID if lam_grid is None else lam_grid)
    p = _clip01(p_inner)
    y = np.asarray(y_inner, dtype=int)
    d = pd.to_datetime(pd.Series(np.asarray(dates_inner))).to_numpy()
    order = np.argsort(d, kind="stable")
    p, y, d = p[order], y[order], d[order]

    safest = max(grid)
    if len(y) < 2 * MIN_CALIB_N or len(np.unique(y)) < 2:
        return {"selected_lambda": float(safest), "val_log_loss_by_lambda": {}, "degenerate": True}

    unique_dates = np.unique(d)
    if len(unique_dates) < 4:
        return {"selected_lambda": float(safest), "val_log_loss_by_lambda": {}, "degenerate": True}
    cut_idx = int(np.floor((1.0 - val_fraction) * len(unique_dates)))
    cut_idx = min(max(cut_idx, 1), len(unique_dates) - 1)
    cut_date = unique_dates[cut_idx]
    fit_mask = d < cut_date
    val_mask = d >= cut_date

    if (fit_mask.sum() < MIN_CALIB_N or val_mask.sum() < 10
            or len(np.unique(y[fit_mask])) < 2 or len(np.unique(y[val_mask])) < 2):
        return {"selected_lambda": float(safest), "val_log_loss_by_lambda": {}, "degenerate": True}

    scores: dict[str, float] = {}
    for lam in grid:
        params = fit_fn(p[fit_mask], y[fit_mask], lam)
        p_val = apply_fn(params, p[val_mask])
        p_val = np.clip(p_val, _EPS, 1.0 - _EPS)
        scores[str(lam)] = float(log_loss(y[val_mask], p_val, labels=[0, 1]))

    # Lowest validation log loss; tie-break toward larger lambda (more shrinkage).
    best_lam = min(grid, key=lambda lam: (scores[str(lam)], -lam))
    return {"selected_lambda": float(best_lam),
            "val_log_loss_by_lambda": scores, "degenerate": False}


def stacker_calibration_dict(stacker: LogisticRegression) -> dict[str, Any]:
    """Serializable calibration block for selected_spec.json."""
    a = float(stacker.coef_[0, 0])
    b = float(stacker.coef_[0, 1])
    c = float(stacker.intercept_[0])
    unc_a = getattr(stacker, "unconstrained_coef_", None)
    unc_c = getattr(stacker, "unconstrained_intercept_", None)
    floor_applied = bool(getattr(stacker, "temperature_floor_applied_", False))
    payload: dict[str, Any] = {
        "method": "logistic_stack",
        "coef_elo_logit": a,
        "coef_rank_logit": b,
        "intercept": c,
        "temperature": float(1.0 / (a + b)) if (a + b) != 0 else float("inf"),
        "elo_weight": float(a / (a + b)) if (a + b) != 0 else float("nan"),
        "rank_weight": float(b / (a + b)) if (a + b) != 0 else float("nan"),
        "temperature_floor": float(getattr(stacker, "temperature_floor_", 1.0) or 1.0),
        "temperature_floor_applied": floor_applied,
        "note": (
            "blend fitted by penalized maximum likelihood (logistic stacking); "
            "equivalent to (w, T, s) via w=a/(a+b), T=1/(a+b), s=c. "
            "If the unconstrained fit had T < temperature_floor, coefficients "
            "were projected to T = temperature_floor (no sharpening) and the "
            "intercept was refit."
        ),
    }
    if unc_a is not None:
        payload["unconstrained_coef_elo_logit"] = float(unc_a[0])
        payload["unconstrained_coef_rank_logit"] = float(unc_a[1])
    if unc_c is not None:
        payload["unconstrained_intercept"] = float(unc_c)
    return payload


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
