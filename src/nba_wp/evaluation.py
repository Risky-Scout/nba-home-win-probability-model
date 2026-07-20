"""Baselines, paired uncertainty, drift diagnostics, and the metrics artifact.

Everything reported in README / reports is generated here into
``reports/metrics.json`` so numbers cannot drift between documents.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import Architecture, build_features
from .model import (
    DIRECT_FEATURES,
    evaluate,
    fit_direct_logistic,
    predict_direct_logistic,
    predict_from_spec,
)

BASELINE_NAMES = [
    "constant_home_rate",
    "record_difference_logistic",
    "elo_only_logistic",
    "selected_model",
]


def _record_diff(frame: pd.DataFrame) -> np.ndarray:
    """Pregame record-strength difference (home - away).

    Uses the engineered `record_logit_diff` (smoothed win-percentage logit
    difference built from the pregame wins/losses columns).
    """
    return frame["record_logit_diff"].to_numpy(dtype=float)


def _single_feature_logistic(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=10_000,
                    tol=1e-12,
                    random_state=0,
                ),
            ),
        ]
    )
    model.fit(x_train.reshape(-1, 1), y_train)
    return model


def baseline_probabilities(
    train: pd.DataFrame,
    score: pd.DataFrame,
    selected_spec: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return per-baseline predicted probabilities for `score` rows.

    All baselines are fit on `train` only.
    """
    y_train = train["home_win"].to_numpy(dtype=int)
    out: dict[str, np.ndarray] = {}

    out["constant_home_rate"] = np.full(len(score), float(y_train.mean()))

    rd_train = _record_diff(train)
    rd_model = _single_feature_logistic(rd_train, y_train)
    out["record_difference_logistic"] = rd_model.predict_proba(
        _record_diff(score).reshape(-1, 1)
    )[:, 1]

    elo_model = _single_feature_logistic(
        train["elo_diff"].to_numpy(dtype=float), y_train
    )
    out["elo_only_logistic"] = elo_model.predict_proba(
        score["elo_diff"].to_numpy(dtype=float).reshape(-1, 1)
    )[:, 1]

    probability, _, _, _ = predict_from_spec(selected_spec, train, score)
    out["selected_model"] = probability
    return out


def per_game_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_block_bootstrap_delta(
    dates: pd.Series,
    loss_candidate: np.ndarray,
    loss_baseline: np.ndarray,
    *,
    repeats: int = 5000,
    seed: int = 2026,
) -> dict[str, float]:
    """Paired date-block bootstrap of mean(candidate - baseline) per-game loss."""
    d = pd.to_datetime(dates).to_numpy()
    uniq = np.array(sorted(pd.unique(d)))
    idx_by_date = {u: np.where(d == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    delta = loss_candidate - loss_baseline
    means = []
    for _ in range(repeats):
        sample_dates = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_date[u] for u in sample_dates])
        means.append(float(delta[idx].mean()))
    arr = np.asarray(means)
    return {
        "mean_delta": float(delta.mean()),
        "ci_low_2p5": float(np.quantile(arr, 0.025)),
        "ci_high_97p5": float(np.quantile(arr, 0.975)),
        "repeats": int(repeats),
        "seed": int(seed),
    }


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "n": int(mask.sum()),
                "mean_prediction": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            }
        )
    return rows


def probability_band_performance(
    y: np.ndarray, p: np.ndarray
) -> list[dict[str, float]]:
    bands = [(0.0, 0.25), (0.25, 0.4), (0.4, 0.6), (0.6, 0.75), (0.75, 1.0001)]
    rows = []
    for lo, hi in bands:
        mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        rows.append(
            {
                "band": f"{lo:.2f}-{min(hi,1.0):.2f}",
                "n": int(mask.sum()),
                "mean_prediction": float(p[mask].mean()),
                "observed_rate": float(np.asarray(y, dtype=float)[mask].mean()),
                "mean_log_loss": float(per_game_log_loss(y[mask], p[mask]).mean()),
            }
        )
    return rows


def feature_drift_table(features: pd.DataFrame) -> list[dict[str, Any]]:
    """Monthly drift diagnostics for the model features (train dist = Oct–Feb)."""
    frame = features.copy()
    frame["month"] = pd.to_datetime(frame["game_date"]).dt.strftime("%Y-%m")
    train_mask = pd.to_datetime(frame["game_date"]) < "2026-03-01"
    rows: list[dict[str, Any]] = []
    for col in DIRECT_FEATURES:
        mu = float(frame.loc[train_mask, col].mean())
        sd = float(frame.loc[train_mask, col].std(ddof=0)) or 1.0
        for month, grp in frame.groupby("month", sort=True):
            z = (grp[col].astype(float) - mu) / sd
            rows.append(
                {
                    "feature": col,
                    "month": month,
                    "n": int(len(grp)),
                    "mean": float(grp[col].mean()),
                    "std": float(grp[col].std(ddof=0)),
                    "q10": float(grp[col].quantile(0.10)),
                    "q90": float(grp[col].quantile(0.90)),
                    "max_abs_z_vs_train": float(z.abs().max()),
                }
            )
    return rows


def evaluate_baselines_and_write(
    games: pd.DataFrame,
    selected_spec: dict[str, Any],
    cfg: dict[str, Any],
    *,
    reports_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Fit baselines, score locked March + frozen April, write reports/metrics.json."""
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    architecture = Architecture.from_dict(selected_spec["architecture"])
    sequential = build_features(games, architecture)
    frozen_april_features = build_features(games, architecture, freeze_date="2026-04-01")

    train_feb = sequential[sequential["game_date"] < "2026-03-01"].copy()
    march = sequential[
        (sequential["game_date"] >= "2026-03-01")
        & (sequential["game_date"] < "2026-04-01")
    ].copy()
    through_march = sequential[sequential["game_date"] < "2026-04-01"].copy()
    frozen_april = frozen_april_features[
        frozen_april_features["game_date"] >= "2026-04-01"
    ].copy()

    boot_cfg = cfg["evaluation"]["bootstrap"]
    repeats = int(boot_cfg.get("repeats", 5000))
    seed = int(boot_cfg.get("seed", 2026))

    def _score_period(
        train: pd.DataFrame, score: pd.DataFrame, period: str
    ) -> dict[str, Any]:
        probs = baseline_probabilities(train, score, selected_spec)
        y = score["home_win"].to_numpy(dtype=int)
        selected_loss = per_game_log_loss(y, probs["selected_model"])
        block: dict[str, Any] = {"n_games": int(len(score)), "models": {}}
        for name in BASELINE_NAMES:
            p = probs[name]
            metrics = evaluate(score["home_win"], p)
            entry: dict[str, Any] = {"metrics": metrics}
            if name != "selected_model":
                base_loss = per_game_log_loss(y, p)
                entry["paired_delta_log_loss_selected_minus_this"] = (
                    paired_block_bootstrap_delta(
                        score["game_date"],
                        selected_loss,
                        base_loss,
                        repeats=repeats,
                        seed=seed,
                    )
                )
            block["models"][name] = entry
        block["reliability_selected_model"] = reliability_table(
            y, probs["selected_model"]
        )
        block["probability_bands_selected_model"] = probability_band_performance(
            y, probs["selected_model"]
        )
        block["prob_summary_selected_model"] = {
            "min": float(np.min(probs["selected_model"])),
            "max": float(np.max(probs["selected_model"])),
            "mean": float(np.mean(probs["selected_model"])),
        }
        return block

    # Challenger correlation / delta on the locked test (report only, no selection).
    blend_delta: dict[str, Any] = {"available": False}
    try:
        from .model import (
            component_probabilities,
            fit_base_models,
            fit_platt_calibration,
            raw_blend_probability,
        )

        elo_w = float(cfg.get("challenger", {}).get("elo_weight", 0.2))
        models = fit_base_models(train_feb, architecture)
        elo_tr, rank_tr = component_probabilities(models, train_feb)
        raw_tr = raw_blend_probability(elo_tr, rank_tr, elo_w)
        platt = fit_platt_calibration(raw_tr, train_feb["home_win"].to_numpy(dtype=int))
        elo_m, rank_m = component_probabilities(models, march)
        blend_march = platt.apply(raw_blend_probability(elo_m, rank_m, elo_w))
        direct = fit_direct_logistic(train_feb, float(selected_spec["logistic_c"]))
        direct_march = predict_direct_logistic(direct, march)
        y_march = march["home_win"].to_numpy(dtype=int)
        blend_delta = {
            "available": True,
            "prediction_correlation": float(np.corrcoef(blend_march, direct_march)[0, 1]),
            "blend_metrics_march": evaluate(march["home_win"], blend_march),
            "paired_delta_log_loss_selected_minus_blend": paired_block_bootstrap_delta(
                march["game_date"],
                per_game_log_loss(y_march, direct_march),
                per_game_log_loss(y_march, blend_march),
                repeats=repeats,
                seed=seed,
            ),
        }
    except Exception as exc:  # pragma: no cover - defensive report path
        blend_delta = {"available": False, "error": str(exc)}

    drift_rows = feature_drift_table(sequential)
    pd.DataFrame(drift_rows).to_csv(reports_path / "feature_drift_monthly.csv", index=False)

    payload: dict[str, Any] = {
        "generated_by": "nba_wp.evaluation.evaluate_baselines_and_write",
        "selected_spec_summary": {
            "model_type": selected_spec["model_type"],
            "architecture": selected_spec["architecture"]["name"],
            "logistic_c": selected_spec.get("logistic_c"),
            "pre_march_validation_metrics": selected_spec.get(
                "pre_march_validation_metrics"
            ),
            "folds": [f["name"] for f in selected_spec.get("fold_definitions", [])],
        },
        "locked_march_test": _score_period(train_feb, march, "march"),
        "frozen_april_forecast": _score_period(through_march, frozen_april, "april"),
        "challenger_blend_on_locked_march": blend_delta,
        "notes": [
            "Baselines are context only; they never gate selection.",
            "Paired deltas resample game-date blocks (5000 reps).",
            "April was previously viewed in the wider project; it is retrospective, not pristine.",
        ],
    }
    (reports_path / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload
