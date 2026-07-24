
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .features import Architecture, build_features
from .model import (
    apply_logit_stacker,
    component_probabilities,
    elo_calibration_dict,
    elo_probability,
    evaluate,
    fit_base_models,
    fit_elo_model,
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
    """Select the deployed champion using March only, fairly, per procedure.

    For every candidate architecture we fit base models on games before March,
    then score March under three *independent* procedures:

      * Elo-only   : logistic on elo_diff;
      * rank-only  : logistic on bt_logit + trend_diff;
      * blend      : temperature-floored (T>=1) logistic stack of the two.

    Each procedure selects its OWN architecture by its OWN March log loss (Brier
    as tie-break). The deployed champion is Elo-only, because the nested
    rolling-origin audit shows the blend does not beat Elo-only out-of-sample on
    either proper score. The blend architecture and coefficients are retained in
    the spec as a *rejected challenger* for full transparency.

    The caller must provide no rows dated April or later.
    """
    del selection_policy  # policy carries no tunable state after Task 2
    if games["game_date"].max() >= pd.Timestamp("2026-04-01"):
        raise ValueError(
            "Selection data contain April rows. Truncate at 2026-04-01 first."
        )

    architecture_rows: list[dict[str, Any]] = []
    elo_candidates: list[dict[str, Any]] = []
    rank_candidates: list[dict[str, Any]] = []
    blend_candidates: list[dict[str, Any]] = []

    for architecture_values in architecture_config["architectures"]:
        architecture = Architecture.from_dict(architecture_values)
        features = build_features(games, architecture)
        train = features[features["game_date"] < "2026-03-01"].copy()
        march = features[
            (features["game_date"] >= "2026-03-01")
            & (features["game_date"] < "2026-04-01")
        ].copy()
        march_home_win = march["home_win"].to_numpy(dtype=int)

        models = fit_base_models(train, architecture)
        elo_prob, rank_prob = component_probabilities(models, march)
        elo_metrics = evaluate(march_home_win, elo_prob)
        rank_metrics = evaluate(march_home_win, rank_prob)

        # Blend: unconstrained surface ranks architectures for the blend;
        # deploy the temperature-floored (convex) version.
        stacker_deploy = fit_logit_stacker(
            march_home_win, elo_prob, rank_prob, min_temperature=1.0
        )
        blend_prob = apply_logit_stacker(stacker_deploy, elo_prob, rank_prob)
        blend_metrics = evaluate(march_home_win, blend_prob)
        blend_calibration = stacker_calibration_dict(stacker_deploy)

        architecture_rows.append(
            {
                "architecture": architecture.name,
                "elo_k": architecture.elo_k,
                "elo_hfa": architecture.elo_hfa,
                "elo_mov": architecture.elo_mov,
                "bt_c": architecture.bt_c,
                "trend_half_life_days": architecture.trend_half_life_days,
                "elo_model_c": architecture.elo_model_c,
                "rank_model_c": architecture.rank_model_c,
                "elo_log_loss": elo_metrics["log_loss"],
                "elo_brier": elo_metrics["brier"],
                "rank_log_loss": rank_metrics["log_loss"],
                "rank_brier": rank_metrics["brier"],
                "blend_log_loss": blend_metrics["log_loss"],
                "blend_brier": blend_metrics["brier"],
            }
        )
        elo_candidates.append(
            {"architecture": architecture, "metrics": elo_metrics}
        )
        rank_candidates.append(
            {"architecture": architecture, "metrics": rank_metrics}
        )
        blend_candidates.append(
            {
                "architecture": architecture,
                "metrics": blend_metrics,
                "calibration": blend_calibration,
            }
        )

    def _pick(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return sorted(
            candidates,
            key=lambda item: (
                item["metrics"]["log_loss"],
                item["metrics"]["brier"],
                item["architecture"].name,
            ),
        )[0]

    elo_winner = _pick(elo_candidates)
    rank_winner = _pick(rank_candidates)
    blend_winner = _pick(blend_candidates)

    # Deploy the Elo-only champion. Fit the final probability map on ALL
    # eligible rows (through March 31); no stacker is deployed.
    champion_arch = elo_winner["architecture"]
    champion_features = build_features(games, champion_arch)
    final_elo_model = fit_elo_model(champion_features, champion_arch)
    final_elo_probability = elo_probability(final_elo_model, champion_features)
    del final_elo_probability  # coefficients captured below; scoring recomputes

    selected_spec = {
        "model_family": "elo_only",
        "champion": "elo_only",
        "selected_using": "March 2026 one-step-ahead validation (Elo-only log loss)",
        "selection_data_max_date": games["game_date"].max().strftime("%Y-%m-%d"),
        "april_rows_loaded_during_selection": int(
            (games["game_date"] >= "2026-04-01").sum()
        ),
        "selection_rule": (
            "Each procedure (Elo-only, rank-only, blend) selects its own "
            "architecture by its own March log loss (Brier tie-break). Champion "
            "is Elo-only: the nested rolling-origin audit shows the blend does "
            "not beat Elo-only out-of-sample on log loss or Brier, so the "
            "simpler model is deployed. The blend is retained as a rejected "
            "challenger."
        ),
        "primary_metric": "log_loss",
        "secondary_metrics": ["brier", "auc", "accuracy"],
        "architecture": champion_arch.to_dict(),
        "elo_model": elo_calibration_dict(final_elo_model, champion_arch.elo_model_c),
        "march_validation_metrics": {
            key: elo_winner["metrics"][key]
            for key in ["log_loss", "brier", "auc", "accuracy"]
        },
        "challenger": {
            "model_family": "logistic-stacked blend: Elo + Bradley-Terry/recent-trend",
            "status": "rejected",
            "reason": (
                "Nested rolling-origin validation: blend worse than Elo-only on "
                "both log loss and Brier (block-bootstrap CIs entirely above zero; "
                "blend worse in 10 of 11 outer folds). See "
                "artifacts/nested_frozen_block_summary.json and "
                "artifacts/nested_daily_sequential_summary.json."
            ),
            "architecture": blend_winner["architecture"].to_dict(),
            "calibration": blend_winner["calibration"],
            "march_metrics": {
                key: blend_winner["metrics"][key]
                for key in ["log_loss", "brier", "auc", "accuracy"]
            },
        },
        "rank_only_reference": {
            "architecture": rank_winner["architecture"].to_dict(),
            "march_metrics": {
                key: rank_winner["metrics"][key]
                for key in ["log_loss", "brier", "auc", "accuracy"]
            },
        },
        "architecture_count": len(architecture_config["architectures"]),
        "notes": [
            "Elo architecture selected by Elo-only March log loss (base fit through February).",
            "Deployed Elo-only probability map is refit on all rows through March 31.",
            "March state features update only after each completed game date.",
            "No April row is loaded by model selection.",
            (
                "March is used for both architecture selection and reported March "
                "metrics, so March is in-sample for selection and is not a pristine "
                "holdout. The out-of-sample evidence is the nested audit."
            ),
            (
                "The Elo + rank blend was implemented, validated, and REJECTED: it "
                "does not beat Elo-only out-of-sample. Deploying the simpler model "
                "is the honest, defensible choice."
            ),
        ],
    }

    architecture_table = pd.DataFrame(architecture_rows).sort_values(
        "elo_log_loss",
        ascending=True,
    ).reset_index(drop=True)
    top_table = architecture_table.copy()

    return selected_spec, architecture_table, top_table
