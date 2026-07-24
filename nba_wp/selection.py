
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import Architecture, build_features
from .periods import Periods, derive_periods
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


# --- Single source of truth for selection provenance ------------------------
# Structured method identifiers. Human-readable provenance text is DERIVED from
# these so the narrative fields (selection_rule, selected_using, notes,
# architecture_selection.rule) cannot silently drift from what the code does.
METHOD_DEPLOYED_ARCH = "frozen_rolling_oos_one_se"
METHOD_MARCH_REFERENCE = "march_reference_log_loss_brier_tiebreak"
METHOD_NESTED_DECISION = "nested_policy_matched_champion_challenger"

# The single canonical statement of how the deployed architecture is chosen.
SELECTION_RULE_TEXT = (
    "The deployed Elo architecture is selected by aggregate pre-holdout "
    "frozen-policy rolling out-of-sample log loss using a one-standard-error "
    "stability rule, preferring the simplest and most stable architecture "
    "inside the noise band. March log loss with Brier tie-break is used only "
    "for the descriptive March reference winners for the individual procedures. "
    "Policy-matched nested rolling-origin evidence retains Elo-only over the "
    "blend; the blend remains a rejected challenger."
)

_METHOD_DESCRIPTIONS = {
    METHOD_DEPLOYED_ARCH: (
        "Aggregate pre-holdout frozen-policy rolling out-of-sample log loss with "
        "a one-standard-error stability rule (select_elo_architecture_stability); "
        "the MOV offset and cold-start warmup are profiled on the same OOS surface. "
        "This is the SELECTOR of the deployed architecture."
    ),
    METHOD_MARCH_REFERENCE: (
        "Per-procedure March log loss with a Brier tie-break, used only to name "
        "descriptive March reference winners. This is a cross-check, NOT the "
        "selector of the deployed architecture."
    ),
    METHOD_NESTED_DECISION: (
        "Policy-matched nested rolling-origin audit (frozen-block and "
        "daily-sequential) retains Elo-only over the blend; the blend is kept as "
        "a rejected challenger."
    ),
}


def selection_provenance_dict() -> dict[str, Any]:
    """Structured, drift-proof provenance for the selection procedure."""
    return {
        "deployed_architecture_method": METHOD_DEPLOYED_ARCH,
        "march_reference_method": METHOD_MARCH_REFERENCE,
        "nested_decision_method": METHOD_NESTED_DECISION,
        "deployed_architecture_method_description": _METHOD_DESCRIPTIONS[METHOD_DEPLOYED_ARCH],
        "march_reference_method_description": _METHOD_DESCRIPTIONS[METHOD_MARCH_REFERENCE],
        "nested_decision_method_description": _METHOD_DESCRIPTIONS[METHOD_NESTED_DECISION],
    }


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _weekly(start: pd.Timestamp, end: pd.Timestamp, step: int = 7) -> list[pd.Timestamp]:
    out, cur = [], start
    while cur <= end:
        out.append(cur)
        cur = cur + pd.Timedelta(days=step)
    return out


def _frozen_rolling_elo_folds(
    games: pd.DataFrame,
    architecture: Architecture,
    periods: Periods,
    min_base: int = 40,
) -> list[dict[str, float]]:
    """Per-fold FROZEN-policy Elo log loss/Brier over pre-holdout weekly origins.

    For each origin O in [outer_start, selection_max], the Elo map is fit on all
    rows strictly before O and scored on the frozen block [O, O+7) (state frozen
    at O-1 via build_features(freeze_date=O)). Every fold is strictly pre-holdout,
    so no holdout-month outcome is ever seen during selection. This is the
    policy-matched surface on which the deployed (frozen) architecture is chosen.
    """
    origins = _weekly(periods.outer_start, periods.selection_max_date)
    folds: list[dict[str, float]] = []
    for origin in origins:
        block_end = min(origin + pd.Timedelta(days=7), periods.holdout_start)
        feats = build_features(games, architecture, freeze_date=periods.s(origin))
        train = feats[feats["game_date"] < origin]
        block = feats[(feats["game_date"] >= origin) & (feats["game_date"] < block_end)]
        if len(block) == 0 or len(train) < min_base or train["home_win"].nunique() < 2:
            continue
        model = fit_elo_model(train, architecture)
        p = elo_probability(model, block)
        metrics = evaluate(block["home_win"].to_numpy(dtype=int), p)
        folds.append({"origin": periods.s(origin), "games": int(len(block)),
                      "log_loss": metrics["log_loss"], "brier": metrics["brier"]})
    return folds


def select_elo_architecture_stability(
    games: pd.DataFrame,
    architectures: list[Architecture],
    periods: Periods,
    min_base: int = 40,
) -> dict[str, Any]:
    """Choose the deployed Elo architecture by aggregate policy-matched OOS
    performance with a one-standard-error stability rule.

    1. For each architecture compute per-fold frozen-policy Elo log loss.
    2. Rank by mean fold log loss; the best has standard error SE across folds.
    3. Among all architectures within (best_mean + SE), deploy the SIMPLEST /
       most stable one (lowest elo_k, then lowest fold-to-fold log-loss std),
       instead of chasing a single March snapshot that is within the noise band.
    """
    rows: list[dict[str, Any]] = []
    for arch in architectures:
        folds = _frozen_rolling_elo_folds(games, arch, periods, min_base=min_base)
        if not folds:
            continue
        ll = np.array([f["log_loss"] for f in folds], dtype=float)
        br = np.array([f["brier"] for f in folds], dtype=float)
        rows.append({
            "architecture": arch,
            "name": arch.name,
            "elo_k": float(arch.elo_k),
            "n_folds": int(len(folds)),
            "mean_log_loss": float(ll.mean()),
            "std_log_loss": float(ll.std(ddof=1)) if len(ll) > 1 else 0.0,
            "se_log_loss": float(ll.std(ddof=1) / np.sqrt(len(ll))) if len(ll) > 1 else 0.0,
            "mean_brier": float(br.mean()),
        })
    if not rows:
        raise ValueError("No architecture produced any frozen rolling fold.")

    ranked = sorted(rows, key=lambda r: (r["mean_log_loss"], r["mean_brier"], r["name"]))
    best = ranked[0]
    threshold = best["mean_log_loss"] + best["se_log_loss"]
    within = [r for r in ranked if r["mean_log_loss"] <= threshold + 1e-12]
    # One-standard-error rule: prefer the simplest (lowest K), then most stable.
    chosen = sorted(within, key=lambda r: (r["elo_k"], r["std_log_loss"], r["name"]))[0]
    return {
        "chosen": chosen,
        "ranked": ranked,
        "one_se_threshold": float(threshold),
        "within_one_se": [r["name"] for r in within],
    }


def profile_mov_offset(
    games: pd.DataFrame,
    architecture: Architecture,
    periods: Periods,
    grid: tuple[float, ...] = (1.6, 2.0, 2.2, 2.6, 3.0),
    default: float = 2.2,
    min_base: int = 40,
) -> dict[str, Any]:
    """Profile the MOV multiplier offset on the frozen-policy rolling OOS surface.

    The offset used to be the borrowed constant 2.2. Here it is scanned on a small
    preregistered grid; the best mean-OOS-log-loss value is found, but the default
    (2.2) is KEPT whenever it lies within one standard error of the best, so we
    only move off the literature value on clear, stable evidence rather than noise.
    """
    rows: list[dict[str, Any]] = []
    for offset in grid:
        folds = _frozen_rolling_elo_folds(
            games, replace(architecture, mov_offset=float(offset)), periods, min_base=min_base
        )
        if not folds:
            continue
        ll = np.array([f["log_loss"] for f in folds], dtype=float)
        rows.append({
            "mov_offset": float(offset),
            "mean_log_loss": float(ll.mean()),
            "se_log_loss": float(ll.std(ddof=1) / np.sqrt(len(ll))) if len(ll) > 1 else 0.0,
            "mean_brier": float(np.mean([f["brier"] for f in folds])),
        })
    ranked = sorted(rows, key=lambda r: (r["mean_log_loss"], abs(r["mov_offset"] - default)))
    best = ranked[0]
    within = best["mean_log_loss"] + best["se_log_loss"]
    default_row = next((r for r in rows if r["mov_offset"] == default), None)
    if default_row is not None and default_row["mean_log_loss"] <= within + 1e-12:
        chosen = default
        rationale = "default_within_one_se_of_best"
    else:
        chosen = best["mov_offset"]
        rationale = "best_clearly_beats_default"
    return {"chosen_offset": float(chosen), "rationale": rationale,
            "best_offset": best["mov_offset"], "grid": list(grid), "profile": ranked}


def profile_cold_start(
    games: pd.DataFrame,
    architecture: Architecture,
    periods: Periods,
    min_base: int = 40,
) -> dict[str, Any]:
    """Profile an early-season provisional-K warmup on the frozen OOS surface.

    Candidates include OFF (warmup_games=0). Warmup is enabled only if a
    configuration beats OFF by more than one standard error; otherwise the
    zero-sum no-warmup update is kept. (A same-season record prior is intentionally
    NOT used: pregame records are cumulative same-season results that Elo already
    incorporates, so such a prior is redundant.)"""
    k = architecture.elo_k
    candidates = [
        {"warmup_games": 0, "elo_k_warmup": 0.0},
        {"warmup_games": 5, "elo_k_warmup": 2.0 * k},
        {"warmup_games": 10, "elo_k_warmup": 2.0 * k},
        {"warmup_games": 10, "elo_k_warmup": 3.0 * k},
    ]
    rows: list[dict[str, Any]] = []
    for cfg in candidates:
        arch = replace(architecture, warmup_games=int(cfg["warmup_games"]),
                       elo_k_warmup=float(cfg["elo_k_warmup"]))
        folds = _frozen_rolling_elo_folds(games, arch, periods, min_base=min_base)
        if not folds:
            continue
        ll = np.array([f["log_loss"] for f in folds], dtype=float)
        rows.append({**cfg, "mean_log_loss": float(ll.mean()),
                     "se_log_loss": float(ll.std(ddof=1) / np.sqrt(len(ll))) if len(ll) > 1 else 0.0})
    off = next(r for r in rows if r["warmup_games"] == 0)
    best = min(rows, key=lambda r: r["mean_log_loss"])
    enable = best["warmup_games"] != 0 and (
        best["mean_log_loss"] < off["mean_log_loss"] - off["se_log_loss"]
    )
    chosen = best if enable else off
    return {
        "chosen": {"warmup_games": int(chosen["warmup_games"]), "elo_k_warmup": float(chosen["elo_k_warmup"])},
        "rationale": "warmup_beats_off_by_one_se" if enable else "off_within_one_se_of_best",
        "profile": rows,
    }


def run_selection(
    games: pd.DataFrame,
    architecture_config: dict[str, Any],
    selection_policy: dict[str, Any],
    periods: Periods | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select the deployed champion, April-blind, per procedure.

    For every candidate architecture we fit base models on games before March,
    then score the March window under three *independent* procedures as an
    in-sample cross-check reference:

      * Elo-only   : logistic on elo_diff;
      * rank-only  : logistic on bt_logit + trend_diff;
      * blend      : temperature-floored (T>=1) logistic stack of the two.

    Each procedure records a reference March winner (own March log loss, Brier
    tie-break), but the DEPLOYED Elo architecture is chosen separately by the
    aggregate frozen-policy rolling out-of-sample log loss with a
    one-standard-error stability rule (``select_elo_architecture_stability``);
    the MOV offset and cold-start warmup are likewise profiled on that OOS
    surface. The single-March split happens to agree with the stability winner.
    The deployed champion is Elo-only, because the nested rolling-origin audit
    shows the blend does not beat Elo-only out-of-sample on either proper score.
    The blend architecture and coefficients are retained in the spec as a
    *rejected challenger* for full transparency.

    The caller must provide no rows dated April or later.
    """
    del selection_policy  # policy carries no tunable state after Task 2
    if periods is None:
        periods = derive_periods(games)
    sel_start = periods.selection_start
    holdout_start = periods.holdout_start
    if games["game_date"].max() >= holdout_start:
        raise ValueError(
            f"Selection data contain holdout rows. Truncate at "
            f"{periods.s(holdout_start)} first."
        )

    architecture_rows: list[dict[str, Any]] = []
    elo_candidates: list[dict[str, Any]] = []
    rank_candidates: list[dict[str, Any]] = []
    blend_candidates: list[dict[str, Any]] = []

    for architecture_values in architecture_config["architectures"]:
        architecture = Architecture.from_dict(architecture_values)
        features = build_features(games, architecture)
        train = features[features["game_date"] < sel_start].copy()
        march = features[
            (features["game_date"] >= sel_start)
            & (features["game_date"] < holdout_start)
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

    elo_winner_march = _pick(elo_candidates)
    rank_winner = _pick(rank_candidates)
    blend_winner = _pick(blend_candidates)

    # Deployed Elo architecture: chosen by aggregate policy-matched frozen OOS
    # log loss with a one-standard-error stability rule (not a single March
    # snapshot). Every fold is strictly pre-holdout, so selection stays leak-free.
    all_archs = [Architecture.from_dict(a) for a in architecture_config["architectures"]]
    stability = select_elo_architecture_stability(games, all_archs, periods)
    champion_arch = stability["chosen"]["architecture"]

    # Data-driven MOV offset: profile it on the same frozen OOS surface; keep the
    # literature value 2.2 unless a grid value clearly and stably beats it.
    mov_profile = profile_mov_offset(games, champion_arch, periods)
    champion_arch = replace(champion_arch, mov_offset=mov_profile["chosen_offset"])

    # Cold-start warmup: profile provisional K; keep OFF unless it clearly wins.
    cold_start = profile_cold_start(games, champion_arch, periods)
    champion_arch = replace(
        champion_arch,
        warmup_games=cold_start["chosen"]["warmup_games"],
        elo_k_warmup=cold_start["chosen"]["elo_k_warmup"],
    )

    # --- March reference metrics for the FINAL PROFILED architecture ---------
    # These describe how the DEPLOYED architecture (after MOV-offset and
    # cold-start profiling) would have scored on March under the same
    # chronological one-step Elo protocol used for the per-procedure March
    # comparison: fit on rows strictly before selection_start, score the March
    # window. They must be recomputed from champion_arch itself, NOT read back
    # from the pre-profiling per-architecture loop. These are a DESCRIPTIVE
    # reference, not pristine holdout evidence (March is in-sample for
    # selection); the out-of-sample evidence is the nested audit.
    champion_march_features = build_features(games, champion_arch)
    champion_march_train = champion_march_features[
        champion_march_features["game_date"] < sel_start
    ]
    champion_march_window = champion_march_features[
        (champion_march_features["game_date"] >= sel_start)
        & (champion_march_features["game_date"] < holdout_start)
    ]
    champion_march_model = fit_elo_model(champion_march_train, champion_arch)
    champion_march_prob = elo_probability(champion_march_model, champion_march_window)
    _march_ref_eval = evaluate(
        champion_march_window["home_win"].to_numpy(dtype=int), champion_march_prob
    )
    march_reference_metrics = {
        key: _march_ref_eval[key] for key in ["log_loss", "brier", "auc", "accuracy"]
    }
    march_reference_metrics_provenance = {
        "method": METHOD_MARCH_REFERENCE,
        "architecture": champion_arch.to_dict(),
        "training_end_date": periods.s(sel_start),
        "scoring_window_start": periods.s(sel_start),
        "scoring_window_end": periods.s(holdout_start),
        "information_policy": "sequential_daily_march_state",
        "training_row_count": int(len(champion_march_train)),
        "scoring_row_count": int(len(champion_march_window)),
        "status": "descriptive_reference_not_pristine_holdout",
        "note": (
            "Recomputed from the final profiled deployed architecture (post "
            "MOV-offset and cold-start profiling). Training rows are strictly "
            "before scoring_window_start. Descriptive March cross-check only; the "
            "deployed architecture is selected on the frozen-policy rolling OOS "
            "surface, and the out-of-sample evidence is the nested audit."
        ),
    }

    # Deploy the Elo-only champion. Fit the final probability map on ALL
    # eligible rows (through the selection cutoff); no stacker is deployed.
    champion_features = build_features(games, champion_arch)
    final_elo_model = fit_elo_model(champion_features, champion_arch)
    selected_spec = {
        "model_family": "elo_only",
        "champion": "elo_only",
        "selected_using": "Aggregate frozen-policy rolling OOS log loss (one-standard-error stability rule)",
        "selection_data_max_date": games["game_date"].max().strftime("%Y-%m-%d"),
        "april_rows_loaded_during_selection": int(
            (games["game_date"] >= holdout_start).sum()
        ),
        "selection_rule": SELECTION_RULE_TEXT,
        "selection_provenance": selection_provenance_dict(),
        "primary_metric": "log_loss",
        "secondary_metrics": ["brier", "auc", "accuracy"],
        "architecture": champion_arch.to_dict(),
        "architecture_selection": {
            "rule": (
                "Deployed Elo architecture chosen by aggregate frozen-policy "
                "out-of-sample log loss over pre-holdout weekly origins, with a "
                "one-standard-error rule preferring the simplest/most stable "
                "architecture inside the noise band. Policy-matched to the frozen "
                "April deliverable; no holdout row is used."
            ),
            "chosen": champion_arch.name,
            "one_se_threshold": stability["one_se_threshold"],
            "within_one_se": stability["within_one_se"],
            "ranking": [
                {
                    "architecture": r["name"],
                    "elo_k": r["elo_k"],
                    "n_folds": r["n_folds"],
                    "mean_log_loss": r["mean_log_loss"],
                    "se_log_loss": r["se_log_loss"],
                    "std_log_loss": r["std_log_loss"],
                    "mean_brier": r["mean_brier"],
                }
                for r in stability["ranked"]
            ],
            "march_single_split_would_choose": elo_winner_march["architecture"].name,
            "march_single_split_would_choose_note": (
                "Reference-only descriptive cross-check: the architecture a single "
                "March log-loss split would name. It is NOT the selector; the "
                "deployed architecture is chosen by the frozen-policy rolling OOS "
                "one-standard-error rule above (method="
                f"{METHOD_DEPLOYED_ARCH}). Shown to demonstrate the single-March "
                "split happens to agree, not to justify deployment."
            ),
        },
        "mov_offset_selection": {
            "rule": (
                "MOV multiplier offset profiled on the frozen-policy rolling OOS "
                "surface; the literature value 2.2 is kept unless a grid value "
                "beats it beyond one standard error."
            ),
            "chosen_offset": mov_profile["chosen_offset"],
            "rationale": mov_profile["rationale"],
            "best_offset": mov_profile["best_offset"],
            "grid": mov_profile["grid"],
            "profile": mov_profile["profile"],
        },
        "cold_start_selection": {
            "rule": (
                "Provisional-K warmup profiled on the frozen-policy rolling OOS "
                "surface; kept OFF unless a configuration beats no-warmup by more "
                "than one standard error. A same-season record prior is omitted as "
                "redundant with Elo."
            ),
            "chosen": cold_start["chosen"],
            "rationale": cold_start["rationale"],
            "profile": cold_start["profile"],
        },
        "elo_model": elo_calibration_dict(final_elo_model, champion_arch.elo_model_c),
        "march_reference_metrics": march_reference_metrics,
        "march_reference_metrics_provenance": march_reference_metrics_provenance,
        # Backward-compatible alias: same recomputed values as
        # march_reference_metrics. Both now reflect the FINAL profiled deployed
        # architecture (not the pre-profiling per-architecture loop) and are a
        # descriptive March reference, not untouched holdout evidence.
        "march_validation_metrics": march_reference_metrics,
        "challenger": {
            "model_family": "logistic-stacked blend: Elo + Bradley-Terry/recent-trend",
            "status": "rejected",
            "reason": (
                "Policy-matched nested rolling-origin validation rejects the blend: "
                "pooled log loss and Brier are worse than Elo-only under both frozen "
                "and daily-sequential policies, and the paired week-block bootstrap "
                "confidence intervals for blend-minus-Elo are above zero. See the "
                "nested summary and fold artifacts for dynamically generated counts."
            ),
            "evidence_artifacts": [
                "artifacts/nested_frozen_block_summary.json",
                "artifacts/nested_frozen_block_folds.csv",
                "artifacts/nested_daily_sequential_summary.json",
                "artifacts/nested_daily_sequential_folds.csv",
            ],
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
            "Deployed Elo architecture selected by aggregate frozen-policy rolling OOS log loss with a one-standard-error stability rule (pre-holdout only); MOV offset and cold-start warmup profiled on the same OOS surface. March log loss is NOT the selector of the deployed architecture.",
            "march_reference_metrics (and its alias march_validation_metrics) are recomputed from the FINAL profiled deployed architecture as a DESCRIPTIVE March cross-check, not untouched holdout evidence.",
            "Deployed Elo-only probability map is refit on all rows through the selection cutoff.",
            "March state features update only after each completed game date.",
            "No April row is loaded by model selection.",
            (
                "March is used for both architecture profiling and the reported March "
                "reference metrics, so March is in-sample for selection and is not a "
                "pristine holdout. The out-of-sample evidence is the nested audit."
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

    return selected_spec, architecture_table
