"""Command-line interface.

The one command a reviewer needs:

    python -m nba_wp.cli predict \
        --data nba-win-probability-data.csv \
        --config configs/model.yaml \
        --output predictions/april_predictions.csv

`predict` uses the locked selected specification (artifacts/current/
selected_spec_pre_march.json). Run `select` first only when regenerating the
whole pipeline from scratch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import architecture_config, benchmarks, load_config, selection_policy
from .data import load_games
from .features import Architecture, build_features
from .model import predict_from_spec

PREDICTION_COLUMNS = [
    "game_id",
    "game_date",
    "away_team",
    "home_team",
    "home_win_probability",
    "away_win_probability",
    "home_fair_decimal_odds",
    "away_fair_decimal_odds",
    "model_version",
    "information_cutoff",
]


def _load_selected_spec(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Selected spec not found at {p}. Run `python -m nba_wp.cli select` first."
        )
    return json.loads(p.read_text())


def _prediction_frame(
    frame: pd.DataFrame,
    probability,
    *,
    model_version: str,
    information_cutoff: str,
) -> pd.DataFrame:
    p_home = pd.Series(probability, index=frame.index).clip(1e-6, 1 - 1e-6)
    out = pd.DataFrame(
        {
            "game_id": frame["game_id"].astype(str),
            "game_date": pd.to_datetime(frame["game_date"]).dt.strftime("%Y-%m-%d"),
            "away_team": frame["away"].astype(str),
            "home_team": frame["home"].astype(str),
            "home_win_probability": p_home.astype(float),
            "away_win_probability": (1.0 - p_home).astype(float),
            "home_fair_decimal_odds": (1.0 / p_home).astype(float),
            "away_fair_decimal_odds": (1.0 / (1.0 - p_home)).astype(float),
            "model_version": model_version,
            "information_cutoff": information_cutoff,
        }
    )
    return out[PREDICTION_COLUMNS]


def cmd_predict(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    games = load_games(args.data)
    spec = _load_selected_spec(args.spec)
    architecture = Architecture.from_dict(spec["architecture"])
    version = str(cfg["output"].get("model_version", "unversioned"))
    cutoff = str(cfg["evaluation"]["forecast"]["information_cutoff"])

    # Frozen primary: state frozen after March 31; fit on games through March.
    frozen = build_features(games, architecture, freeze_date="2026-04-01")
    through_march = frozen[frozen["game_date"] < "2026-04-01"].copy()
    april = frozen[frozen["game_date"] >= "2026-04-01"].copy()
    probability, _, _, _ = predict_from_spec(spec, through_march, april)
    frame = _prediction_frame(
        april, probability, model_version=version, information_cutoff=cutoff
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f"Wrote {len(frame)} frozen April predictions -> {out_path}")

    if args.rolling_output:
        sequential = build_features(games, architecture)
        seq_train = sequential[sequential["game_date"] < "2026-04-01"].copy()
        seq_april = sequential[sequential["game_date"] >= "2026-04-01"].copy()
        seq_prob, _, _, _ = predict_from_spec(spec, seq_train, seq_april)
        rolling = _prediction_frame(
            seq_april,
            seq_prob,
            model_version=version,
            information_cutoff="rolling_daily_scenario",
        )
        roll_path = Path(args.rolling_output)
        roll_path.parent.mkdir(parents=True, exist_ok=True)
        rolling.to_csv(roll_path, index=False)
        print(f"Wrote {len(rolling)} rolling-scenario predictions -> {roll_path}")
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    from .selection import assert_pre_march_selection_frame, run_selection

    cfg = load_config(args.config)
    games = load_games(args.data)
    selection_games = games[
        games["game_date"] < str(cfg["selection"]["cutoff"])
    ].copy()
    assert_pre_march_selection_frame(selection_games)
    spec, fold_table, selection_table = run_selection(
        selection_games,
        architecture_config(cfg),
        selection_policy(cfg),
        benchmarks(cfg),
    )
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "selected_spec_pre_march.json").write_text(json.dumps(spec, indent=2))
    (artifact_dir / "selected_spec.json").write_text(json.dumps(spec, indent=2))
    fold_table.to_csv(artifact_dir / "pre_march_fold_results.csv", index=False)
    selection_table.to_csv(artifact_dir / "pre_march_selection_results.csv", index=False)
    n_direct = int((selection_table["model_type"] == "direct_logistic").sum())
    n_blend = int((selection_table["model_type"] == "blend_platt_challenger").sum())
    proof = {
        "selection_input_rows": int(len(selection_games)),
        "selection_data_end": selection_games["game_date"].max().strftime("%Y-%m-%d"),
        "march_rows_used_in_selection": int(
            (selection_games["game_date"] >= "2026-03-01").sum()
        ),
        "april_rows_used_in_selection": int(
            (selection_games["game_date"] >= "2026-04-01").sum()
        ),
        "selection_metric": str(cfg["selection"]["metric"]),
        "selected_model_type": spec["model_type"],
        "selected_architecture": spec["architecture"]["name"],
        "selected_logistic_c": spec.get("logistic_c"),
        "direct_logistic_candidates": n_direct,
        "blend_challengers": n_blend,
        "total_candidates": int(len(selection_table)),
        "fold_names": [f["name"] for f in selection_policy(cfg)["folds"]],
        "original_submission_tag": "v1-original-submission",
    }
    (artifact_dir / "pre_march_selection_proof.json").write_text(
        json.dumps(proof, indent=2)
    )
    (artifact_dir / "selection_proof.json").write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .evaluation import evaluate_baselines_and_write

    cfg = load_config(args.config)
    games = load_games(args.data)
    spec = _load_selected_spec(args.spec)
    payload = evaluate_baselines_and_write(games, spec, cfg, reports_dir=args.reports_dir)
    march_ll = payload["locked_march_test"]["models"]["selected_model"]["metrics"]["log_loss"]
    april_ll = payload["frozen_april_forecast"]["models"]["selected_model"]["metrics"]["log_loss"]
    print(
        f"metrics.json written: locked March LL={march_ll:.6f}, frozen April LL={april_ll:.6f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nba_wp.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pred = sub.add_parser("predict", help="Write frozen April predictions CSV.")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--config", default="configs/model.yaml")
    p_pred.add_argument("--output", default="predictions/april_predictions.csv")
    p_pred.add_argument(
        "--spec", default="artifacts/current/selected_spec_pre_march.json"
    )
    p_pred.add_argument(
        "--rolling-output",
        default=None,
        help="Optional separate file for the rolling-daily descriptive scenario.",
    )
    p_pred.set_defaults(func=cmd_predict)

    p_sel = sub.add_parser("select", help="Run pre-March model selection.")
    p_sel.add_argument("--data", required=True)
    p_sel.add_argument("--config", default="configs/model.yaml")
    p_sel.add_argument("--artifact-dir", default="artifacts/current")
    p_sel.set_defaults(func=cmd_select)

    p_rep = sub.add_parser("report", help="Write reports/metrics.json (baselines, CIs).")
    p_rep.add_argument("--data", required=True)
    p_rep.add_argument("--config", default="configs/model.yaml")
    p_rep.add_argument("--spec", default="artifacts/current/selected_spec_pre_march.json")
    p_rep.add_argument("--reports-dir", default="reports")
    p_rep.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
