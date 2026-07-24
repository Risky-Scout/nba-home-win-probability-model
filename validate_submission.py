
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from nba_wp.data import audit_games, load_games
from nba_wp.model import evaluate
from nba_wp.periods import derive_periods
from nba_wp.reporting import score_and_write


def compare_metrics(
    expected: dict[str, float],
    observed: dict[str, float],
    tolerance: float = 1e-8,
) -> None:
    for key in ["log_loss", "brier", "auc", "accuracy"]:
        if not np.isclose(expected[key], observed[key], atol=tolerance, rtol=0):
            raise AssertionError(
                f"Metric mismatch for {key}: expected={expected[key]}, observed={observed[key]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate source data, metrics, predictions, and selection provenance."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Also rebuild predictions in a temporary directory and compare game by game.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    games = load_games(args.data)
    audit = audit_games(games, args.data)
    if audit["row_count"] != 1230:
        raise AssertionError(f"Expected 1230 rows, found {audit['row_count']}")
    if audit["pregame_record_reconciliation"]["mismatch_count"] != 0:
        raise AssertionError("Pregame record reconciliation failed.")

    periods = derive_periods(games)
    selection_cutoff = periods.s(periods.selection_max_date)
    march_count = int(((games["game_date"] >= periods.selection_start)
                       & (games["game_date"] < periods.holdout_start)).sum())
    april_count = int((games["game_date"] >= periods.holdout_start).sum())

    selected = json.loads((root / "artifacts" / "selected_spec.json").read_text())
    proof = json.loads((root / "artifacts" / "selection_proof.json").read_text())
    if proof["april_rows_loaded"] != 0:
        raise AssertionError("Model selection loaded April rows.")
    if selected["selection_data_max_date"] != selection_cutoff:
        raise AssertionError(
            f"Unexpected selection cutoff: {selected['selection_data_max_date']} != {selection_cutoff}"
        )

    final_metrics = json.loads(
        (root / "artifacts" / "final_metrics.json").read_text()
    )
    for period in ["march", "april"]:
        predictions = pd.read_csv(
            root / "outputs" / f"{period}_predictions.csv",
            dtype={"game_id": "string"},
        )
        expected_count = march_count if period == "march" else april_count
        if len(predictions) != expected_count:
            raise AssertionError(
                f"{period}: expected {expected_count} predictions, found {len(predictions)}"
            )
        if not predictions["home_win_probability"].between(0, 1).all():
            raise AssertionError(f"{period}: invalid probability.")
        observed = evaluate(
            predictions["home_win"],
            predictions["home_win_probability"].to_numpy(),
        )
        compare_metrics(
            final_metrics["sequential_daily"][period],
            observed,
        )

    if args.recompute:
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            rebuilt = score_and_write(
                games,
                selected,
                temp_root / "outputs",
                temp_root / "artifacts",
                temp_root / "figures",
            )
            for period in ["march", "april"]:
                saved = pd.read_csv(
                    root / "outputs" / f"{period}_predictions.csv",
                    dtype={"game_id": "string"},
                ).sort_values("game_id")
                fresh = pd.read_csv(
                    temp_root / "outputs" / f"{period}_predictions.csv",
                    dtype={"game_id": "string"},
                ).sort_values("game_id")
                if saved["game_id"].tolist() != fresh["game_id"].tolist():
                    raise AssertionError(f"{period}: game IDs differ.")
                if not np.allclose(
                    saved["home_win_probability"],
                    fresh["home_win_probability"],
                    atol=5e-8,
                    rtol=0,
                ):
                    raise AssertionError(f"{period}: probabilities do not reproduce.")
            compare_metrics(
                final_metrics["sequential_daily"]["march"],
                rebuilt["sequential_daily"]["march"],
            )
            compare_metrics(
                final_metrics["sequential_daily"]["april"],
                rebuilt["sequential_daily"]["april"],
            )

    print(
        json.dumps(
            {
                "status": "PASS",
                "data_rows": audit["row_count"],
                "selection_max_date": selected["selection_data_max_date"],
                "april_rows_used_in_selection": proof["april_rows_loaded"],
                "march_predictions": 239,
                "april_predictions": 96,
                "recompute_checked": bool(args.recompute),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
