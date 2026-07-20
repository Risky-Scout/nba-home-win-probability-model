from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from nba_wp.data import audit_games, load_games
from nba_wp.model import evaluate
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


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


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
    artifact_dir = root / "artifacts" / "current"
    games = load_games(args.data)
    audit = audit_games(games, args.data)
    if audit["row_count"] != 1230:
        raise AssertionError(f"Expected 1230 rows, found {audit['row_count']}")
    if audit["pregame_record_reconciliation"]["mismatch_count"] != 0:
        raise AssertionError("Pregame record reconciliation failed.")

    selected_path = artifact_dir / "selected_spec_pre_march.json"
    if not selected_path.exists():
        selected_path = artifact_dir / "selected_spec.json"
    selected = json.loads(selected_path.read_text())

    proof_path = artifact_dir / "pre_march_selection_proof.json"
    if not proof_path.exists():
        proof_path = artifact_dir / "selection_proof.json"
    proof = json.loads(proof_path.read_text())

    if proof.get("april_rows_used_in_selection", proof.get("april_rows_loaded", 1)) != 0:
        raise AssertionError("Model selection loaded April rows.")
    if proof.get("march_rows_used_in_selection", 1) != 0:
        raise AssertionError("Model selection loaded March rows.")
    if selected.get("selection_data_end", selected.get("selection_data_max_date")) > "2026-02-28":
        raise AssertionError("Selection cutoff must end on or before 2026-02-28.")

    final_metrics = json.loads((artifact_dir / "final_metrics.json").read_text())

    # Primary April assignment result is the frozen snapshot.
    frozen_april = pd.read_csv(
        root / "outputs" / "april_predictions_frozen_snapshot.csv",
        dtype={"game_id": "string"},
    )
    if len(frozen_april) != 96:
        raise AssertionError(
            f"Frozen April: expected 96 predictions, found {len(frozen_april)}"
        )
    if not frozen_april["home_win_probability"].between(0, 1).all():
        raise AssertionError("Frozen April: invalid probability.")
    observed_frozen = evaluate(
        frozen_april["home_win"],
        frozen_april["home_win_probability"].to_numpy(),
    )
    compare_metrics(final_metrics["primary_april_result"]["metrics"], observed_frozen)

    march = pd.read_csv(
        root / "outputs" / "march_predictions.csv",
        dtype={"game_id": "string"},
    )
    if len(march) != 239:
        raise AssertionError(f"March: expected 239 predictions, found {len(march)}")
    observed_march = evaluate(
        march["home_win"],
        march["home_win_probability"].to_numpy(),
    )
    compare_metrics(final_metrics["locked_march_test"]["metrics"], observed_march)

    recompute_max_abs_diff = None
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
            saved = frozen_april.sort_values("game_id")
            fresh = pd.read_csv(
                temp_root / "outputs" / "april_predictions_frozen_snapshot.csv",
                dtype={"game_id": "string"},
            ).sort_values("game_id")
            if saved["game_id"].tolist() != fresh["game_id"].tolist():
                raise AssertionError("Recomputed frozen April game_id mismatch.")
            diffs = (
                saved["home_win_probability"].to_numpy(dtype=float)
                - fresh["home_win_probability"].to_numpy(dtype=float)
            )
            recompute_max_abs_diff = float(np.max(np.abs(diffs)))
            if not np.allclose(
                saved["home_win_probability"],
                fresh["home_win_probability"],
                atol=1e-10,
                rtol=0,
            ):
                raise AssertionError("Recomputed frozen April probabilities differ.")
            compare_metrics(
                rebuilt["primary_april_result"]["metrics"],
                observed_frozen,
            )

    git_commit = _git_commit(root)
    report = {
        "status": "PASS",
        "git_commit": git_commit,
        "data_rows": audit["row_count"],
        "selection_data_end": selected.get(
            "selection_data_end", selected.get("selection_data_max_date")
        ),
        "march_rows_used_in_selection": proof.get("march_rows_used_in_selection", 0),
        "april_rows_used_in_selection": proof.get(
            "april_rows_used_in_selection", proof.get("april_rows_loaded", 0)
        ),
        "march_predictions": len(march),
        "april_predictions_frozen_primary": len(frozen_april),
        "primary_april_policy": "frozen_snapshot",
        "recompute_checked": bool(args.recompute),
        "recompute_max_abs_probability_diff": recompute_max_abs_diff,
        "artifact_dir": "artifacts/current",
        "original_submission_tag": "v1-original-submission",
        "governance_note": (
            "Pre-March selection is a reconstructed governance path on the "
            "remediation branch; it is not claimed as historical preregistration."
        ),
    }
    print(json.dumps(report, indent=2))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "validation_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
