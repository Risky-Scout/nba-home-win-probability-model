
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nba_wp.data import load_games
from nba_wp.selection import assert_pre_march_selection_frame, load_json, run_selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the model using expanding-window folds that end before March. "
            "March and April rows are forbidden in selection."
        )
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    games = load_games(args.data)
    selection_games = games[games["game_date"] < "2026-03-01"].copy()
    assert_pre_march_selection_frame(selection_games)

    started = time.perf_counter()
    selected_spec, fold_table, selection_table = run_selection(
        selection_games,
        load_json(config_dir / "architecture_candidates.json"),
        load_json(config_dir / "selection_policy.json"),
        load_json(config_dir / "benchmarks.json"),
    )
    selection_seconds = time.perf_counter() - started

    (artifact_dir / "selected_spec_pre_march.json").write_text(
        json.dumps(selected_spec, indent=2)
    )
    # Compatibility alias used by scoring/validator entry points.
    (artifact_dir / "selected_spec.json").write_text(
        json.dumps(selected_spec, indent=2)
    )
    fold_table.to_csv(artifact_dir / "pre_march_fold_results.csv", index=False)
    selection_table.to_csv(
        artifact_dir / "pre_march_selection_results.csv",
        index=False,
    )
    proof = {
        "selection_input_rows": int(len(selection_games)),
        "selection_data_end": selection_games["game_date"].max().strftime("%Y-%m-%d"),
        "selection_input_max_date": selection_games["game_date"]
        .max()
        .strftime("%Y-%m-%d"),
        "march_rows_used_in_selection": int(
            (selection_games["game_date"] >= "2026-03-01").sum()
        ),
        "april_rows_used_in_selection": int(
            (selection_games["game_date"] >= "2026-04-01").sum()
        ),
        "april_rows_loaded": int(
            (selection_games["game_date"] >= "2026-04-01").sum()
        ),
        "selection_metric": "mean_validation_log_loss",
        "selected_model_type": selected_spec["model_type"],
        "selected_architecture": selected_spec["architecture"]["name"],
        "selected_logistic_c": selected_spec.get("logistic_c"),
        "selection_rule": selected_spec["selection_rule"],
        "original_submission_tag": "v1-original-submission",
    }
    (artifact_dir / "pre_march_selection_proof.json").write_text(
        json.dumps(proof, indent=2)
    )
    (artifact_dir / "selection_proof.json").write_text(json.dumps(proof, indent=2))
    runtime_path = artifact_dir / "runtime_benchmark.json"
    runtime = {}
    if runtime_path.exists():
        runtime = json.loads(runtime_path.read_text())
    runtime["model_selection_wall_seconds"] = selection_seconds
    runtime["selection_candidates_evaluated"] = int(len(selection_table))
    runtime_path.write_text(json.dumps(runtime, indent=2))
    print(json.dumps(selected_spec, indent=2))


if __name__ == "__main__":
    main()
