
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nba_wp.data import load_games
from nba_wp.periods import derive_periods
from nba_wp.selection import load_json, run_selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the Elo + Bradley-Terry/trend model using March only."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    games = load_games(args.data)
    periods = derive_periods(games)
    selection_games = games[games["game_date"] < periods.holdout_start].copy()

    selected_spec, architecture_table, top_table = run_selection(
        selection_games,
        load_json(config_dir / "architecture_candidates.json"),
        load_json(config_dir / "selection_policy.json"),
        periods=periods,
    )

    (artifact_dir / "selected_spec.json").write_text(
        json.dumps(selected_spec, indent=2)
    )
    architecture_table.to_csv(
        artifact_dir / "march_architecture_results.csv",
        index=False,
    )
    top_table.to_csv(
        artifact_dir / "march_tuning_top_candidates.csv",
        index=False,
    )
    proof = {
        "selection_input_rows": int(len(selection_games)),
        "selection_input_max_date": selection_games["game_date"].max().strftime("%Y-%m-%d"),
        "april_rows_loaded": int((selection_games["game_date"] >= periods.holdout_start).sum()),
        "selected_architecture": selected_spec["architecture"]["name"],
        "model_family": selected_spec["model_family"],
        "deployed_elo_model": selected_spec["elo_model"],
        "rejected_challenger": selected_spec["challenger"]["model_family"],
        "selection_rule": selected_spec["selection_rule"],
    }
    (artifact_dir / "selection_proof.json").write_text(
        json.dumps(proof, indent=2)
    )
    print(json.dumps(selected_spec, indent=2))


if __name__ == "__main__":
    main()
