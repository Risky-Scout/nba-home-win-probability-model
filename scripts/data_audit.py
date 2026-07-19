
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nba_wp.data import audit_games, load_games


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the NBA assignment dataset.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    games = load_games(args.data)
    audit = audit_games(games, args.data)
    (artifact_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2)
    )

    monthly = pd.DataFrame(audit["monthly_summary"])
    monthly.to_csv(artifact_dir / "date_split_summary.csv", index=False)

    team_counts = pd.DataFrame(
        [
            {"team": team, "games": games_count}
            for team, games_count in audit["team_game_counts"].items()
        ]
    )
    team_counts.to_csv(artifact_dir / "team_game_counts.csv", index=False)

    print(json.dumps({
        "row_count": audit["row_count"],
        "date_min": audit["date_min"],
        "date_max": audit["date_max"],
        "team_count": audit["team_count"],
        "record_mismatches": audit["pregame_record_reconciliation"]["mismatch_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
