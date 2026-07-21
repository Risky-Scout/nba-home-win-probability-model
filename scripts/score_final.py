
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nba_wp.data import load_games
from nba_wp.reporting import score_and_write


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refit through March and score the declared April model."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--selected-spec", default="artifacts/selected_spec.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--figure-dir", default="figures")
    args = parser.parse_args()

    selected_spec = json.loads(Path(args.selected_spec).read_text())
    games = load_games(args.data)
    metrics = score_and_write(
        games,
        selected_spec,
        args.output_dir,
        args.artifact_dir,
        args.figure_dir,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
