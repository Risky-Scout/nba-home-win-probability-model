
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from nba_wp.data import load_games, sha256_file
from nba_wp.reporting import score_and_write


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Refit through March and score locked March plus primary frozen "
            "March-31 April probabilities."
        )
    )
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--selected-spec",
        default="artifacts/current/selected_spec_pre_march.json",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--artifact-dir", default="artifacts/current")
    parser.add_argument("--figure-dir", default="figures")
    parser.add_argument("--config", default="configs/model.yaml")
    args = parser.parse_args()

    selected_path = Path(args.selected_spec)
    if not selected_path.exists():
        selected_path = Path("artifacts/current/selected_spec.json")
    selected_spec = json.loads(selected_path.read_text())
    games = load_games(args.data)

    started = time.perf_counter()
    metrics = score_and_write(
        games,
        selected_spec,
        args.output_dir,
        args.artifact_dir,
        args.figure_dir,
    )
    elapsed = time.perf_counter() - started

    import subprocess

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    metadata = {
        "git_tag_original_submission": "v1-original-submission",
        "git_commit": git_commit,
        "selected_spec_path": str(selected_path),
        "model_type": selected_spec.get("model_type"),
        "architecture": selected_spec.get("architecture"),
        "logistic_c": selected_spec.get("logistic_c"),
        "regularization": {"logistic_c": selected_spec.get("logistic_c")},
        "training_cutoff_for_april": "2026-03-31",
        "primary_april_policy": "frozen_snapshot",
        "feature_names": selected_spec.get("features", ["elo_diff"]),
        "data_sha256": sha256_file(args.data),
        "score_wall_seconds": elapsed,
        "package_versions": {
            "numpy": __import__("numpy").__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
        },
        "production_claim": "prototype_research_readiness_not_deployable_sportsbook_system",
    }
    Path(args.artifact_dir, "model_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    Path(args.artifact_dir, "runtime_benchmark.json").write_text(
        json.dumps(
            {
                "feature_generation_and_scoring_wall_seconds": elapsed,
                "score_and_write_wall_seconds": elapsed,
                "april_games_scored": 96,
                "includes": [
                    "feature rebuild (sequential + frozen)",
                    "final fit",
                    "locked March and frozen April scoring",
                    "calibration diagnostics",
                    "date-block bootstrap",
                    "figure export",
                ],
                "note": (
                    "Wall-clock for end-to-end score_and_write on this machine. "
                    "Separate model-selection timing is recorded when "
                    "scripts/select_model.py is run."
                ),
                "peak_memory_mb": None,
                "peak_memory_note": "Not instrumented in this prototype run.",
            },
            indent=2,
        )
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
