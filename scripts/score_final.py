
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
    parser.add_argument("--benchmarks", default="configs/benchmarks.json")
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

    benchmarks = json.loads(Path(args.benchmarks).read_text())
    rows = []
    for policy in ["sequential_daily", "frozen_snapshot_sensitivity"]:
        for period in ["march", "april"]:
            for metric in ["log_loss", "brier", "auc", "accuracy"]:
                model_value = metrics[policy][period][metric]
                target = benchmarks[period][metric]
                higher_is_better = metric in {"auc", "accuracy"}
                passed = model_value > target if higher_is_better else model_value < target
                rows.append(
                    {
                        "information_policy": policy,
                        "period": period,
                        "metric": metric,
                        "model_value": model_value,
                        "benchmark": target,
                        "difference_model_minus_benchmark": model_value - target,
                        "higher_is_better": higher_is_better,
                        "passes_numerical_target": passed,
                    }
                )
    import pandas as pd
    pd.DataFrame(rows).to_csv(
        Path(args.artifact_dir) / "benchmark_comparison.csv",
        index=False,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
