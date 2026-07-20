"""Fail CI when the committed predictions file is missing or malformed."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED = [
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="predictions/april_predictions.csv")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"FAIL: required predictions file missing: {path}", file=sys.stderr)
        return 1
    frame = pd.read_csv(path)
    if list(frame.columns) != EXPECTED:
        print(f"FAIL: schema mismatch.\nExpected {EXPECTED}\nFound    {list(frame.columns)}", file=sys.stderr)
        return 1
    p = frame["home_win_probability"].to_numpy(dtype=float)
    q = frame["away_win_probability"].to_numpy(dtype=float)
    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        print("FAIL: non-finite probabilities.", file=sys.stderr)
        return 1
    if not (((p > 0) & (p < 1)).all() and ((q > 0) & (q < 1)).all()):
        print("FAIL: probabilities outside (0, 1).", file=sys.stderr)
        return 1
    if not np.allclose(p + q, 1.0, atol=1e-9):
        print("FAIL: home + away probabilities do not sum to 1.", file=sys.stderr)
        return 1
    if not np.allclose(frame["home_fair_decimal_odds"], 1.0 / p, atol=1e-6):
        print("FAIL: home fair odds are not 1/p.", file=sys.stderr)
        return 1
    if len(frame) != 96:
        print(f"FAIL: expected 96 April games, found {len(frame)}.", file=sys.stderr)
        return 1
    print(f"OK: {path} schema, bounds, odds, and row count verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
