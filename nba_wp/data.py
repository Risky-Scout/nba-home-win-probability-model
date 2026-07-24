
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_COLUMNS = [
    "game_id",
    "game_date",
    "away",
    "away_wins",
    "away_losses",
    "away_points",
    "away_turnovers",
    "away_fouls",
    "away_rebounds",
    "home",
    "home_wins",
    "home_losses",
    "home_points",
    "home_turnovers",
    "home_fouls",
    "home_rebounds",
]

POSTGAME_COLUMNS = [
    "away_points",
    "away_turnovers",
    "away_fouls",
    "away_rebounds",
    "home_points",
    "home_turnovers",
    "home_fouls",
    "home_rebounds",
]


class DataValidationError(ValueError):
    """Raised when the assignment data violate an expected contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_games(path: str | Path) -> pd.DataFrame:
    """Load and validate the raw game table.

    `game_id` is read as text and zero padded so exported IDs match the
    ten-character NBA identifier convention.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path, dtype={"game_id": "string"})
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in EXPECTED_COLUMNS]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    if extra:
        raise DataValidationError(f"Unexpected columns: {extra}")

    frame = frame[EXPECTED_COLUMNS].copy()
    frame["game_id"] = frame["game_id"].astype("string").str.zfill(10)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="raise")
    frame["away"] = frame["away"].astype("string")
    frame["home"] = frame["home"].astype("string")

    numeric = [c for c in EXPECTED_COLUMNS if c not in {"game_id", "game_date", "away", "home"}]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if frame.isna().any().any():
        nulls = frame.isna().sum()
        raise DataValidationError(f"Missing values detected: {nulls[nulls > 0].to_dict()}")
    if frame["game_id"].duplicated().any():
        dupes = frame.loc[frame["game_id"].duplicated(), "game_id"].tolist()
        raise DataValidationError(f"Duplicate game IDs: {dupes[:10]}")
    if (frame["home"] == frame["away"]).any():
        raise DataValidationError("A team appears as both home and away in the same game.")
    if (frame["home_points"] == frame["away_points"]).any():
        raise DataValidationError("Tied final scores are not expected in NBA data.")

    frame["home_win"] = (frame["home_points"] > frame["away_points"]).astype("int8")
    return frame.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)


def _record_reconciliation(frame: pd.DataFrame) -> dict[str, Any]:
    teams = sorted(set(frame["home"]) | set(frame["away"]))
    wins = {team: 0 for team in teams}
    losses = {team: 0 for team in teams}
    mismatches: list[dict[str, Any]] = []

    for date, day in frame.groupby("game_date", sort=True):
        for row in day.itertuples(index=False):
            checks = [
                ("home_wins", int(row.home_wins), wins[row.home]),
                ("home_losses", int(row.home_losses), losses[row.home]),
                ("away_wins", int(row.away_wins), wins[row.away]),
                ("away_losses", int(row.away_losses), losses[row.away]),
            ]
            for field, observed, expected in checks:
                if observed != expected:
                    mismatches.append(
                        {
                            "game_id": row.game_id,
                            "game_date": date.strftime("%Y-%m-%d"),
                            "field": field,
                            "observed": observed,
                            "expected": expected,
                        }
                    )

        # Update only after every game on the date has been checked.
        for row in day.itertuples(index=False):
            if int(row.home_win) == 1:
                wins[row.home] += 1
                losses[row.away] += 1
            else:
                wins[row.away] += 1
                losses[row.home] += 1

    return {
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:20],
        "final_records": {
            team: {"wins": wins[team], "losses": losses[team], "games": wins[team] + losses[team]}
            for team in teams
        },
    }


def audit_games(frame: pd.DataFrame, source_path: str | Path | None = None) -> dict[str, Any]:
    """Create a machine-readable audit used by the README and validator."""
    teams = sorted(set(frame["home"]) | set(frame["away"]))
    team_game_counts = {
        team: int(((frame["home"] == team) | (frame["away"] == team)).sum())
        for team in teams
    }
    monthly = (
        frame.assign(month=frame["game_date"].dt.strftime("%Y-%m"))
        .groupby("month", sort=True)
        .agg(
            games=("game_id", "size"),
            home_wins=("home_win", "sum"),
            home_win_rate=("home_win", "mean"),
            average_home_margin=("home_points", lambda s: 0.0),
        )
        .reset_index()
    )
    # The grouped margin is easier and less error-prone as a separate series.
    margin_by_month = (
        frame.assign(
            month=frame["game_date"].dt.strftime("%Y-%m"),
            home_margin=frame["home_points"] - frame["away_points"],
        )
        .groupby("month")["home_margin"]
        .mean()
    )
    monthly["average_home_margin"] = monthly["month"].map(margin_by_month).astype(float)

    reconciliation = _record_reconciliation(frame)
    date_team_duplicates = pd.concat(
        [
            frame[["game_date", "home"]].rename(columns={"home": "team"}),
            frame[["game_date", "away"]].rename(columns={"away": "team"}),
        ],
        ignore_index=True,
    ).duplicated(["game_date", "team"]).sum()

    audit: dict[str, Any] = {
        "source_sha256": sha256_file(source_path) if source_path is not None else None,
        "row_count": int(len(frame)),
        "column_count_raw": len(EXPECTED_COLUMNS),
        "expected_column_count_documentation_note": (
            "The task text says fourteen columns, but the listed schema contains "
            "two game-level columns plus seven away and seven home columns: 16 total."
        ),
        "date_min": frame["game_date"].min().strftime("%Y-%m-%d"),
        "date_max": frame["game_date"].max().strftime("%Y-%m-%d"),
        "team_count": len(teams),
        "teams": teams,
        "missing_value_count": int(frame[EXPECTED_COLUMNS].isna().sum().sum()),
        "duplicate_game_id_count": int(frame["game_id"].duplicated().sum()),
        "tied_game_count": int((frame["home_points"] == frame["away_points"]).sum()),
        "same_team_multiple_games_same_date_count": int(date_team_duplicates),
        "home_win_rate": float(frame["home_win"].mean()),
        "team_game_counts": team_game_counts,
        "all_teams_play_82_games": bool(all(value == 82 for value in team_game_counts.values())),
        "monthly_summary": monthly.to_dict(orient="records"),
        "pregame_record_reconciliation": reconciliation,
        "postgame_columns_excluded_from_same_game_model": POSTGAME_COLUMNS,
    }
    return audit
