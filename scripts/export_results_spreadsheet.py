"""Export human-readable April/March result spreadsheets.

Writes a multi-sheet .xlsx (and matching clean .csv files) to the repo
`outputs/` folder. Optionally also copies the workbook to any directories
listed (os.pathsep-separated) in the ``NBA_EXPORT_COPY_DIRS`` environment
variable, for local convenience.

Usage (from repo root):
  python -m scripts.export_results_spreadsheet
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd


def _extra_copy_dirs() -> list[Path]:
    raw = os.environ.get("NBA_EXPORT_COPY_DIRS", "")
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]

READABLE_COLUMNS = {
    "game_date": "Date",
    "home": "Home",
    "away": "Away",
    "home_win_probability": "P(Home Win)",
    "fair_home_decimal_odds": "Fair Home Odds",
    "fair_away_decimal_odds": "Fair Away Odds",
    "home_win": "Home Won?",
    "predicted_home_win": "Pick Correct?",
    "elo_component_probability": "Elo Component",
    "rank_component_probability": "Rank Component",
    "state_policy": "Policy",
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    if "predicted_home_win" in df.columns:
        df["predicted_home_win"] = (
            df["predicted_home_win"].astype(int) == df["home_win"].astype(int)
        )
    df = df.sort_values(["game_date", "home"]).reset_index(drop=True)
    keep = [c for c in READABLE_COLUMNS if c in df.columns]
    out = df[keep].rename(columns=READABLE_COLUMNS)
    for col in ["P(Home Win)", "Elo Component", "Rank Component"]:
        if col in out.columns:
            out[col] = out[col].round(4)
    for col in ["Fair Home Odds", "Fair Away Odds"]:
        if col in out.columns:
            out[col] = out[col].round(2)
    if "Home Won?" in out.columns:
        out["Home Won?"] = out["Home Won?"].map({1: "Yes", 0: "No"})
    if "Pick Correct?" in out.columns:
        out["Pick Correct?"] = out["Pick Correct?"].map({True: "Yes", False: "No"})
    return out


def _summary(root: Path) -> pd.DataFrame:
    import json

    metrics = json.loads((root / "artifacts" / "final_metrics.json").read_text())
    rows = []
    mapping = [
        ("April (frozen, PRIMARY)", metrics["primary_holdout"]["april"]),
        ("March (deploy stacker)", metrics["sequential_daily"]["march"]),
        (
            "April (sequential backtest)",
            metrics["sequential_daily_backtest"]["april"],
        ),
    ]
    for label, m in mapping:
        rows.append(
            {
                "Set": label,
                "Games": int(m["games"]),
                "Log Loss": round(m["log_loss"], 4),
                "Brier": round(m["brier"], 4),
                "AUC": round(m["auc"], 3),
                "Accuracy": f"{m['accuracy']:.1%}",
                "Home Win Rate": f"{m['home_win_rate']:.1%}",
                "Mean P": round(m["mean_probability"], 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs"

    april = _clean(pd.read_csv(outputs / "april_predictions.csv"))
    march = _clean(pd.read_csv(outputs / "march_predictions.csv"))
    backtest = _clean(
        pd.read_csv(outputs / "april_predictions_sequential_backtest.csv")
    )
    summary = _summary(root)

    xlsx = outputs / "NBA_model_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        april.to_excel(writer, sheet_name="April (PRIMARY frozen)", index=False)
        march.to_excel(writer, sheet_name="March", index=False)
        backtest.to_excel(writer, sheet_name="April sequential backtest", index=False)
        for sheet in writer.sheets.values():
            for column_cells in sheet.columns:
                width = max(len(str(c.value)) if c.value is not None else 0
                            for c in column_cells) + 2
                sheet.column_dimensions[column_cells[0].column_letter].width = width

    april.to_csv(outputs / "april_predictions_readable.csv", index=False)
    summary.to_csv(outputs / "results_summary.csv", index=False)

    print(f"Wrote {xlsx}")
    for dest in _extra_copy_dirs():
        if dest.exists():
            shutil.copy(xlsx, dest / "NBA_model_results.xlsx")
            april.to_csv(dest / "april_predictions_readable.csv", index=False)
            print(f"Copied NBA_model_results.xlsx to {dest}")


if __name__ == "__main__":
    main()
