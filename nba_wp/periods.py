"""Data-driven evaluation periods.

Every split boundary in the pipeline used to be a hard-coded ``"2026-03-01"`` /
``"2026-04-01"`` string. That silently mis-slices if a different season is
supplied. Instead we derive the boundaries from the data itself:

  * ``holdout_start``   = first day of the LAST calendar month in the data
                          (April for this dataset) -> the primary deliverable.
  * ``selection_start`` = first day of the month before the holdout
                          (March) -> architecture selection / in-sample report.

For the supplied 2025-10 .. 2026-04 season this reproduces the historical
constants exactly (selection_start=2026-03-01, holdout_start=2026-04-01,
selection_data_max_date=2026-03-31), so no artifact changes; but dropping in a
new season now "just works".
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Periods:
    selection_start: pd.Timestamp
    holdout_start: pd.Timestamp

    @property
    def selection_max_date(self) -> pd.Timestamp:
        """Last date eligible for selection (day before the holdout month)."""
        return self.holdout_start - pd.Timedelta(days=1)

    @property
    def outer_start(self) -> pd.Timestamp:
        """Default nested rolling-origin outer start: one month before selection."""
        return (self.selection_start - pd.Timedelta(days=1)).replace(day=1)

    def s(self, ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%d")

    def as_dict(self) -> dict[str, str]:
        return {
            "selection_start": self.s(self.selection_start),
            "holdout_start": self.s(self.holdout_start),
            "selection_max_date": self.s(self.selection_max_date),
            "outer_start": self.s(self.outer_start),
        }


def derive_periods(frame: pd.DataFrame) -> Periods:
    """Derive selection/holdout month boundaries from a games frame."""
    max_date = pd.Timestamp(frame["game_date"].max())
    holdout_start = max_date.replace(day=1)
    selection_start = (holdout_start - pd.Timedelta(days=1)).replace(day=1)
    return Periods(selection_start=selection_start, holdout_start=holdout_start)
