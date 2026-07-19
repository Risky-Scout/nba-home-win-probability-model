
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from nba_wp.data import load_games


@pytest.fixture(scope="session")
def real_data_path() -> Path:
    candidates = []
    if os.environ.get("NBA_DATA_PATH"):
        candidates.append(Path(os.environ["NBA_DATA_PATH"]))
    candidates.append(Path("data/nba-win-probability-data.csv"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pytest.skip(
        "Real assignment data not available. Set NBA_DATA_PATH or place the CSV in data/."
    )


@pytest.fixture(scope="session")
def real_games(real_data_path: Path) -> pd.DataFrame:
    return load_games(real_data_path)
