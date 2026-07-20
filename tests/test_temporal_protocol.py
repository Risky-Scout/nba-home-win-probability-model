"""Temporal-protocol contracts: selection can never see March or April."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_wp.config import load_config, selection_policy
from nba_wp.selection import SELECTION_CUTOFF, assert_pre_march_selection_frame


def test_selection_cutoff_is_march_first() -> None:
    assert SELECTION_CUTOFF == pd.Timestamp("2026-03-01")


def test_march_row_rejected_by_selection_guard() -> None:
    frame = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2026-01-05", "2026-03-01"]),
            "home_win": [1, 0],
        }
    )
    with pytest.raises(ValueError, match="March"):
        assert_pre_march_selection_frame(frame)


def test_april_row_rejected_by_selection_guard() -> None:
    frame = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2026-01-05", "2026-04-02"]),
            "home_win": [1, 0],
        }
    )
    with pytest.raises(ValueError):
        assert_pre_march_selection_frame(frame)


def test_config_folds_end_before_march() -> None:
    cfg = load_config("configs/model.yaml")
    policy = selection_policy(cfg)
    for fold in policy["folds"]:
        assert fold["validation_end"] <= "2026-03-01"
        assert fold["train_end"] < "2026-03-01"


def test_committed_selection_proof_has_zero_march_april_rows() -> None:
    proof_path = Path("artifacts/current/pre_march_selection_proof.json")
    if not proof_path.exists():
        pytest.skip("Selection proof not generated yet.")
    proof = json.loads(proof_path.read_text())
    assert proof["march_rows_used_in_selection"] == 0
    assert proof["april_rows_used_in_selection"] == 0
    assert proof["selection_data_end"] <= "2026-02-28"
