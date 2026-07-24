"""The deployed Elo architecture must be chosen by the aggregate frozen-policy
one-standard-error stability rule, not a single March snapshot."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "artifacts" / "selected_spec.json"


def _spec():
    return json.loads(SPEC.read_text())


def test_architecture_selection_block_present():
    sel = _spec()["architecture_selection"]
    for key in ("rule", "chosen", "one_se_threshold", "within_one_se", "ranking"):
        assert key in sel


def test_chosen_matches_deployed_architecture():
    spec = _spec()
    assert spec["architecture_selection"]["chosen"] == spec["architecture"]["name"]


def test_chosen_is_within_one_standard_error_band():
    sel = _spec()["architecture_selection"]
    assert sel["chosen"] in sel["within_one_se"]


def test_ranking_is_sorted_by_mean_log_loss():
    ranking = _spec()["architecture_selection"]["ranking"]
    lls = [r["mean_log_loss"] for r in ranking]
    assert lls == sorted(lls)


def test_chosen_is_simplest_within_band():
    """One-SE rule deploys the lowest-K architecture inside the band."""
    sel = _spec()["architecture_selection"]
    ranking = {r["architecture"]: r for r in sel["ranking"]}
    band = [ranking[name] for name in sel["within_one_se"]]
    min_k = min(r["elo_k"] for r in band)
    assert ranking[sel["chosen"]]["elo_k"] == min_k
