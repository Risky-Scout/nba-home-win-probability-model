"""Single-source configuration loading for the project (configs/model.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/model.yaml") -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    required = {"selection", "search_budget", "feature_defaults", "evaluation", "output"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"configs/model.yaml missing sections: {sorted(missing)}")
    return cfg


def architecture_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Shape the YAML into the dict expected by selection.run_selection."""
    challenger = cfg.get("challenger", {"enabled": False})
    return {
        "search_budget": cfg["search_budget"],
        "feature_defaults": cfg["feature_defaults"],
        "blend_challenger": {
            "enabled": bool(challenger.get("enabled", False)),
            "elo_weight": float(challenger.get("elo_weight", 0.2)),
        },
    }


def selection_policy(cfg: dict[str, Any]) -> dict[str, Any]:
    sel = cfg["selection"]
    return {
        "selection_cutoff": str(sel["cutoff"]),
        "method": str(sel.get("method", "expanding_folds")),
        "validation_start": str(sel.get("validation_start", "2026-01-01")),
        "validation_end": str(sel.get("validation_end", "2026-03-01")),
        "selection_metric": str(sel["metric"]),
        "secondary_metrics": list(sel.get("secondary_metrics", [])),
        "descriptive_metrics": list(sel.get("descriptive_metrics", [])),
        "folds": [
            {
                "name": str(f["name"]),
                "train_end": str(f["train_end"]),
                "validation_start": str(f["validation_start"]),
                "validation_end": str(f["validation_end"]),
            }
            for f in sel.get("folds", [])
        ],
    }


def benchmarks(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("benchmarks", {}))
