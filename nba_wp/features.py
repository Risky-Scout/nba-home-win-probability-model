
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class Architecture:
    name: str
    elo_k: float
    elo_hfa: float
    elo_mov: str
    bt_c: float
    trend_half_life_days: float
    trend_short_games: int
    elo_model_c: float
    rank_model_c: float
    # Margin-of-victory multiplier constants (FiveThirtyEight form). Defaults
    # reproduce the historical borrowed values; both are now tunable so the
    # offset can be empirically profiled on out-of-sample data instead of
    # inherited unquestioned. multiplier = ln(|margin|+1) * offset /
    # (slope * winner_rating_diff + offset).
    mov_offset: float = 2.2
    mov_slope: float = 0.001
    # Cold-start: a team uses ``elo_k_warmup`` for its first ``warmup_games``
    # games so early-season ratings differentiate faster. warmup_games=0 (default)
    # disables it and the update stays exactly the historical zero-sum form.
    warmup_games: int = 0
    elo_k_warmup: float = 0.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Architecture":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in values.items() if k in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_logit(probability: float | np.ndarray) -> float | np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-9, 1.0 - 1e-9)
    value = np.log(p / (1.0 - p))
    return float(value) if value.ndim == 0 else value


def _fit_bradley_terry(
    prior_games: list[dict[str, Any]],
    team_index: dict[str, int],
    c_value: float,
) -> LogisticRegression | None:
    if len(prior_games) < len(team_index):
        return None

    x = np.zeros((len(prior_games), len(team_index)), dtype=float)
    y = np.empty(len(prior_games), dtype=int)
    for i, game in enumerate(prior_games):
        x[i, team_index[game["home"]]] = 1.0
        x[i, team_index[game["away"]]] = -1.0
        y[i] = int(game["home_win"])

    return LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        fit_intercept=True,
        max_iter=10_000,
        tol=1e-12,
        random_state=0,
    ).fit(x, y)


def _ewma(values: np.ndarray, ages_days: np.ndarray, half_life_days: float) -> float:
    if len(values) == 0:
        return 0.0
    weights = np.power(0.5, ages_days / float(half_life_days))
    return float(np.dot(weights, values) / weights.sum())


def _team_state(
    team: str,
    date: pd.Timestamp,
    histories: dict[str, list[dict[str, Any]]],
    schedule_dates: dict[str, list[pd.Timestamp]],
    architecture: Architecture,
) -> dict[str, float]:
    history = histories[team]
    schedule = schedule_dates[team]
    if not history:
        rest = 7.0 if not schedule else float(min(max((date - schedule[-1]).days, 0), 7))
        return {
            "games": 0.0,
            "record_logit": 0.0,
            "cumulative_margin": 0.0,
            "recent_margin": 0.0,
            "trend": 0.0,
            "turnover_advantage": 0.0,
            "rebound_advantage": 0.0,
            "foul_advantage": 0.0,
            "rest_days": rest,
            "back_to_back": float(rest <= 1.0),
            "games_in_4_days": float(sum((date - d).days < 4 for d in schedule)),
            "games_in_6_days": float(sum((date - d).days < 6 for d in schedule)),
        }

    game_dates = pd.to_datetime([entry["date"] for entry in history])
    ages = np.asarray((date - game_dates).days, dtype=float)
    margins = np.asarray([entry["margin"] for entry in history], dtype=float)
    wins = float(sum(entry["win"] for entry in history))
    games = float(len(history))

    # Beta(4, 4) prior: eight games of neutral information.
    smoothed_win_probability = (wins + 4.0) / (games + 8.0)
    cumulative_margin = float(margins.mean())
    recent_margin = _ewma(
        margins,
        ages,
        architecture.trend_half_life_days,
    )
    short_margin = float(margins[-architecture.trend_short_games :].mean())
    trend = short_margin - recent_margin

    rest = 7.0 if not schedule else float(min(max((date - schedule[-1]).days, 0), 7))
    return {
        "games": games,
        "record_logit": float(_safe_logit(smoothed_win_probability)),
        "cumulative_margin": cumulative_margin,
        "recent_margin": recent_margin,
        "trend": trend,
        "turnover_advantage": float(np.mean([entry["turnover_advantage"] for entry in history])),
        "rebound_advantage": float(np.mean([entry["rebound_advantage"] for entry in history])),
        "foul_advantage": float(np.mean([entry["foul_advantage"] for entry in history])),
        "rest_days": rest,
        "back_to_back": float(rest <= 1.0),
        "games_in_4_days": float(sum((date - d).days < 4 for d in schedule)),
        "games_in_6_days": float(sum((date - d).days < 6 for d in schedule)),
    }


def _elo_multiplier(
    margin: float,
    rating_difference_without_hfa: float,
    mode: str,
    *,
    offset: float = 2.2,
    slope: float = 0.001,
) -> float:
    if mode == "none":
        return 1.0
    if mode == "log":
        denominator = rating_difference_without_hfa * slope + offset
        # Guard against a pathological denominator if a future dataset is extreme.
        denominator = max(0.25, denominator)
        return float(np.log(abs(margin) + 1.0) * (offset / denominator))
    raise ValueError(f"Unsupported Elo margin mode: {mode}")


def build_features(
    games: pd.DataFrame,
    architecture: Architecture,
    *,
    freeze_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build one row of pregame features per game.

    Information order for every date:
      1. Fit/refresh Bradley-Terry from strictly earlier performance results.
      2. Read both teams' pregame states.
      3. Store every feature row for that date.
      4. Update performance and schedule states after all rows for the date.

    With ``freeze_date`` supplied, performance state stops updating on that
    date while schedule dates continue updating. This implements a strict
    month-start snapshot sensitivity analysis without falsely freezing rest.
    """
    frame = games.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)
    freeze = pd.Timestamp(freeze_date) if freeze_date is not None else None

    teams = sorted(set(frame["home"]) | set(frame["away"]))
    team_index = {team: i for i, team in enumerate(teams)}
    ratings = {team: 1500.0 for team in teams}
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    schedule_dates: dict[str, list[pd.Timestamp]] = defaultdict(list)
    prior_games: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    cached_bt: LogisticRegression | None = None
    cached_bt_game_count = -1

    for date, day in frame.groupby("game_date", sort=True):
        if len(prior_games) != cached_bt_game_count:
            cached_bt = _fit_bradley_terry(prior_games, team_index, architecture.bt_c)
            cached_bt_game_count = len(prior_games)

        for row in day.itertuples(index=False):
            home = str(row.home)
            away = str(row.away)
            home_state = _team_state(
                home, date, histories, schedule_dates, architecture
            )
            away_state = _team_state(
                away, date, histories, schedule_dates, architecture
            )

            raw_elo_difference = (
                ratings[home] - ratings[away] + architecture.elo_hfa
            )
            elo_probability = 1.0 / (
                1.0 + 10.0 ** (-raw_elo_difference / 400.0)
            )

            if cached_bt is None:
                bt_logit = 0.0
            else:
                matchup = np.zeros(len(teams), dtype=float)
                matchup[team_index[home]] = 1.0
                matchup[team_index[away]] = -1.0
                bt_logit = float(cached_bt.decision_function([matchup])[0])

            rows.append(
                {
                    "game_id": str(row.game_id),
                    "game_date": date,
                    "away": away,
                    "home": home,
                    "home_win": int(row.home_win),
                    "state_policy": "frozen_snapshot" if freeze is not None else "sequential_daily",
                    "performance_cutoff": (
                        (freeze - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                        if freeze is not None and date >= freeze
                        else (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    ),
                    "elo_diff": raw_elo_difference / 400.0,
                    "elo_probability_raw": elo_probability,
                    "bt_logit": bt_logit,
                    "trend_diff": home_state["trend"] - away_state["trend"],
                    "record_logit_diff": (
                        home_state["record_logit"] - away_state["record_logit"]
                    ),
                    "cumulative_margin_diff": (
                        home_state["cumulative_margin"]
                        - away_state["cumulative_margin"]
                    ),
                    "recent_margin_diff": (
                        home_state["recent_margin"]
                        - away_state["recent_margin"]
                    ),
                    "turnover_advantage_diff": (
                        home_state["turnover_advantage"]
                        - away_state["turnover_advantage"]
                    ),
                    "rebound_advantage_diff": (
                        home_state["rebound_advantage"]
                        - away_state["rebound_advantage"]
                    ),
                    "foul_advantage_diff": (
                        home_state["foul_advantage"]
                        - away_state["foul_advantage"]
                    ),
                    "rest_advantage": (
                        home_state["rest_days"] - away_state["rest_days"]
                    ),
                    "home_rest_days": home_state["rest_days"],
                    "away_rest_days": away_state["rest_days"],
                    "back_to_back_advantage": (
                        away_state["back_to_back"] - home_state["back_to_back"]
                    ),
                    "games_in_4_days_advantage": (
                        away_state["games_in_4_days"]
                        - home_state["games_in_4_days"]
                    ),
                    "games_in_6_days_advantage": (
                        away_state["games_in_6_days"]
                        - home_state["games_in_6_days"]
                    ),
                    "home_games_before": home_state["games"],
                    "away_games_before": away_state["games"],
                }
            )

        # Schedule is observable independently of game outcome and always moves.
        for row in day.itertuples(index=False):
            schedule_dates[str(row.home)].append(date)
            schedule_dates[str(row.away)].append(date)

        # A frozen target batch cannot use its own outcomes to update performance.
        if freeze is not None and date >= freeze:
            continue

        # Performance updates occur only after all same-date rows were created.
        for row in day.itertuples(index=False):
            home = str(row.home)
            away = str(row.away)
            margin = float(row.home_points - row.away_points)
            pregame_home_elo = ratings[home]
            pregame_away_elo = ratings[away]
            expected_home = 1.0 / (
                1.0
                + 10.0
                ** (
                    -(
                        pregame_home_elo
                        - pregame_away_elo
                        + architecture.elo_hfa
                    )
                    / 400.0
                )
            )
            # MOV autocorrelation term must use winner Elo − loser Elo
            # (FiveThirtyEight form). Using home − away for every game
            # suppresses upset adjustments when the favorite is home.
            home_won = int(row.home_win) == 1
            if home_won:
                winner_rating_diff = pregame_home_elo - pregame_away_elo
            else:
                winner_rating_diff = pregame_away_elo - pregame_home_elo
            multiplier = _elo_multiplier(
                abs(margin),
                winner_rating_diff,
                architecture.elo_mov,
                offset=architecture.mov_offset,
                slope=architecture.mov_slope,
            )
            # Per-team provisional K (cold-start). With warmup_games=0 this is a
            # single K for both teams and the update is exactly zero-sum.
            k_home = (
                architecture.elo_k_warmup
                if len(histories[home]) < architecture.warmup_games
                else architecture.elo_k
            )
            k_away = (
                architecture.elo_k_warmup
                if len(histories[away]) < architecture.warmup_games
                else architecture.elo_k
            )
            step = multiplier * (int(row.home_win) - expected_home)
            ratings[home] += k_home * step
            ratings[away] -= k_away * step

            histories[home].append(
                {
                    "date": date,
                    "win": int(row.home_win),
                    "margin": margin,
                    "turnover_advantage": float(
                        row.away_turnovers - row.home_turnovers
                    ),
                    "rebound_advantage": float(
                        row.home_rebounds - row.away_rebounds
                    ),
                    "foul_advantage": float(
                        row.away_fouls - row.home_fouls
                    ),
                }
            )
            histories[away].append(
                {
                    "date": date,
                    "win": 1 - int(row.home_win),
                    "margin": -margin,
                    "turnover_advantage": float(
                        row.home_turnovers - row.away_turnovers
                    ),
                    "rebound_advantage": float(
                        row.away_rebounds - row.home_rebounds
                    ),
                    "foul_advantage": float(
                        row.home_fouls - row.away_fouls
                    ),
                }
            )
            prior_games.append(
                {
                    "home": home,
                    "away": away,
                    "home_win": int(row.home_win),
                }
            )

    return pd.DataFrame(rows)


MODEL_FEATURES = ["elo_diff", "bt_logit", "trend_diff"]

CANDIDATE_FEATURES = [
    "elo_diff",
    "bt_logit",
    "trend_diff",
    "record_logit_diff",
    "cumulative_margin_diff",
    "recent_margin_diff",
    "turnover_advantage_diff",
    "rebound_advantage_diff",
    "foul_advantage_diff",
    "rest_advantage",
    "back_to_back_advantage",
    "games_in_4_days_advantage",
    "games_in_6_days_advantage",
]


def feature_dictionary() -> list[dict[str, str]]:
    return [
        {
            "feature": "elo_diff",
            "formula": "(home Elo - away Elo + home advantage) / 400",
            "source": "Prior game winners and margins",
            "cutoff": "Strictly earlier dates; same-day batch update",
            "role": "Selected Elo component",
        },
        {
            "feature": "bt_logit",
            "formula": "Bradley-Terry home coefficient - away coefficient + intercept",
            "source": "Prior game winners",
            "cutoff": "Strictly earlier dates; same-day batch update",
            "role": "Selected rank component",
        },
        {
            "feature": "trend_diff",
            "formula": "(short margin - EWMA margin)_home - same_away",
            "source": "Prior game point margins",
            "cutoff": "Strictly earlier dates; same-day batch update",
            "role": "Selected rank component",
        },
        {
            "feature": "record_logit_diff",
            "formula": "logit(Beta-smoothed home win pct) - away",
            "source": "Prior wins/losses",
            "cutoff": "Strictly earlier dates",
            "role": "Baseline / ablation",
        },
        {
            "feature": "cumulative_margin_diff",
            "formula": "Home average point margin - away average point margin",
            "source": "Prior game points",
            "cutoff": "Strictly earlier dates",
            "role": "Candidate / ablation",
        },
        {
            "feature": "recent_margin_diff",
            "formula": "Home EWMA point margin - away EWMA point margin",
            "source": "Prior game points",
            "cutoff": "Strictly earlier dates",
            "role": "Candidate / redundancy audit",
        },
        {
            "feature": "turnover_advantage_diff",
            "formula": "Home mean (opponent TOV - own TOV) - away",
            "source": "Prior game turnovers",
            "cutoff": "Strictly earlier dates",
            "role": "Candidate; not selected",
        },
        {
            "feature": "rebound_advantage_diff",
            "formula": "Home mean (own REB - opponent REB) - away",
            "source": "Prior game rebounds",
            "cutoff": "Strictly earlier dates",
            "role": "Candidate; not selected",
        },
        {
            "feature": "foul_advantage_diff",
            "formula": "Home mean (opponent fouls - own fouls) - away",
            "source": "Prior game fouls",
            "cutoff": "Strictly earlier dates",
            "role": "Candidate; not selected",
        },
        {
            "feature": "rest_advantage",
            "formula": "Capped home rest days - capped away rest days",
            "source": "Game dates and team schedule",
            "cutoff": "Schedule dates known before game",
            "role": "Candidate; not selected",
        },
    ]
