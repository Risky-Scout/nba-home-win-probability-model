# Data

Place the assignment file here (gitignored):

```text
data/nba-win-probability-data.csv
```

1,230 rows, 16 columns: `game_id`, `game_date`, and seven columns per side
(`away*`, `home*`). The wins/losses columns are **pregame** season records
(verified by replay reconciliation); points/turnovers/fouls/rebounds are
**postgame** box-score totals and are never used as same-game features.
