# Cursor / screen-share setup (interview)

## Goal

In under two minutes: validator green, script open, key artifacts one click away.

## Cloud workspace

```bash
cd /workspace
export NBA_DATA_PATH="$PWD/data/nba-win-probability-data.csv"
python3 validate_submission.py --root . --data "$NBA_DATA_PATH" --recompute
```

## Local Mac (typical)

```bash
cd "/path/to/nba-home-win-probability-model"
cursor .
source .venv/bin/activate
export NBA_DATA_PATH="$PWD/data/nba-win-probability-data.csv"
python validate_submission.py --root . --data "$NBA_DATA_PATH" --recompute
```

Leave `"status": "PASS"` visible.

## Editor layout

**Left / main pane:** `docs/interview/PRESENTATION_SCRIPT_90MIN.md`  
**Right pane:** `docs/SENIOR_QUANT_QA.md`  
**Pinned tabs:** `START_HERE.md`, `docs/interview/PARAMETER_LEDGER.md`,
`artifacts/current/selected_spec_pre_march.json`,
`artifacts/current/model_coefficients.json`,
`predictions/april_predictions.csv`

Full tab list: `docs/interview/FILE_INDEX.md`.

## Do not open as “current”

- `artifacts/v1_legacy/*` (old March search)
- Blend temperature/shift stories from v1 (challenger only now)

## If asked to re-run the model

```bash
make reproduce DATA=data/nba-win-probability-data.csv
```

Authoritative outputs land in `artifacts/current/` and `outputs/`.
