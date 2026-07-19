# Present the model from Cursor (Mac)

Your local project path:

```text
/Users/josephshackelford/nba-home-win-probability-model
```

## Open the project in Cursor

From Terminal:

```bash
cd "/Users/josephshackelford/nba-home-win-probability-model"
cursor .
```

If `cursor` is not on your PATH, use **File → Open Folder** and select that
directory.

## One-time setup (until validator shows PASS)

Run these in **Cursor’s integrated terminal** (or Terminal.app), from the
project root.

### 0. Prefer the interview-prep branch (has the 90-minute script)

```bash
cd "/Users/josephshackelford/nba-home-win-probability-model"
git fetch origin
git checkout cursor/interview-presentation-prep-7b85
```

If checkout fails, stay on `main` — the model still runs; you just will not
have `docs/PRESENTATION_SCRIPT_90MIN.md` until that branch is available.

### 1. Leave any wrong home virtualenv

```bash
deactivate 2>/dev/null || true
```

### 2. Put the CSV in place

Find it:

```bash
find ~/Downloads ~/Desktop ~/Documents -name "*nba*.csv" 2>/dev/null
```

Copy using the **real** path that `find` prints, for example:

```bash
mkdir -p data
cp ~/Downloads/nba-win-probability-data.csv data/nba-win-probability-data.csv
ls -la data/nba-win-probability-data.csv
```

### 3. Create / activate the project venv

```bash
bash scripts/bootstrap_macos.sh
source .venv/bin/activate
which python
python --version
```

`which python` should look like:

```text
/Users/josephshackelford/nba-home-win-probability-model/.venv/bin/python
```

If it shows `/Users/josephshackelford/.venv/bin/python`, you are in the wrong
environment — `deactivate`, then `source .venv/bin/activate` again from the
project root.

### 4. Safest live command (use this in the interview)

```bash
export NBA_DATA_PATH="$PWD/data/nba-win-probability-data.csv"
python validate_submission.py --root . --data "$NBA_DATA_PATH"
```

Optional stricter check (still avoids re-running the huge selection search):

```bash
python validate_submission.py --root . --data "$NBA_DATA_PATH" --recompute
```

Do **not** run `python run_submission.py --mode select` or `--mode full` live
unless they explicitly ask — selection is slow.

## Checkpoint

| Checkpoint | Meaning | What to do |
|---|---|---|
| **A. Validator shows PASS** | You are presentation-ready | Keep that terminal tab open with PASS visible |
| **B. Validator fails in the clean clone** | Setup incomplete | Fix the error below, then re-run |

### Common Checkpoint B fixes

| Symptom | Fix |
|---|---|
| `can't open file ... validate_submission.py` | You are not in the project folder — `cd` to the path above |
| `FileNotFoundError` for the CSV | Copy the data file into `data/nba-win-probability-data.csv` |
| wrong Python / missing packages | `deactivate` then `bash scripts/bootstrap_macos.sh` and `source .venv/bin/activate` |
| import / sklearn errors | `python -m pip install -r requirements-dev.txt` |

## Tabs to keep open while presenting

Primary narrative:

1. `docs/PRESENTATION_SCRIPT_90MIN.md` (if on the prep branch)
2. `docs/INTERVIEW_WALKTHROUGH.md`
3. `README.md` or `SUMMARY.md`

Code and proof:

4. `nba_wp/features.py`
5. `nba_wp/model.py`
6. `nba_wp/selection.py`
7. `artifacts/selected_spec.json`
8. `artifacts/selection_proof.json`
9. `artifacts/final_metrics.json`
10. `artifacts/feature_group_ablation.csv`
11. `outputs/april_predictions.csv`
12. `validate_submission.py`

## Day-of sequence

1. Open the folder in Cursor.  
2. Integrated terminal → `source .venv/bin/activate`.  
3. Run the validator → confirm **PASS** (Checkpoint A).  
4. Present from `docs/PRESENTATION_SCRIPT_90MIN.md`.  
5. When they ask for a live demo, re-run the same validator (or `--recompute`).
