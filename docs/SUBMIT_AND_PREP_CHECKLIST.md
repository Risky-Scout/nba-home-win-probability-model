# Submission and interview prep checklist

## Send to recruiter (≥ 24 hours before)

1. Push this repository to GitHub (private is fine if they can access it).
2. Email the recruiter:
   - GitHub repository URL
   - one-paragraph summary (use `SUMMARY.md`)
   - optional: attach `SUMMARY.md` as a brief write-up
3. Confirm they can clone and run:

```bash
bash scripts/bootstrap_macos.sh   # or: python -m venv .venv && pip install -r requirements-dev.txt
source .venv/bin/activate
# place nba-win-probability-data.csv under data/
make verify DATA=data/nba-win-probability-data.csv
```

Do **not** commit the CSV if the recruiter already has it; `data/*.csv` is
gitignored. Tell them the expected local path.

## Night before

- [ ] Clone a clean copy and run `validate_submission.py` successfully
- [ ] Green-check the self-scorecard in `docs/EVALUATION_MATRIX_PREP.md`
- [ ] Memorize the one-pager: `docs/PRESENTATION_ONE_PAGER.md`
- [ ] Rehearse `docs/PRESENTATION_SCRIPT_90MIN.md` once out loud (full pass)
- [ ] Know the condensed backup `docs/PRESENTATION_SCRIPT.md` if they ask for a short version
- [ ] Drill Q&A: AUC miss, March selection bias, correct blend formula, April 5 info set
- [ ] Open the tab checklist from the presentation script
- [ ] Know your live demo command by heart
- [ ] Sleep

## Day of interview

Follow `docs/CURSOR_PRESENTATION_SETUP.md`. Short version:

```bash
cd "/Users/josephshackelford/nba-home-win-probability-model"
cursor .
# Cursor integrated terminal:
source .venv/bin/activate
export NBA_DATA_PATH="$PWD/data/nba-win-probability-data.csv"
python validate_submission.py --root . --data "$NBA_DATA_PATH"
```

Leave Checkpoint **A** (`"status": "PASS"`) visible when you join the call.

## Presentation order (do not skip)

Follow `docs/PRESENTATION_SCRIPT_90MIN.md` Acts I–VII. Never skip:

1. Pricing objective / log loss primary
2. Pregame vs postgame
3. Leakage / feature-before-update
4. Ablation story (kitchen sink loses)
5. Elo + BT/trend + correct blend formula
6. March selection with April excluded
7. April results including AUC miss
8. Live game trace
9. Limitations / production next steps

## Mindset for senior QAs at Bet365

- Sound like a pricing analyst, not a Kaggle contestant.
- Prefer proper scoring rules over accuracy bragging.
- Admit selection bias and the AUC miss early.
- Point to executable proof instead of adjectives.
- Tie roadmap items to trading/ops needs (injuries, market, risk, monitoring).
