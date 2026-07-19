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
- [ ] Rehearse `docs/PRESENTATION_SCRIPT.md` once out loud (12-15 min)
- [ ] Drill Q&A: AUC miss, March selection bias, correct blend formula, April 5 info set
- [ ] Open the tab checklist from the presentation script
- [ ] Know your live demo command by heart
- [ ] Sleep

## Day of interview

```bash
cd /path/to/nba-home-win-probability-model
source .venv/bin/activate
export NBA_DATA_PATH="$PWD/data/nba-win-probability-data.csv"
python validate_submission.py --root . --data "$NBA_DATA_PATH"
```

Leave the terminal green/`PASS` visible when you join the call.

## Presentation order (do not skip)

1. Objective / metrics
2. Data audit
3. Leakage / feature timing
4. Feature engineering
5. Model (Elo, BT/trend, blend)
6. Selection protocol + proof
7. Ablation / importance
8. Results (including AUC miss)
9. Limitations / production next steps
10. Live validate or score

## Mindset for senior QAs at Bet365

- Sound like a pricing analyst, not a Kaggle contestant.
- Prefer proper scoring rules over accuracy bragging.
- Admit selection bias and the AUC miss early.
- Point to executable proof instead of adjectives.
- Tie roadmap items to trading/ops needs (injuries, market, risk, monitoring).
