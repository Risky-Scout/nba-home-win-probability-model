# Presentation script: every modeling stage

Use this as your screen-share narrative. Speak the quoted lines. Open the
files listed under **Show**. Keep answers short; point to artifacts when a
senior analyst asks for proof.

Suggested total length: **12-15 minutes**, then Q&A.

---

## Stage 0 - Framing (60 seconds)

**Show:** `SUMMARY.md`

> I treated this as a **pricing** task, not a classification task. The
> deliverable is a home-win probability for April games, trained on
> October-March information. Primary metrics are log loss and Brier because
> they score the probability itself. AUC and accuracy are secondary ranking
> and threshold diagnostics.

**One sentence on the role fit:**

> That matches how a sportsbook prices markets: a well-calibrated probability
> is the product; a hard 0/1 call is not.

**Anchor numbers to memorize:**

| Period | Games | Role |
|---|---:|---|
| Oct-Feb | ~895 | Fit base component coefficients for March |
| March | 239 | Architecture + calibration selection |
| April | 96 | Final retrospective scoring |

---

## Stage 1 - Data audit and manipulation (90 seconds)

**Show:** `artifacts/data_audit.json`, then briefly `nba_wp/data.py`

> The file has 1,230 games, 30 teams, no missing values, and no duplicate
> `game_id`s. Pregame wins and losses reconcile exactly against the
> chronological season-to-date record. One schema note: the brief says
> fourteen columns; the listed fields are sixteen.

**Data manipulation choices to state:**

1. Parse `game_date`, sort chronologically, never randomly shuffle.
2. Define the target as `home_win = 1{home_points > away_points}`.
3. Treat box-score totals as **postgame** fields: points, turnovers, fouls,
   rebounds.
4. Treat pregame record fields as auditable, not as the sole strength signal.
5. Split by calendar month, not by a random train/test cut.

**If asked "what did you clean?":**

> Nothing needed cleaning for missingness or duplicates. The work was
> validating integrity and enforcing temporal availability, not imputation.

---

## Stage 2 - Leakage control / information policy (2 minutes)

**Show:** `nba_wp/features.py` around `build_features`

> This is the most important engineering decision. For each date I:
>
> 1. read both teams' states from history strictly before that date;
> 2. write every matchup feature row for that date;
> 3. only then observe outcomes and update Elo, margins, and tendencies.
>
> Same-day games are batched, so a matinee cannot leak into a night game on
> the same date. Current-game points would reveal the target, so they never
> enter their own row. Turnovers, fouls, and rebounds are also postgame and
> follow the same rule.

**Live proof (optional, ~10s):**

```bash
python -m pytest tests/test_feature_timing.py -q
```

**Two evaluation policies to name:**

| Policy | Meaning |
|---|---|
| `sequential_daily` | After a date finishes, update state for later dates |
| `frozen_snapshot` | Freeze performance state at month start |

> I report both. Mixing them silently would be dishonest about the information
> set.

---

## Stage 3 - Feature engineering (2 minutes)

**Show:** `docs/FEATURE_ENGINEERING.md`, then `artifacts/feature_dictionary.csv`

> I generated a broader feature set than I shipped. Everything is expressed
> from the home perspective: positive means home-favored.

**Families you engineered (even if rejected):**

| Family | Examples | Fate |
|---|---|---|
| Record | `record_logit_diff` with Beta(4,4) smoothing | Strong baseline; not champion alone |
| Margin form | cumulative margin, EWMA margin | Useful; correlated with Elo/BT |
| Elo | `elo_diff` with MOV and HFA | Champion component |
| Bradley-Terry | `bt_logit` from regularized paired comparisons | Dominant ranking signal |
| Trend | short (10-game) minus long (45-day half-life) margin | Small correction in champion |
| Schedule | rest advantage, B2B, density | Built; not promoted |
| Box tendencies | turnover / rebound / foul advantages | Built; not promoted |

**What you refuse to invent:**

> I did **not** invent pace or Four Factors. The file lacks FGA, FTA, OREB,
> and makes. Defining possessions as points over a constant makes offensive
> rating identically constant. That would be a fake feature.

**Champion features only:**

```text
elo_diff
bt_logit
trend_diff
```

---

## Stage 4 - Model architecture (2 minutes)

**Show:** `docs/METHODOLOGY.md` and `nba_wp/model.py` (`fit_base_models`, `blend_probabilities`)

> The champion has two transparent probability components, then a calibrated
> blend.

### Component A - Margin-of-victory Elo

> Teams start at 1500. Home adjustment is 75 Elo points in the selected
> architecture. Updates use \(K=10\) and a log margin multiplier. I calibrate
> the Elo difference with an L2 logistic model.

### Component B - Bradley-Terry + trend

> Bradley-Terry estimates regularized team strengths from prior outcomes:
> \(P(home) = \sigma(\alpha + q_h - q_a)\). Trend measures whether a team's
> recent point margin differs from its longer EWMA form. A second logistic
> maps `(bt_logit, trend_diff)` to a rank-component probability.

### Blend

\[
z = w\,\mathrm{logit}(p_{Elo}) + (1-w)\,\mathrm{logit}(p_{rank})
\]

\[
p = \sigma(z/\tau + b)
\]

Selected: \(w=0.19\), \(\tau=0.59\), \(b=0.33\).

> Blend in log-odds because that is the additive scale of logistic evidence.
> Temperature sharpens or flattens; shift absorbs residual home baseline.

**Why not a black-box tree model as the headline?**

> With 1,230 games and highly collinear team-strength signals, a regularized
> logistic stack is faster, deterministic, coefficient-interpretable, and
> produces probabilities directly. I would promote XGBoost only after a
> forward proper-score win, not for complexity theater.

---

## Stage 5 - Training, validation, and selection (2 minutes)

**Show:** `scripts/select_model.py`, `artifacts/selection_proof.json`,
`configs/architecture_candidates.json`

> Selection is chronological and machine-enforced.

**Protocol:**

1. Fit base coefficients on games through February.
2. Score March one-step-ahead with date-batched state updates.
3. Truncate the selection input at March 31; reject any April row.
4. Search 5 declared architectures × 68,231 calibration settings each.
5. Keep only candidates that beat **all four** March numerical targets.
6. Among eligible candidates, lexicographically minimize log loss, then Brier,
   then maximize AUC, then accuracy.
7. Write `artifacts/selected_spec.json`; the scorer loads that file instead of
   hard-coding magic numbers.

**Selected architecture:** `hfa_75`

```text
Elo K=10, HFA=75, BT C=0.15, trend half-life=45d, short window=10
calibration: w=0.19, temperature=0.59, shift=0.33
```

**Honesty line (say this):**

> March is the selection set, so its champion score is optimistic. I call it a
> selection result, not an untouched test. April is retrospective: the code
> cannot load it during selection, but I do not claim perfect human blindness
> across the whole project history.

---

## Stage 6 - Diagnostics and feature proof (90 seconds)

**Show:** `artifacts/feature_group_ablation.csv`,
`figures/permutation_importance.png`, calibration figures

> Ablation shows the story:

| Stage | March log loss |
|---|---:|
| Constant home prior | 0.681 |
| Record-only logistic | 0.549 |
| Elo alone | 0.511 |
| BT + trend | 0.529 |
| Selected blend | **0.488** |

> Permutation importance: Bradley-Terry dominates ranking, Elo is meaningful,
> trend is a small correction. Rest and box-score tendency features did not
> earn promotion in the rich linear challenger.

**Calibration:**

> Temperature 0.59 sharpened March probabilities. I keep calibration plots for
> March and April as monitoring artifacts, not as a claim that one temperature
> is universal forever.

---

## Stage 7 - Final results (90 seconds)

**Show:** `artifacts/final_metrics.json`, `outputs/april_predictions.csv`

### Operational one-step-ahead

| Period | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| March model | 0.4876 | 0.1568 | 0.831798 | 77.82% |
| March target | 0.5096 | 0.1676 | 0.831798 | 77.82% |
| April model | 0.4634 | 0.1456 | 0.8502 | 83.33% |
| April target | 0.4686 | 0.1506 | **0.8682** | 81.25% |

**How to say it:**

> March clears all four rounded targets; the AUC and accuracy margins are tiny
> and I will not oversell them. April beats log loss, Brier, and accuracy, and
> misses AUC. I did **not** retune after seeing that miss.

**Why AUC can miss while proper scores improve:**

> AUC only cares about pairwise ranking. Calibration can improve log loss and
> Brier without fixing every ranking disagreement. For a pricing desk, the
> proper scores are the more relevant miss/hit story.

**Optional frozen-snapshot line:**

> Under a strict March 31 snapshot, April log loss is 0.4589 and AUC rises to
> 0.8628, still short of 0.8682. Different information policies answer different
> questions; both are in the repo.

**Fair odds:**

> Each April row also exports zero-margin decimal odds. Those are mathematical
> fair prices, not a production quote. No overround, liability, or trader
> override is applied.

---

## Stage 8 - Limitations and production roadmap (60 seconds)

**Show:** `docs/LIMITATIONS_AND_ROADMAP.md`

> This is a one-season, team-level technical-task model. Missing for production:
> player availability, minutes, multi-season hierarchy, valid possessions,
> travel/time zones, and market anchors. Next, I would add hierarchical
> dynamic strength, injury/lineup adjustments, drift monitoring, shadow
> pricing against the market, and versioned quotes with risk controls.

**Close with the interview position:**

> The strongest claim is not that every target was beaten. It is that the full
> evidence chain is reproducible, April is excluded from selection code,
> probability quality is strong on proper scores, and the AUC miss is reported
> without post-hoc retuning.

---

## Stage 9 - Live demo (choose one)

### Fast path (< 1 minute)

```bash
export NBA_DATA_PATH="$PWD/data/nba-win-probability-data.csv"
python validate_submission.py --root . --data "$NBA_DATA_PATH"
```

### Rebuild prices (~ tens of seconds)

```bash
python run_submission.py --root . --data "$NBA_DATA_PATH" --mode score
```

### Full recompute proof

```bash
python validate_submission.py --root . --data "$NBA_DATA_PATH" --recompute
```

Open one interesting April row from `outputs/april_predictions.csv` and narrate:

> Here is the matchup, the three features, the two component probabilities,
> the blended home probability, and the fair decimal odds.

---

## Tab checklist before the call

1. `SUMMARY.md`
2. `nba_wp/features.py`
3. `nba_wp/model.py`
4. `scripts/select_model.py`
5. `artifacts/selection_proof.json`
6. `artifacts/feature_group_ablation.csv`
7. `artifacts/final_metrics.json`
8. `outputs/april_predictions.csv`
9. `figures/march_calibration.png` / `april_calibration.png`
10. `docs/LIMITATIONS_AND_ROADMAP.md`
11. This file: `docs/PRESENTATION_SCRIPT.md`
12. `docs/INTERVIEW_QA_CHEATSHEET.md`
