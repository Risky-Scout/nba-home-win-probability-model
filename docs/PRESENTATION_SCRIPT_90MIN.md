# 90-minute presentation script

**How to use this.** Read the lines under **Say** almost verbatim. Do the
actions under **Do**. When you see **Show the work**, open that file and
leave it on screen while you speak — that is the evidence for how a number
was calculated or selected. Pause at **Invite questions** / **If they
interrupt** blocks — expected in a 90-minute senior-quant interview.

**Full Q&A catalog (Q1–Q110):** keep `docs/SENIOR_QUANT_QA.md` open as a
second speaker pane. Every mathematical / adversarial question maps there
with a **Show:** file. This script tells the build story; that file is your
interrupt answer key.

Timing is a guide, not a clock to fight.

**Parameter rule you will repeat:**

> Every number in this model is one of three things:
> 1. **Fitted** from data (MLE / logistic / sequential Elo update),
> 2. **Selected** on March by a declared search, or
> 3. **Fixed by design** as a standard convention.
> I will show the file that proves which class each number belongs to.

**Fact traps (do not recite wrong prep numbers):**

- Full-sample home rate ≈ **55.4%** (`data_audit.json`)
- Elo HFA 75 ⇒ ≈ **60.6% only for equal-rated teams**
- March home rate ≈ **60.3%**
- Temperature divides **full** blended logit \(z\)

**Setup before you join (2 minutes early):**

Use `docs/CURSOR_PRESENTATION_SETUP.md`. In this cloud workspace:

```bash
cd /workspace
export NBA_DATA_PATH="$PWD/data/nba-win-probability-data.csv"
python3 validate_submission.py --root . --data "$NBA_DATA_PATH"
```

On your Mac for the real interview:

```bash
cd "/Users/josephshackelford/nba-home-win-probability-model"
cursor .
source .venv/bin/activate
export NBA_DATA_PATH="$PWD/data/nba-win-probability-data.csv"
python validate_submission.py --root . --data "$NBA_DATA_PATH"
```

Leave Checkpoint **A** (`PASS`) visible.

**Tabs to keep open:**

1. This file
2. `docs/SENIOR_QUANT_QA.md` (interrupt answer key — Q1–Q110)
3. `SUMMARY.md`
4. `nba_wp/data.py`
5. `nba_wp/features.py`
6. `nba_wp/model.py`
7. `nba_wp/selection.py`
8. `scripts/select_model.py`
9. `configs/architecture_candidates.json`
10. `configs/selection_policy.json`
11. `configs/benchmarks.json`
12. `artifacts/data_audit.json`
13. `artifacts/selection_proof.json`
14. `artifacts/selected_spec.json`
15. `artifacts/march_architecture_results.csv`
16. `artifacts/march_tuning_top_candidates.csv`
17. `artifacts/coefficient_table.csv`
18. `artifacts/feature_group_ablation.csv`
19. `artifacts/permutation_importance.csv`
20. `artifacts/final_metrics.json`
21. `outputs/april_predictions.csv`
22. `docs/LIMITATIONS_AND_ROADMAP.md`
23. `figures/march_calibration.png` / `april_calibration.png`

**Correct blend formula (never get this wrong):**

```text
z = 0.19 * logit(p_elo) + 0.81 * logit(p_rank)
p = sigmoid(z / 0.59 + 0.33)
```

Temperature divides the **entire** blended logit.

**Quick parameter cheat (champion values):**

| Number | Class | Value |
|---|---|---|
| Elo start | Fixed | 1500 |
| Elo scale | Fixed | 400 |
| Elo \(K\) | Selected (arch) | 10 |
| Elo HFA | Selected (arch) | 75 |
| MOV mode | Fixed in search | log |
| BT `C` | Selected (arch) | 0.15 |
| Trend half-life | Selected (arch) | 45 days |
| Short window | Selected (arch) | 10 games |
| `elo_model_c` | Selected (arch) | 100 |
| `rank_model_c` | Selected (arch) | 0.1 |
| Blend weight \(w\) | Selected (grid) | 0.19 |
| Temperature \(\tau\) | Selected (grid) | 0.59 |
| Shift \(b\) | Selected (grid) | 0.33 |
| Logistic coeffs | Fitted | see coefficient table |

Full ledger: **Appendix D**. Q&A index: **Appendix E** + `SENIOR_QUANT_QA.md`.

---

# Act I — Framing the problem (0:00–0:08)

## Chapter 1 — Opening and agenda (0:00–0:04)

**Do:** Share screen. Show `SUMMARY.md`.

**Say:**

> Thanks for the time. I’ll walk you through this model the way I built it —
> chronologically — so you can see not just the final specification, but the
> decisions, the rejected paths, and the controls that make the probabilities
> trustworthy.
>
> The assignment asked for a home-team win probability for April games, using
> October through March information. Bet365 prices markets, so I treated this
> as a **pricing** problem, not a classification contest. The product I need
> to defend is a calibrated probability, and the fair decimal odds that fall
> out of it — not a 0/1 pick.
>
> Here’s the agenda:
>
> 1. How I defined success
> 2. Data audit and pregame vs postgame
> 3. Leakage-safe feature engine
> 4. Feature candidates — what I built, kept, rejected
> 5. Elo, Bradley-Terry, trend — formulas and fitted pieces
> 6. Blend and calibration — and how \(w\), \(\tau\), \(b\) were selected
> 7. Architecture search — how \(K\), HFA, BT \(C\), trend windows were chosen
> 8. Diagnostics and April results, including the AUC miss
> 9. Live price of one April game
> 10. Production roadmap
>
> When we hit a number, I will say whether it was **fitted**, **selected**, or
> **fixed by design**, and I will open the file that shows the work.
>
> Please interrupt as we go.

**Invite questions:** “Any preference on depth versus breadth before I start?”

**If they interrupt — Phase 1 (Q1–Q4):** open `SENIOR_QUANT_QA.md`.
Ready answers: 90-second summary; output is a price; why not binary winners;
why log loss. **Show:** `SUMMARY.md`, `april_predictions.csv` fair-odds
columns, `benchmarks.json`.

---

## Chapter 2 — Success metrics and the sportsbook lens (0:04–0:08)

**Do:** Stay on `SUMMARY.md`, or open `docs/METHODOLOGY.md` metrics section.
Also open `configs/benchmarks.json`.

**Say:**

> Before touching code, I locked the objective. The target for game \(g\) is
> whether the home team wins:
>
> \(Y_g = 1\) if home points exceed away points, else \(0\).
>
> The submitted object is \(P(Y_g = 1)\).
>
> Metric order for a pricing desk:
>
> - **Log loss** primary — strictly proper; punishes confident wrong prices
> - **Brier** secondary — proper but quadratic
> - **AUC** ranking diagnostic
> - **Accuracy** at 0.5 — coarse; not optimized
>
> **Show the work — benchmarks are declared, not invented midstream.**
> In `configs/benchmarks.json` the March and April numerical gates are stored
> as version-controlled constants. Selection must beat all four March targets
> before a candidate is eligible. Those thresholds are the assignment’s
> numerical bar; my search policy is written against them.

**Optional aside:**

> Outputs also include zero-margin fair decimal odds \(1/p\) and \(1/(1-p)\).
> Mathematical fair prices — not customer quotes. No overround.

---

# Act II — Data (0:08–0:18)

## Chapter 3 — First contact with the file (0:08–0:13)

**Do:** Open `nba_wp/data.py` at `EXPECTED_COLUMNS` and `load_games`.
Then `artifacts/data_audit.json`. Then `artifacts/date_split_summary.csv`.

**Say:**

> Step one: treat the CSV as a contract.
>
> **Calculated from the file (audit, not model parameters):**
>
> | Quantity | Value | Where shown |
> |---|---|---|
> | Rows | 1,230 | `data_audit.json` → `row_count` |
> | Teams | 30 | `team_count` |
> | Games/team | 82 | `team_game_counts` |
> | Date range | 2025-10-21 → 2026-04-12 | `date_min` / `date_max` |
> | Home-win rate | ~55.4% | `home_win_rate` |
> | Ties / dupes / nulls | 0 | corresponding `*_count` fields |
>
> Schema note: brief says 14 columns; listed schema is 16. I validate 16.
>
> In `load_games`:
>
> - `game_id` as string, zero-fill to 10 — **design choice** so leading zeros
>   survive
> - strict dates/numerics; fail on nulls
> - reject home=away and ties
> - label `home_win = 1{home_points > away_points}` — **calculated**
> - stable sort by date, game_id — never shuffle
>
> Pregame W–L columns are **reconciled** by replaying the season. That check
> lives in `_record_reconciliation` in `data.py`; the audit JSON records that
> they match. So those fields are genuinely pregame.
>
> Month sizes from `date_split_summary.csv` drive the split design:
> Oct–Feb fit coefficients for March selection; March = 239 selection games;
> April = 96 final scoring games.
>
> Full-sample home rate ~55.4% is **not** the same as Elo HFA=75 implying
> ~60.6% for equal-rated teams. I will not conflate them.

---

## Chapter 4 — Postgame versus pregame (0:13–0:18)

**Do:** In `data.py`, highlight `POSTGAME_COLUMNS`.

**Say:**

> The first modeling decision is an information classification, not a
> hyperparameter.
>
> - **Pregame:** teams, date, season-to-date W–L
> - **Postgame:** points, turnovers, fouls, rebounds
>
> Postgame may update **future** state only. Never the current feature row.
>
> I refuse invented pace / Four Factors — missing FGA, FTA, OREB, makes.
> Points/constant possessions make ORtg identically constant. No fake features.
>
> **Show the work:** `POSTGAME_COLUMNS` in `data.py` is the explicit list the
> feature engine is forbidden from reading for the current row.

**Invite questions.**

**If they interrupt — data math (Q27–Q29, Q78, Q89):** records are pregame
(reconcile mismatch 0); Beta(4,4) at 0–0; no same-team doubleheaders;
audit raises on bad schema/ties/self-play. **Show:** `data_audit.json`,
`data.py::load_games`. Remember home rate ≈ **55.4%** full sample.

---

# Act III — Leakage-safe feature engine (0:18–0:32)

## Chapter 5 — Information policy (0:18–0:24)

**Do:** Open `nba_wp/features.py` at the `build_features` docstring and the
`for date, day in frame.groupby(...)` loop. Scroll to schedule update and
`if freeze is not None and date >= freeze: continue`.

**Say:**

> Feature timing is the leakage control. For every date \(d\):
>
> 1. Refresh Bradley-Terry on **strictly earlier** games
> 2. Read both teams’ states
> 3. Write feature rows for **every** game that date
> 4. Only then update performance state from that date’s results
>
> Same-date batching: matinee cannot leak into night game.
> Test: `tests/test_feature_timing.py::test_same_day_games_are_batched`.
>
> Two evaluation policies — not parameters, information sets:
>
> - `sequential_daily` — later dates may use earlier completed results
> - `frozen_snapshot` — performance frozen at month start; schedule/rest still
>   moves
>
> **Show the work:** the `freeze` branch updates `schedule_dates` then
> `continue`s past performance updates. Rest moves; Elo/BT do not.

---

## Chapter 6 — Team state: every feature parameter calculated (0:24–0:32)

**Do:** Open `_team_state`, `_ewma`, `_elo_multiplier`, `_fit_bradley_terry`.
Keep `features.py` on screen.

**Say:**

> Inside the date loop I build team state. I will name every constant and say
> how it got there.

### 6a. Beta(4,4) record smoother — fixed by design

> Formula: \(\tilde p = (W+4)/(N+8)\), then `record_logit = logit(\tilde p)`.
> Matchup: `record_logit_diff = home − away`.
>
> **Why +4/+8?** Eight games of neutral Beta(4,4) prior so 0–0 is 50%, not
> undefined, and 1–0 is 5/9 not 100%.
>
> **Class:** fixed by design (candidate feature).
> **Show the work:** `_team_state` lines with `(wins + 4.0) / (games + 8.0)`.
> **Fate:** strong in ablation B1; **not** in champion three features.

### 6b. Trend half-life 45 and short window 10 — selected with architecture

> Long form: EWMA of past margins with
> \(w_k = 0.5^{\mathrm{age\_days}/45}\).
> Short form: mean of last 10 margins.
> `trend = short − long`; matchup `trend_diff = home − away`.
>
> **Why these numbers?** They are part of the architecture vector. Across the
> five declared architectures I tried half-lives 30/45/60 and windows 8/10/12.
> The winning architecture `hfa_75` uses 45 and 10.
>
> **Class:** selected (architecture search).
> **Show the work now:** briefly flash `configs/architecture_candidates.json`
> `trend_half_life_days` / `trend_short_games`. Full comparison comes in Act V
> with `march_architecture_results.csv`.
> **Calculation code:** `_ewma` and `_team_state` in `features.py`.

### 6c. Rest cap 7 — fixed by design (candidate only)

> `rest_days = min(days since last scheduled game, 7)`.
> Also B2B and games-in-4 / games-in-6 differentials.
>
> **Class:** fixed design candidates.
> **Show the work:** `_team_state` rest logic.
> **Fate:** rejected — ablation B6 did not beat the blend.

### 6d. Elo: start 1500, scale 400 — fixed; K and HFA — selected

> Before the game:
>
> \(p^{Elo}_{raw} = 1/(1+10^{-(R_h-R_a+H)/400})\)
>
> Feature: `elo_diff = (R_h − R_a + H) / 400`.
>
> After the date:
>
> \(R \leftarrow R + K \cdot m \cdot (S - E)\)
>
> with
>
> \(m = \log(|margin|+1)\cdot(2.2 / \max(0.25,\ \Delta R\cdot 0.001 + 2.2))\).
>
> | Symbol | Value | Class | Why / work |
> |---|---|---|---|
> | Start \(R\) | 1500 | Fixed | Common Elo origin; only diffs matter. Code: `ratings = {team: 1500.0}` in `build_features` |
> | Scale 400 | 400 | Fixed | Classic Elo: 400 pts ↔ 10:1 odds. In the \(10^{-x/400}\) formula |
> | \(H\) HFA | **75** | **Selected** | Architectures tried 55, 65, 75; `hfa_75` won March eligibility race |
> | \(K\) | **10** | **Selected** | Architectures tried 7.5, 10, 15; winner uses 10 |
> | MOV 2.2, 0.001 | those constants | Fixed form | FiveThirtyEight-style MOV; mode `log` declared for all archs |
> | `max(0.25,…)` | 0.25 floor | Fixed guard | Prevents pathological denominator |
>
> **Show the work:**
> - Formula/update: `features.py` Elo block + `_elo_multiplier`
> - Selected \(K\), \(H\): `configs/architecture_candidates.json` → later
>   `artifacts/march_architecture_results.csv`

### 6e. Bradley-Terry `C = 0.15` — selected; strengths — fitted daily

> Design matrix: home +1, away −1, label home win.
> Fit `LogisticRegression(C=bt_c, …)`.
> Intercept = home advantage \(\alpha\); coeffs = team strengths \(\theta_i\).
> Feature `bt_logit = decision_function(matchup)`.
>
> | Piece | Class | Work |
> |---|---|---|
> | \(\theta_i\), \(\alpha\) | **Fitted** each date on prior games | `_fit_bradley_terry` |
> | `bt_c = 0.15` | **Selected** with architecture | `architecture_candidates.json`; rivals used 0.1, 0.25, 0.3 |
>
> **Why refit daily?** Batch MLE uses all prior games; cheap at N≈1200.
> Cache invalidates when `len(prior_games)` changes — show that in
> `build_features`.

**Invite questions.** Optional:

```bash
python3 -m pytest tests/test_feature_timing.py -q
```

**If they interrupt — Phase 3 leakage (Q21–Q26, Q70–Q71, Q74):** this is the
cross-examination. Walk April 5 information set; postgame no self-leak;
same-day batch; frozen rest still moves; rest is schedule-public.
**Show:** `build_features` date loop + run the three timing tests.
Answers verbatim in `SENIOR_QUANT_QA.md` Phase 3.

---

# Act IV — Architecture, fitted coefficients, blend (0:32–0:55)

## Chapter 7 — Ablation: why these features shipped (0:32–0:38)

**Do:** Open `artifacts/feature_group_ablation.csv`.

**Say:**

> With 1,230 games I will not dump every differential into one model.
> Ablation is the work behind **which features are parameters of the champion**.
>
> | Stage | Model | March LL |
> |---|---|---:|
> | B0 | Constant home prior | 0.681 |
> | B1 | Record logit only | 0.549 |
> | B2 | Record + cumulative margin | 0.552 |
> | B3 | Elo component | 0.511 |
> | B4 | BT alone | 0.537 |
> | B5 | BT + trend | 0.529 |
> | B6 | Rich linear (rest, TOV, REB, …) | 0.547 |
> | **B7** | **Calibrated Elo + BT/trend blend** | **0.488** |
>
> **Show the work:** this CSV. B6 proves more features ≠ better. Champion
> parameters are only `elo_diff`, `bt_logit`, `trend_diff` plus blend
> calibration — because B7 won on log loss.
>
> Why not XGBoost headline? Small N, collinear strength signals, need
> deterministic probs and trader-inspectable components.

---

## Chapter 8 — Elo component parameters in depth (0:38–0:44)

**Do:** Open `docs/METHODOLOGY.md` Elo section, `features.py::_elo_multiplier`,
`model.py::fit_base_models`, then `artifacts/coefficient_table.csv`.

**Say:**

> Elo update is online gradient ascent on a Bradley-Terry log-likelihood
> (Kiraly & Qian): residual \(S-E\) is the gradient; \(K\) is the learning rate;
> MOV multiplier reshapes the step.
>
> **Selected Elo structural params** (preview; full table in Act V):
> \(K=10\), \(H=75\), MOV=`log`.
>
> **Equal-team implication of H=75 (calculated, not fitted):**
>
> \(P(home)=1/(1+10^{-75/400})\approx 0.606\).
>
> That is structural HFA inside Elo — not the 55.4% sample base rate.
>
> **Fitted Elo logistic (Class: fitted):**
> After features exist, I map `elo_diff` → probability with
> `LogisticRegression(C=elo_model_c)`.
>
> For the champion, `elo_model_c = 100` ≈ unregularized — **selected with
> architecture** because one feature has no multicollinearity to fight.
>
> **Show the work — fitted coefficients** in `coefficient_table.csv`:
>
> | Component | Feature | Std. coef | Raw unit coef |
> |---|---|---:|---:|
> | elo | `elo_diff` | 0.927 | 2.713 |
> | elo | intercept | 0.242 | — |
>
> Those numbers are **calculated by MLE on the training window**, then used to
> produce `elo_component_probability`. I did not type 0.927 by hand.

---

## Chapter 9 — BT + trend parameters in depth (0:44–0:50)

**Do:** Show `_fit_bradley_terry`, `fit_base_models` rank fit,
`coefficient_table.csv`, `permutation_importance.csv`.

**Say:**

> Bradley-Terry:
> \(\mathrm{logit}P = \alpha + \theta_h - \theta_a\).
> Implemented as +1/−1 logistic. Team strengths **fitted**; `bt_c` **selected**.
>
> Rank component logistic on `(bt_logit, trend_diff)` with
> `rank_model_c = 0.1` (**selected** — stronger shrinkage because two features
> sit on top of 30 BT coefficients).
>
> **Show the work — fitted rank coefficients** (`coefficient_table.csv`):
>
> | Feature | Std. coef | Role |
> |---|---:|---|
> | `bt_logit` | 0.804 | dominant |
> | `trend_diff` | 0.144 | small correction |
> | intercept | 0.234 | baseline |
>
> **Show the work — importance** (`permutation_importance.csv`):
> shuffle BT → LL +0.232; Elo +0.020; trend +0.006.
> That is calculated evidence that BT carries ranking, Elo secondary, trend
> small — consistent with blend weight 0.19 later.

---

## Chapter 10 — Blend parameters \(w\), \(\tau\), \(b\): calculated via grid search (0:50–0:55)

**Do:** Open `model.py::blend_probabilities` and `search_calibration`.
Then open `configs/selection_policy.json`.

**Say:**

> Two component probabilities enter the blend:
>
> \(z = w\,\mathrm{logit}(p_{Elo}) + (1-w)\,\mathrm{logit}(p_{rank})\)
>
> \(p = \sigma(z/\tau + b)\)
>
> **These three scalars are not fitted by gradient descent inside sklearn.**
> They are **selected** by exhaustive March grid search.
>
> **Show the work — declared grid** (`selection_policy.json`):
>
> | Param | Range | Step | Count |
> |---|---|---|---:|
> | `elo_weight` \(w\) | 0.00–0.35 | 0.005 | 71 |
> | `temperature` \(\tau\) | 0.55–0.85 | 0.01 | 31 |
> | `shift` \(b\) | 0.10–0.40 | 0.01 | 31 |
>
> Per architecture: \(71\times31\times31 = 68{,}231\) calibrations.
>
> **Show the work — search code** (`model.py::search_calibration`):
> for each weight, vectorize over temperature and shift; compute log loss,
> Brier, accuracy; AUC once per weight (affine logit calibration does not
> change ranking); keep points that beat all four March benchmarks; pick
> lexicographic min LL, then Brier, then max AUC, then accuracy.
>
> **Champion values:** \(w=0.19\), \(\tau=0.59\), \(b=0.33\).
>
> Interpretation:
>
> - \(w=0.19\) — Elo secondary; BT/trend dominates (matches importance)
> - \(\tau=0.59<1\) — sharpen underconfident components on March
> - \(b=0.33\) — residual home baseline on log-odds scale
>
> I treat \(\tau\) cautiously: March is 239 games and the selection set.
>
> **Proof the triple was generated, not typed into scoring code:** coming next
> in Act V with `march_tuning_top_candidates.csv` and `selected_spec.json`.

**Invite questions.**

**If they interrupt — Phase 2 + Phase 7 math (Q5–Q20, Q61–Q69):**
derive Elo as SGD on BT; BT +1/−1 design; MOV multiplier; why w=0.19;
why log-odds blend; why τ=0.59 (divides **full** z); why shift=+0.33
*after* `/τ`; why C=100 vs 0.1; why log loss strictly proper; why τ
preserves AUC; Beta(4,4); permutation vs coefficients.
**Show:** `_elo_multiplier`, `_fit_bradley_terry`, `blend_probabilities`,
`coefficient_table.csv`, `permutation_importance.csv`.
Full derivations: `SENIOR_QUANT_QA.md` Phases 2 and 7.

---

# Act V — Selection work: how structural params were chosen (0:55–1:10)

## Chapter 11 — Declared architectures (0:55–1:00)

**Do:** Open `configs/architecture_candidates.json` side by side with
`artifacts/march_architecture_results.csv`.

**Say:**

> If I nudge knobs until April looks good, that is leakage. So structural
> hyperparameters were declared as five named architectures before the search
> ran.
>
> **Show the work — search space** (`architecture_candidates.json`):

| Name | \(K\) | HFA | BT \(C\) | Half-life | Short | `elo_model_c` | `rank_model_c` |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 10 | 65 | 0.15 | 45 | 10 | 100 | 0.1 |
| conservative | 7.5 | 55 | 0.1 | 60 | 12 | 10 | 0.1 |
| responsive | 15 | 65 | 0.25 | 30 | 8 | 10 | 0.1 |
| **hfa_75** | **10** | **75** | **0.15** | **45** | **10** | **100** | **0.1** |
| lower_bt_shrinkage | 10 | 65 | 0.3 | 45 | 10 | 100 | 0.3 |

> **Show the work — who survived** (`march_architecture_results.csv`):

| Architecture | Eligible count | Best eligible LL | Best \((w,\tau,b)\) |
|---|---:|---:|---|
| **hfa_75** | 30,686 | **0.487569** | 0.19, 0.59, 0.33 |
| balanced | 30,663 | 0.487588 | 0.19, 0.59, 0.33 |
| lower_bt_shrinkage | 30,586 | 0.487667 | 0.215, 0.58, 0.31 |
| conservative | **0** | — | failed all-four-target gate |
| responsive | **0** | — | failed all-four-target gate |

> **Why \(K=10\)?** Among declared options {7.5, 10, 15}, only architectures
> with \(K=10\) (plus the BT/HFA variants above) produced eligible calibrations.
> Responsive (\(K=15\)) and conservative (\(K=7.5\)) produced **zero** eligible
> points — their best March metrics missed the AUC/target gate.
>
> **Why HFA=75?** Among eligible architectures, `hfa_75` had the lowest March
> log loss under the lexicographic rule — beating `balanced` (HFA 65) by a
> small but decisive amount under that rule.
>
> **Why BT \(C=0.15\), half-life 45, short 10?** They are the structural bundle
> attached to `hfa_75`. I do not claim each was optimized on a 1-D scan in
> isolation; I claim the **joint architecture** won the declared comparison.
> `lower_bt_shrinkage` tried freer BT (\(C=0.3\)) and lost on March LL.

---

## Chapter 12 — Calibration grid work and April exclusion (1:00–1:10)

**Do:** Open in order:
1. `scripts/select_model.py` (April truncate)
2. `nba_wp/selection.py::run_selection` (reject April; fit Oct–Feb; search March)
3. `artifacts/march_tuning_top_candidates.csv` (top rows)
4. `artifacts/selected_spec.json`
5. `artifacts/selection_proof.json`
6. `configs/benchmarks.json`

**Say:**

> Chronological selection protocol — this is how the champion scalars were
> **calculated as the argmin of a declared objective**:
>
> 1. Truncate raw input to `game_date < 2026-04-01` **before** features
>    — **Show:** `scripts/select_model.py`
> 2. `run_selection` raises if any April row present
>    — **Show:** `selection.py` guard
> 3. For each architecture, `build_features` on truncated history
> 4. Fit base logistic coefficients on October–February
> 5. Score March one-step-ahead component probabilities
> 6. Run `search_calibration` over 68,231 points
> 7. Eligibility: beat all four March targets in `benchmarks.json`
>    (LL < 0.509645, Brier < 0.167618, AUC > 0.831798, Acc > 0.7782)
> 8. Among eligible, minimize LL, then Brier, then maximize AUC, accuracy
> 9. Write `selected_spec.json`; scorer loads it — no magic constants in score code
>
> **Show the work — leaderboard** (`march_tuning_top_candidates.csv`):
>
> ```text
> #1 hfa_75  w=0.19   τ=0.59  b=0.33  LL=0.487569  ← champion
> #2 hfa_75  w=0.185  τ=0.59  b=0.33  LL=0.487572
> #3 hfa_75  w=0.18   τ=0.59  b=0.33  LL=0.487577
> ```
>
> Neighboring rows are extremely close. That is why I describe 0.59 as a
> selected operating point on a noisy selection set — not a law of nature.
>
> **Show the work — frozen spec** (`selected_spec.json`): architecture block
> plus calibration block plus March metrics plus notes that base coefficients
> use games through February for March scoring.
>
> **Show the work — integrity** (`selection_proof.json`):
> `selection_data_max_date = 2026-03-31`,
> `april_rows_loaded_during_selection = 0`.
>
> Honesty: March metrics are optimistic (selection set). April is retrospective
> for human history even though code excludes it. After freeze, refit
> coefficients through March and score April.

**Invite questions.**

**If they interrupt — Phase 4 integrity (Q30–Q41, Q72–Q73):** April exclusion
code + proof; March optimistic; don’t retune AUC; candidate counts;
lexicographic rule; sequential vs frozen and why frozen April LL can be
better. **Show:** `select_model.py`, `selection_proof.json`,
`march_tuning_top_candidates.csv`, run validator.
Answers: `SENIOR_QUANT_QA.md` Phase 4.

---

# Act VI — Results and live price (1:10–1:25)

## Chapter 13 — Scoreboard (1:10–1:15)

**Do:** `artifacts/final_metrics.json`, calibration figures.

**Say:**

> Operational results:
>
> **March (selection):** LL 0.4876, Brier 0.1568, AUC 0.831798246, Acc 77.82%
> — clears rounded targets; AUC/Acc margins tiny.
>
> **April (retrospective):** LL 0.4634 ✓, Brier 0.1456 ✓, AUC 0.8502 ✗
> (target 0.8682), Acc 83.33% ✓.
>
> I did not retune after the AUC miss. Proper scores can improve without
> fixing ranking. Frozen April: LL 0.4589, AUC 0.8628 — still short of AUC
> target.
>
> Calibration plots monitor sharpness from \(\tau=0.59\); I would not freeze
> that scalar forever in production without drift checks.

---

## Chapter 14 — Trace one April game with parameter path (1:15–1:22)

**Do:** `outputs/april_predictions.csv` → 2026-04-05 UTA @ OKC.
Optionally `outputs/engineered_features.csv` same `game_id`.

**Say:**

> Utah at OKC, April 5. Operational policy: earlier April results may inform
> state; same-day/future may not.
>
> **Feature values on this row (calculated from prior games + selected arch):**
>
> | Feature | ≈ value | Produced by |
> |---|---:|---|
> | `elo_diff` | 1.43 | Elo state with \(K=10\), \(H=75\) |
> | `bt_logit` | 1.89 | Daily BT fit with \(C=0.15\) |
> | `trend_diff` | 9.22 | short10 − EWMA45 |
>
> **Component probs (calculated from fitted logistics):** Elo high; rank high.
>
> **Final price (selected calibration):**
> \(w=0.19\), \(\tau=0.59\), \(b=0.33\) → \(p\approx 99.6\%\).
>
> Outcome: OKC won. Sharp price, all three signals aligned.
>
> Contrast miss: Apr 1 SAC @ TOR, model ~96.8%, SAC won, per-game LL ~3.44 —
> why log loss matches liability.
>
> Live proof:

```bash
python3 validate_submission.py --root . --data "$NBA_DATA_PATH"
# optional:
python3 validate_submission.py --root . --data "$NBA_DATA_PATH" --recompute
```

---

## Chapter 15 — Limitations (1:22–1:25)

**Do:** `docs/LIMITATIONS_AND_ROADMAP.md`.

**Say:**

> One-season team model. No injuries, minutes, market, travel, possessions.
> March reused for selection. Fair odds have no overround.
>
> Production order: player availability → market residuals → multi-season
> hierarchical strength → real possessions → monitoring/governance.
>
> In-play is a different model.

**If they interrupt — Phase 5–6 domain/production (Q42–Q60, Q91–Q96):**
HCA numbers (55.4% vs 60.6% trap); injuries #1; no pace; rest/B2B candidates
rejected; overround; market blend; CLV; monitoring; in-play is separate;
versioned quotes. **Show:** `LIMITATIONS_AND_ROADMAP.md`, fair-odds columns.
Answers: `SENIOR_QUANT_QA.md` Phases 5–6, 10.

---

# Act VII — Close (1:25–1:30)

## Chapter 16 — Closing position

**Do:** `SUMMARY.md`.

**Say:**

> Evaluate me on this: leakage-audited, reproducible pricing model; parameters
> either fitted, selected on a machine-enforced pre-April path, or fixed by
> declared design; strong proper scores; April AUC miss reported without
> retuning; every number traceable to a file.

## Chapter 17 — Open floor (use SENIOR_QUANT_QA.md)

**Say:**

> Happy to go deeper on any stage — leakage tests, Elo-as-gradient derivation,
> the 68k calibration grid, a live feature add, overround arithmetic, or
> adversarial “change K to 20” stress tests.

**If they go quiet, offer:**

1. Walk `build_features` line by line (Q70)
2. Derive Elo from BT log-likelihood (Q61)
3. Trace one April row through blend (Q19, Q76)
4. Overround from a 60% fair price (Q51)
5. Adversarial: set τ=1.0 — what changes? (Q103)

**If they attack — Phase 11 (Q101–Q110):** keep `SENIOR_QUANT_QA.md` Phase 11
open. Short pattern: state effect → point to artifact that already explored
nearby setting → refuse silent April retune.

---

# Appendix A — Timing cheat card

| Clock | Chapter | Must land |
|---|---|---|
| 0:00–0:08 | Framing | Pricing + three parameter classes |
| 0:08–0:18 | Data | Audit numbers from `data_audit.json` |
| 0:18–0:32 | Features | Every state constant + file pointer |
| 0:32–0:55 | Model | Ablation; fitted coeffs; blend grid |
| 0:55–1:10 | Selection | Arch CSV + tuning CSV + proof JSON |
| 1:10–1:25 | Results | AUC miss; OKC trace |
| 1:25–1:30 | Close | Honesty claim |

Never skip: leakage, blend formula, selection CSVs, AUC miss.

---

# Appendix B — If running long / short

**Cut first:** coefficient decimals, February aside, second miss example,
calibration narration.

**Expand first:** line-by-line `build_features`; hand-compute OKC blend;
scroll more rows of `march_tuning_top_candidates.csv`; overround example.

---

# Appendix C — Story spine

1. Price → proper scores  
2. Audit → pregame/postgame  
3. Feature-before-update  
4. Ablation kills kitchen sink  
5. Fitted Elo/BT coeffs; selected arch + calibration  
6. April: strong LL/Brier; miss AUC; no retune  
7. Every number has a file  

---

# Appendix D — Complete parameter ledger (open these files)

Use this when they ask “how was X chosen?” Find the row, open the file, speak
the class.

## D1. Fitted from data (MLE / sequential updates)

| Parameter | How calculated | Why | Open this |
|---|---|---|---|
| Elo ratings \(R_i\) | Start 1500; daily \(R+=K m (S-E)\) | Online strength | `nba_wp/features.py` Elo update |
| `elo_diff` | \((R_h-R_a+H)/400\) | Scaled matchup edge | same, feature row write |
| BT \(\theta_i\), \(\alpha\) | Daily L2 logistic +1/−1 | Batch paired strength | `features.py::_fit_bradley_terry` |
| `bt_logit` | `decision_function` | Matchup log-odds | same |
| Margins / EWMA / short | From prior game margins | Form inputs | `features.py::_team_state`, `_ewma` |
| `trend_diff` | \((S-L)_h-(S-L)_a\) | Relative form | `_team_state` |
| Elo logistic coef / intercept | `LogisticRegression` on `elo_diff` | Map rating edge → \(p\) | `model.py::fit_base_models`; `artifacts/coefficient_table.csv` |
| Rank logistic coefs | LR on `bt_logit`, `trend_diff` | Map BT+trend → \(p\) | same |
| Component probabilities | `predict_proba` | Inputs to blend | `model.py::component_probabilities` |
| Final \(p\) given \(w,\tau,b\) | blend formula | Price | `model.py::blend_probabilities` |
| Fair odds | \(1/p\), \(1/(1-p)\) | Zero-margin quote | `outputs/april_predictions.csv` columns |

## D2. Selected on March (declared search)

| Parameter | Search space | Winner | Why that winner | Open this |
|---|---|---|---|---|
| Architecture name | 5 named bundles | `hfa_75` | Best eligible March LL | `configs/architecture_candidates.json` + `artifacts/march_architecture_results.csv` |
| Elo \(K\) | 7.5, 10, 15 | 10 | Only \(K=10\) arches were eligible | same |
| Elo HFA | 55, 65, 75 | 75 | `hfa_75` beat `balanced` on LL | same |
| BT `C` | 0.1, 0.15, 0.25, 0.3 | 0.15 | Bundle on winning arch; freer BT lost | same |
| Trend half-life | 30, 45, 60 | 45 | Bundle on winning arch | same |
| Short window | 8, 10, 12 | 10 | Bundle on winning arch | same |
| `elo_model_c` | 10 or 100 | 100 | One-feature Elo; low regularization | same |
| `rank_model_c` | 0.1 or 0.3 | 0.1 | Shrink 2-feature rank model | same |
| `elo_weight` \(w\) | 0–0.35 step 0.005 | 0.19 | Min March LL among eligible | `configs/selection_policy.json` + `artifacts/march_tuning_top_candidates.csv` |
| `temperature` \(\tau\) | 0.55–0.85 step 0.01 | 0.59 | same grid argmin | same + `model.py::search_calibration` |
| `shift` \(b\) | 0.10–0.40 step 0.01 | 0.33 | same | same |
| Frozen champion spec | output of search | JSON below | Scorer loads file | `artifacts/selected_spec.json` |
| April exclusion | truncate + raise | 0 rows | Integrity | `scripts/select_model.py` + `artifacts/selection_proof.json` |

## D3. Fixed by design (convention / guard — not March-optimized)

| Parameter | Value | Why | Open this |
|---|---|---|---|
| Elo start | 1500 | Common origin | `features.py` `ratings = {team: 1500.0}` |
| Elo scale | 400 | Classic Elo odds scale | Elo probability formula |
| MOV constants | 2.2, 0.001 | FTE-style MOV form | `features.py::_elo_multiplier` |
| MOV denom floor | 0.25 | Numerical guard | same |
| MOV mode | `log` | Declared for all archs | `architecture_candidates.json` `elo_mov` |
| Beta prior | (4,4) | Early-season record shrink | `_team_state` |
| Rest cap | 7 days | Bounded schedule feature | `_team_state` |
| March targets | see JSON | Assignment gates | `configs/benchmarks.json` |
| Selection rule | lexico LL→Brier→AUC→Acc | Declared policy | `configs/selection_policy.json` |
| Permutation seed / reps | 365 / 100 | Importance stability | `nba_wp/reporting.py` |
| Bootstrap seed / reps | 2026 / 2000 | Paired LL diffs | same |

## D4. 90-second “parameter tour” if they ask only for hypertuning

**Do this click path while speaking:**

1. `configs/architecture_candidates.json` — structural knobs declared  
2. `configs/selection_policy.json` — calibration grid declared  
3. `nba_wp/model.py::search_calibration` — how each point is scored  
4. `artifacts/march_architecture_results.csv` — `hfa_75` wins; 2 arches ineligible  
5. `artifacts/march_tuning_top_candidates.csv` — row 1 = 0.19 / 0.59 / 0.33  
6. `artifacts/coefficient_table.csv` — fitted logistic coefficients  
7. `artifacts/selected_spec.json` + `selection_proof.json` — frozen + April=0  

**Say:**

> Structural parameters were chosen by comparing five declared architectures on
> March. Calibration parameters were chosen by a 68,231-point grid per
> architecture under an eligibility filter. Logistic coefficients were fitted
> by maximum likelihood. Elo’s 1500/400 and MOV 2.2 are fixed conventions.
> Nothing important was “eyeballed into April.”

---

# Appendix E — Senior Quant Q&A index (Q1–Q110)

**Master answer file:** `docs/SENIOR_QUANT_QA.md`

During the interview, when they ask a math / leakage / production /
adversarial question, switch to that tab, find the Q number, read the
answer, open the **Show:** file.

| Phase | Qs | When in the talk | Theme |
|---|---|---|---|
| 1 Opening | Q1–Q4 | Act I | 90s summary, price vs pick, log loss |
| 2 Architecture | Q5–Q20 | Act IV | Elo/BT/trend/blend math |
| 3 Leakage | Q21–Q29 | Act III | Make-or-break cross-exam |
| 4 Selection | Q30–Q41 | Act V | April exclusion, March bias, grid |
| 5 Domain | Q42–Q50 | Act VI / anywhere | HCA, injuries, pace, rest |
| 6 Production | Q51–Q60 | Act VI–VII | Overround, CLV, monitoring, in-play |
| 7 Theory | Q61–Q69 | Act IV interrupts | Derivations, proper scores, AUC |
| 8 Code | Q70–Q80 | Live demo | Walk code, tests, validator |
| 9 Edge cases | Q81–Q90 | Anytime | Cold start, OT, multicollinearity |
| 10 Roadmap | Q91–Q100 | Act VI–VII | Improvements, why Bet365 |
| 11 Adversarial | Q101–Q110 | End stress test | Change K/τ; “are you lying?” |

**Must-land adversarial answers (memorize):**

| Attack | One-line |
|---|---|
| K→20 | Responsive K=15 already ineligible; noisier Elo |
| Remove trend | Still runs; small LL hit (~0.006 importance) |
| τ→1.0 | Less sharp; LL likely worse; **AUC unchanged** |
| Peek April? | Truncate + raise + proof April=0 |
| Retune AUC? | Test-set leakage; reported miss |
| Home rate 60.6%? | That’s **equal-team Elo HFA**; sample ≈ **55.4%** |

**Live commands (cloud: `python3`; Mac: `python` after venv):**

```bash
python3 validate_submission.py --root . --data "$NBA_DATA_PATH"
python3 -m pytest tests/test_feature_timing.py -q
```
