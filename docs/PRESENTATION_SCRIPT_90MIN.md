# 90-minute presentation script

**How to use this.** Read the lines under **Say** almost verbatim. Do the
actions under **Do**. Pause at **Invite questions** if they jump in — that is
expected in a 90-minute technical interview. Timing is a guide, not a clock to
fight.

**Setup before you join (2 minutes early):**

Use `docs/CURSOR_PRESENTATION_SETUP.md`. Short version:

```bash
cd "/Users/josephshackelford/nba-home-win-probability-model"
cursor .
# Cursor integrated terminal:
source .venv/bin/activate
export NBA_DATA_PATH="$PWD/data/nba-win-probability-data.csv"
python validate_submission.py --root . --data "$NBA_DATA_PATH"
```

Leave Checkpoint **A** (`PASS`) visible. Open these tabs:

1. This file  
2. `SUMMARY.md`  
3. `nba_wp/data.py`  
4. `nba_wp/features.py`  
5. `nba_wp/model.py`  
6. `nba_wp/selection.py`  
7. `scripts/select_model.py`  
8. `artifacts/data_audit.json`  
9. `artifacts/selection_proof.json`  
10. `artifacts/feature_group_ablation.csv`  
11. `artifacts/final_metrics.json`  
12. `outputs/april_predictions.csv`  
13. `docs/LIMITATIONS_AND_ROADMAP.md`  
14. `figures/march_calibration.png` and `figures/april_calibration.png`

**Correct blend formula (never get this wrong):**

```text
z = 0.19 * logit(p_elo) + 0.81 * logit(p_rank)
p = sigmoid(z / 0.59 + 0.33)
```

Temperature divides the **entire** blended logit.

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
> Here’s the agenda for the next hour or so:
>
> 1. How I defined success  
> 2. What the data actually contain, and what they do **not**  
> 3. How I enforced a pregame information set — leakage control  
> 4. Feature engineering: what I built, kept, and rejected  
> 5. Why Elo and Bradley-Terry with a trend correction  
> 6. How I blend and calibrate those components  
> 7. How I selected the architecture without using April  
> 8. Diagnostics: ablation, importance, calibration  
> 9. April results — including the AUC miss  
> 10. A live walkthrough of one priced game  
> 11. What I would do next in production  
>
> Please interrupt with questions as we go. I’d rather defend a step while
> we’re on it than batch every challenge to the end.

**Invite questions:** “Any preference on depth versus breadth before I start?”

---

## Chapter 2 — Success metrics and the sportsbook lens (0:04–0:08)

**Do:** Stay on `SUMMARY.md`, or open `docs/METHODOLOGY.md` section on metrics.

**Say:**

> Before touching code, I locked the objective. The target for game \(g\) is
> whether the home team wins:
>
> \(Y_g = 1\) if home points exceed away points, else \(0\).
>
> The submitted object is \(P(Y_g = 1)\).
>
> I ordered the metrics the way a pricing desk should:
>
> - **Log loss** is primary. It is a strictly proper scoring rule: you
>   minimize expected loss by reporting your true belief. It also punishes
>   confident wrong prices exponentially — which matches sportsbook liability.
>   A price of 0.99 that is wrong is catastrophic; a price of 0.51 that is
>   wrong is a small mistake. Accuracy treats both as the same “wrong side of
>   0.5.”
> - **Brier** is secondary. It is also proper, but quadratic, so it underweights
>   the catastrophic tail relative to log loss.
> - **AUC** is a ranking diagnostic. It asks whether I order matchups
>   correctly. Useful, but it does not care whether I said 55% or 85%.
> - **Accuracy** at 0.5 is a coarse threshold summary. I report it; I do not
>   optimize for it.
>
> That choice matters later when April improves log loss and Brier but misses
> AUC. Under a pricing objective, that is an incomplete win — not a reason to
> quietly retune on the test month.

**Optional aside if they nod:**

> The outputs also include zero-margin fair decimal odds: one over \(p\) for
> home, one over \(1-p\) for away. Those are mathematical fair prices. They are
> not a customer quote. No overround, no liability shaping, no trader override.

---

# Act II — Data: what I was given and what I refused to invent (0:08–0:18)

## Chapter 3 — First contact with the file (0:08–0:13)

**Do:** Open `nba_wp/data.py` at `EXPECTED_COLUMNS` and `load_games`. Then open
`artifacts/data_audit.json`.

**Say:**

> Step one was to treat the CSV as a contract, not a dataframe to trust.
>
> The file has 1,230 games from the 2025–26 season, 30 teams, each team plays
> 82 games. Dates run from October 21, 2025 through April 12, 2026. There are
> no missing values, no duplicate game IDs, and no ties — which is what we
> expect from NBA finals.
>
> One documentation note: the brief says fourteen columns; the listed schema
> is sixteen — two game fields plus seven for away and seven for home. I
> validate against the sixteen-column schema.
>
> In `load_games` I do a few deliberate things:
>
> - Read `game_id` as a string and zero-fill to ten characters. Leading zeros
>   are semantically part of the NBA game ID. Parsing as an integer would
>   corrupt joins later.
> - Parse dates strictly, coerce numerics strictly, fail loud on nulls.
> - Reject home equals away, and reject ties.
> - Create the label `home_win` from the final score.
> - Sort stably by date then game ID — never shuffle. This is a time-ordered
>   pricing problem.
>
> I also reconcile the supplied pregame win/loss columns. I replay the season
> chronologically and check that `home_wins`, `home_losses`, and the away
> counterparts match the true season-to-date record **before** each game.
> They reconcile exactly. That tells me those fields are genuinely pregame,
> not silently contaminated.

**Do:** Scroll `data_audit.json` to `home_win_rate`, `date_min`, `date_max`,
`tied_game_count`, record reconciliation fields.

**Say:**

> Full-sample home-win rate is about **55.4%**. I’ll come back to that when we
> talk about Elo home-court advantage, because a 75-point Elo HFA implies about
> **60.6%** only when two teams are equal-rated. Those are different quantities.
> I will not conflate the structural HFA with the empirical base rate.

**Do:** Briefly show `artifacts/date_split_summary.csv`.

**Say:**

> Month sizes matter for later design:
>
> - October through February are the bulk of the season — that becomes the
>   coefficient-fitting period for March selection.
> - March has 239 games — large enough to select on, small enough that I must
>   not overclaim precision.
> - April has 96 games — the final scoring month.
>
> February’s home-win rate dips below 50% in this sample. That is a reminder
> that month-level rates move around; I should not hard-code a single home
> prior as the model.

---

## Chapter 4 — Postgame versus pregame: the first modeling decision (0:13–0:18)

**Do:** In `data.py`, highlight `POSTGAME_COLUMNS`.

**Say:**

> The most important data decision happens before any model is fit.
>
> Points, turnovers, fouls, and rebounds are **outcomes of the game being
> priced**. Points determine the label directly. Turnovers, fouls, and rebounds
> do not determine the winner with certainty, but they are still unavailable at
> the moment a sportsbook must post a pregame price.
>
> So I classified fields into two families:
>
> - **Known pregame:** teams, date, season-to-date wins and losses.
> - **Known postgame:** points, turnovers, fouls, rebounds.
>
> Postgame fields are allowed to update **future** team state. They are
> forbidden from entering the feature row of their own game.
>
> I also refused to invent features the schema cannot support. Pace and Dean
> Oliver’s Four Factors need field-goal attempts, free-throw attempts,
> offensive rebounds, and makes. Those are missing. Defining possessions as
> points divided by a constant makes offensive rating identically constant —
> a fake feature. So I do not ship pseudo-pace.
>
> That is the discipline I want you to see throughout: if the information is
> not available at pricing time, it does not enter the price.

**Invite questions.**

---

# Act III — Building a leakage-safe feature engine (0:18–0:32)

## Chapter 5 — Designing the information policy (0:18–0:24)

**Do:** Open `nba_wp/features.py` at the `build_features` docstring and the
date loop (the `for date, day in frame.groupby(...)` block).

**Say:**

> Once I knew what was pregame and postgame, I designed the feature engine
> around an explicit information policy.
>
> For every game date \(d\), the engine does four things in this order:
>
> 1. Refresh the Bradley-Terry fit using **strictly earlier** games only.  
> 2. Read both teams’ current states — Elo, trends, records, rest, and so on.  
> 3. Write a feature row for **every** game on that date.  
> 4. Only then observe that date’s results and update performance state.
>
> If I reversed steps 3 and 4, a night game could see a matinee result from
> the same calendar date. In a sportsbook, that would be posting a price with
> information the market did not have at open. That is a disqualifying error.
>
> Same-date games are therefore batched: they all freeze the same pregame
> snapshot. The unit test `test_same_day_games_are_batched` changes one game’s
> score dramatically and asserts the other same-date game’s features are
> unchanged.
>
> There is a second information policy for evaluation sensitivity:
>
> - **`sequential_daily`** — after a date completes, later dates may use those
>   results. This is what a live pregame system would do across a month.
> - **`frozen_snapshot`** — performance state freezes at month start. Schedule
>   still updates, because rest is observable from the schedule without knowing
>   outcomes.
>
> I export both. Mixing them silently would be dishonest about the information
> set. The assignment’s April ask is closest in spirit to a month-start
> discipline, while sequential is the operational sportsbook analogue. You
> should see both numbers.

**Do:** Scroll to the schedule update and the `if freeze is not None and date >= freeze: continue` block.

**Say:**

> Notice the code updates `schedule_dates` even when frozen, then skips
> performance updates. That is intentional. Rest days move; Elo and
> Bradley-Terry do not.

---

## Chapter 6 — What a team state contains (0:24–0:32)

**Do:** Open `_team_state`, `_elo_multiplier`, and `_fit_bradley_terry`.

**Say:**

> Inside the date loop, each team is summarized by a state object before the
> matchup features are formed. I’ll narrate the story of how that state grew.
>
> **Early season problem.** On opening night everyone is 0–0. A raw win rate
> is undefined or extreme. So for record-based candidates I use a Beta(4,4)
> smoother:
>
> \((W + 4) / (N + 8)\).
>
> After zero games that is 50%. After one win it is 5/9, not 100%. That feature
> becomes `record_logit_diff`. It is a strong baseline later in the ablation,
> but it is **not** in the final three-feature champion — I’ll show why.
>
> **Margin history.** For each prior game I store the team’s point margin. From
> that I build cumulative average margin and an exponentially weighted margin
> with a 45-day half-life: a game 45 days ago gets half the weight of a game
> today. The short window is the mean of the last 10 games. Trend is short
> minus long. If a team is outrunning its longer form, trend is positive.
> Matchup feature: home trend minus away trend.
>
> **Rest and schedule density.** Rest days are capped at seven. I also build
> back-to-back and games-in-4 / games-in-6 density differentials. These are
> real NBA effects, but in this one-season sample they did not earn a place in
> the champion. I still compute them so I can show the rejection in ablation
> rather than claiming I never considered them.
>
> **Box-score tendencies.** From completed games I accumulate turnover,
> rebound, and foul advantages — always from history before the current date.
> Same story: candidates, not champions.
>
> **Elo state.** Every team starts at 1500. Before a game I form
>
> \(p^{Elo} = 1 / (1 + 10^{-(R_h - R_a + H)/400})\),
>
> and the feature `elo_diff = (R_h - R_a + H) / 400`. After the date completes,
> ratings update with \(K\) times a margin-of-victory multiplier times the
> residual \(S - E\). The multiplier is
>
> `log(|margin| + 1) * (2.2 / max(0.25, rating_diff * 0.001 + 2.2))`.
>
> The log stops a 40-point blowout from counting as four times a 10-point win.
> The denominator down-weights beatdowns that were already expected from the
> rating gap. That is the FiveThirtyEight-style intuition: unexpected margins
> move ratings more than expected ones.
>
> **Bradley-Terry state.** Each date, if enough history exists, I fit a
> regularized logistic regression on a +1/−1 design matrix: home team +1, away
> team −1, label = home win. The intercept is home advantage; the coefficients
> are team strengths. The matchup feature `bt_logit` is the decision function
> on that matchup. With only about a thousand games, refitting daily is cheap,
> and the batch estimator gives a globally consistent ranking that Elo’s online
> updates do not automatically guarantee.
>
> At this point in the project I had a rich feature factory and a hard
> guarantee that every row was priced with only earlier information. Next I had
> to decide which signals actually improve probability scores.

**Invite questions.** Optional live proof:

```bash
python -m pytest tests/test_feature_timing.py -q
```

---

# Act IV — From features to a model architecture (0:32–0:52)

## Chapter 7 — Why not one big logistic or a tree model? (0:32–0:38)

**Do:** Open `artifacts/feature_group_ablation.csv`.

**Say:**

> With 1,230 games, dumping every differential into a large model is a good way
> to overfit March and embarrass yourself in April. I wanted structure that
> matches how team strength actually evolves.
>
> Look at the ablation — this is the empirical story of the build:
>
> - **B0** — constant training-period home prior. Log loss about 0.681. This is
>   “always price the historical home rate.” Inadequate.
> - **B1** — record logit difference alone. Drops to about 0.549. Season-to-date
>   records already carry real strength information.
> - **B2** — records plus cumulative margin. Roughly similar; margins are
>   correlated with records and do not buy much alone.
> - **B3** — Elo component alone. About 0.511. Sequential margin-adjusted
>   strength is a clear lift.
> - **B4** — Bradley-Terry alone. About 0.537. Strong ranking signal, a bit
>   behind Elo on March proper scores in isolation.
> - **B5** — Bradley-Terry plus trend. About 0.529. Recent-form correction
>   helps BT.
> - **B6** — rich linear challenger: records, margins, trend, rest, turnovers,
>   rebounds. About 0.547. **More features, worse than Elo alone.** This is the
>   key anti-complexity result. Rest and noisy box tendencies did not earn
>   promotion on March proper scores.
> - **B7** — the selected calibrated Elo + BT/trend blend. About **0.488**.
>   Best probability quality among these stages.
>
> So the architecture decision was not “I like Elo.” It was: structured
> strength components beat a kitchen-sink linear model on the metric I care
> about, and a calibrated blend beats either component alone.
>
> Why not XGBoost as the headline? Small \(N\), highly collinear strength
> features, and a production need for deterministic training, direct
> probabilities, and coefficient-level interpretability. Traders should be able
> to override or inspect a component. I would promote a nonlinear challenger
> only after a forward proper-score win — not because it is fashionable.

---

## Chapter 8 — Component A: margin-of-victory Elo in depth (0:38–0:44)

**Do:** Open `docs/METHODOLOGY.md` Elo section and `model.py::fit_base_models`.

**Say:**

> Let me slow down on Elo, because interviewers often ask for the derivation.
>
> Elo is not a mysterious rating. Kiraly and Qian showed that the classic update
> is online gradient ascent on a Bradley-Terry log-likelihood. The gradient with
> respect to a team’s strength is exactly the residual \(S - E\): outcome minus
> expected win probability. \(K\) is a learning rate. My multiplier reshapes
> that gradient using margin information.
>
> In the selected architecture:
>
> - \(K = 10\)
> - Home-field adjustment \(H = 75\) rating points
> - Margin mode = log
>
> Why 75? Across the five declared architectures I tried 55, 65, and 75. For
> equal-rated teams, 75 points implies
>
> \(P(home) = 1 / (1 + 10^{-75/400}) \approx 60.6\%\).
>
> That is a structural home prior inside Elo, not the 55.4% full-sample rate.
> March selection preferred the stronger HFA in this season’s data.
>
> Elo alone still needs a probability map from `elo_diff` to \(P(home)\). I fit
> a simple L2 logistic with `C = 100` — essentially unregularized — because
> there is only one feature and no multicollinearity to fight. That produces
> the Elo-component probability.

---

## Chapter 9 — Component B: Bradley-Terry + trend (0:44–0:49)

**Do:** Show `_fit_bradley_terry` and the rank model fit in `fit_base_models`.

**Say:**

> Bradley-Terry says
>
> \(\mathrm{logit}\,P(i\ \mathrm{beats}\ j) = \theta_i - \theta_j\),
>
> and with home advantage,
>
> \(\mathrm{logit}\,P = \alpha + \theta_h - \theta_a\).
>
> That is exactly a logistic regression on a +1/−1 team design. I use
> `C = 0.15` on the 30 team coefficients so strengths shrink toward zero when
> data are thin. Early season, that regularization matters.
>
> Why keep BT if Elo is already a BT online learner? Because the estimators
> process information differently:
>
> - Elo updates locally and sequentially — fast to adapt, can trail or overreact.
> - Batch BT re-estimates all strengths from the full history each day — more
>   globally consistent, slower to express a one-game swing unless the data
>   support it.
>
> Trend then asks a different question: is a team’s **recent** point margin
> ahead of or behind its longer EWMA form? That is not the same as “who is
> better.” It is “who is improving relative to themselves.”
>
> The rank component is a second logistic on `(bt_logit, trend_diff)` with
> stronger regularization `C = 0.1`, because now there are two features and the
> BT logit already compresses thirty team parameters.

**Do:** Show `artifacts/coefficient_table.csv` and `permutation_importance.csv`.

**Say:**

> After fitting through the training window, the standardized coefficients show
> BT carrying most of the rank model, with trend a smaller positive coefficient.
> Permutation importance on March agrees: shuffle `bt_logit` and log loss jumps
> by about 0.23; shuffle `elo_diff` and it jumps about 0.02; shuffle
> `trend_diff` and it jumps about 0.006. Trend is a correction, not the engine.

---

## Chapter 10 — Log-odds blend and calibration (0:49–0:52)

**Do:** Open `model.py::blend_probabilities`. Highlight the exact lines.

**Say:**

> Now I have two probabilities for each game: \(p_{Elo}\) and \(p_{rank}\).
> I do **not** average them on the probability scale. On \([0,1]\), averaging
> 0.70 and 0.90 gives 0.80 regardless of how extreme the underlying evidence
> was. On the log-odds scale, those values are 0.85 and 2.20 — the weighted
> average preserves evidence magnitude.
>
> So:
>
> \(z = w\,\mathrm{logit}(p_{Elo}) + (1-w)\,\mathrm{logit}(p_{rank})\)
>
> \(p = \sigma(z / \tau + b)\)
>
> Selected values: \(w = 0.19\), \(\tau = 0.59\), \(b = 0.33\).
>
> Interpretation:
>
> - Weight 0.19 means Elo contributes real but secondary evidence; BT/trend
>   dominates. That matches permutation importance.
> - Temperature 0.59 is less than one, so we **sharpen**. The March grid found
>   the raw components underconfident — probabilities not extreme enough versus
>   outcomes. Dividing the full \(z\) by 0.59 is equivalent to scaling evidence
>   up. I treat that cautiously because March is only 239 games and is also the
>   selection set.
> - Shift 0.33 is a residual home baseline adjustment after the blend.
>
> Temperature scaling is a standard calibration move: one scalar for sharpness,
> without changing the architecture. It is not a license to invent features.

**Invite questions.**

---

# Act V — Selection without looking at April (0:52–1:05)

## Chapter 11 — Declaring candidates before searching (0:52–0:57)

**Do:** Open `configs/architecture_candidates.json` and
`configs/selection_policy.json`.

**Say:**

> If I tune casually until April looks good, I have not built a model — I have
> performed target leakage with extra steps. So I declared the search space up
> front.
>
> Five architectures span responsiveness and regularization:
>
> 1. **balanced** — moderate K and HFA  
> 2. **conservative** — slower Elo, stronger shrinkage  
> 3. **responsive** — higher K, faster adaptation  
> 4. **hfa_75** — stronger home adjustment  
> 5. **lower_bt_shrinkage** — freer team strengths  
>
> For each architecture, the calibration grid is:
>
> - Elo weight from 0.00 to 0.35 step 0.005  
> - Temperature from 0.55 to 0.85 step 0.01  
> - Shift from 0.10 to 0.40 step 0.01  
>
> That is 71 × 31 × 31 = 68,231 calibrations per architecture, about 341k total
> combinations. AUC is computed once per weight because positive affine
> transforms of the logit do not change ranking.
>
> Eligibility rule: beat **all four** March numerical targets. Then
> lexicographically minimize log loss, then Brier, then maximize AUC, then
> accuracy. Architecture name breaks residual ties for determinism.
>
> The selected point is written to `artifacts/selected_spec.json` by the search.
> The scorer loads that file. I do not paste magic numbers into scoring code.

---

## Chapter 12 — March protocol and April exclusion (0:57–1:05)

**Do:** Open `scripts/select_model.py`, then `nba_wp/selection.py::run_selection`,
then `artifacts/selection_proof.json`.

**Say:**

> Here is the chronological protocol:
>
> 1. Truncate the raw input to dates strictly before April 1 **before** feature
>    construction.  
> 2. `run_selection` refuses to proceed if any April row is present.  
> 3. For each architecture, build features on the truncated history.  
> 4. Fit base component coefficients on October through February.  
> 5. Score March one-step-ahead with date-batched updates.  
> 6. Search the calibration grid; keep eligible champions.  
> 7. Emit `selected_spec.json` and `selection_proof.json`.
>
> Proof fields you should care about: selection max date March 31, April rows
> loaded during selection equal to zero, selected architecture `hfa_75`,
> calibration weight 0.19 / temperature 0.59 / shift 0.33.
>
> **Honesty about March.** March is the selection set. Its champion metrics are
> optimistic as estimates of future performance. I will call them selection
> results, not an untouched test.
>
> **Honesty about April.** The executable path cannot load April during
> selection. The broader project had already viewed April during development,
> so I describe April as a **retrospective** final scoring period. I will not
> claim perfect human blindness. What I will claim is machine-enforced
> pre-April selection and no post-hoc retuning after the AUC miss.
>
> After selection freezes, I refit base coefficients through March and score
> April under both sequential and frozen policies.

**Do:** Show `artifacts/march_architecture_results.csv` briefly if time.

**Say:**

> You can see which architectures produced eligible calibrations and how
> `hfa_75` won on the declared rule — not because I typed it into a notebook
> after peeking at April.

**Invite questions.**

---

# Act VI — Results, diagnostics, and a live price (1:05–1:25)

## Chapter 13 — March and April scoreboard (1:05–1:12)

**Do:** Open `artifacts/final_metrics.json`.

**Say:**

> Operational one-step-ahead results:
>
> **March — selection period**
>
> | Metric | Model | Target |
> |---|---:|---:|
> | Log loss | 0.4876 | 0.5096 |
> | Brier | 0.1568 | 0.1676 |
> | AUC | 0.831798246 | 0.831798 |
> | Accuracy | 77.82% | 77.82% |
>
> March clears all four rounded targets. I want to be precise: the AUC and
> accuracy margins are tiny. The AUC edge is on the order of \(10^{-7}\). I will
> not sell that as decisive generalization. It means the selected point cleared
> the eligibility gate under the declared rule.
>
> **April — retrospective scoring**
>
> | Metric | Model | Target | Outcome |
> |---|---:|---:|---|
> | Log loss | 0.4634 | 0.4686 | Beat |
> | Brier | 0.1456 | 0.1506 | Beat |
> | AUC | 0.8502 | 0.8682 | **Miss** |
> | Accuracy | 83.33% | 81.25% | Beat |
>
> I beat the proper scoring targets and accuracy. I missed AUC. I did **not**
> retune. Retuning after seeing the miss would be the same sin as leaking April
> into selection.
>
> Why can proper scores improve while AUC misses? Because they measure
> different things. Log loss and Brier care about probability magnitude —
> calibration and sharpness. AUC cares only about pairwise ordering. You can
> sharpen and recalibrate prices in a way that helps expected log loss without
> fixing every ranking disagreement versus a benchmark ordering.
>
> Frozen-snapshot sensitivity: April frozen log loss is 0.4589, AUC 0.8628 —
> closer on ranking, still short of 0.8682, and accuracy drops versus
> sequential. Within-April state updates are not free. Sometimes they help
> operational pricing; sometimes a short sample overreacts. That is why both
> policies are reported.

**Do:** Open calibration figures.

**Say:**

> Calibration plots for March and April are monitoring artifacts. Temperature
> 0.59 sharpened the blend; I would not freeze that scalar forever in
> production without drift checks.

---

## Chapter 14 — Trace one April game end-to-end (1:12–1:20)

**Do:** Open `outputs/april_predictions.csv`. Filter or scroll to
`2026-04-05`, away `UTA`, home `OKC`.

**Say:**

> Let’s price a concrete game the way I would on a desk.
>
> On April 5, Utah at Oklahoma City. Under the operational policy, the
> performance cutoff for that row is whatever completed before that date —
> earlier April results are allowed, same-day and future are not.
>
> The three champion features on this row are roughly:
>
> - `elo_diff` ≈ 1.43 — large home strength edge on the Elo scale  
> - `bt_logit` ≈ 1.89 — Bradley-Terry agrees, strongly  
> - `trend_diff` ≈ 9.22 — OKC’s recent margin form also outpaces Utah’s  
>
> Component probabilities:
>
> - Elo component ≈ very high  
> - Rank component ≈ very high  
>
> Blend with \(w=0.19\), \(\tau=0.59\), \(b=0.33\) and you get a home price of
> about **99.6%**. Fair decimal odds are essentially 1.00 home and a huge away
> number. Outcome: OKC won. This is an example of a sharp price that was
> justified by all three signals aligning.
>
> Now contrast a painful miss — April 1, Sacramento at Toronto. The model
> priced Toronto around **96.8%**. Sacramento won. Log-loss contribution on that
> single game is about 3.44. That is what confident wrong prices cost under log
> loss — and why log loss matches bookmaker risk better than accuracy. Accuracy
> just says “wrong.” Log loss says “wrong and expensive.”
>
> If you ask me to recompute the blend live, I can do it from the component
> probabilities in the CSV using the formula on the whiteboard — temperature on
> the full \(z\).

**Optional live command:**

```bash
python validate_submission.py --root . --data "$NBA_DATA_PATH" --recompute
```

**Say:**

> The validator rebuilds metrics from the saved prices and, with `--recompute`,
> rebuilds prices from the locked specification. That is the reproducibility
> claim in executable form.

---

## Chapter 15 — Limitations and production roadmap (1:20–1:25)

**Do:** Open `docs/LIMITATIONS_AND_ROADMAP.md`.

**Say:**

> I want to end the prepared narrative by saying what this is **not**.
>
> It is a one-season, team-level technical-task model. It does not know
> injuries, expected starters, minutes, trades, travel distance, time zones,
> altitude, or market prices. It cannot compute valid possessions. March was
> reused for selection. April was historically viewed even though selection
> code excludes it. Fair odds have no overround.
>
> If I were productionizing toward a Bet365 pregame price, my order would be:
>
> 1. **Player availability** — largest missing shock to a team-level price.  
> 2. **Market-implied probabilities** — benchmark and residual blend; closing
>    line value as an external quality metric.  
> 3. **Multi-season hierarchical dynamic strength** — offseason regression,
>    team-specific uncertainty, not a single-season Elo reset story.  
> 4. **Valid possession / Four Factors inputs** — if the feed provides them.  
> 5. **Monitoring and governance** — calibration drift, shadow challengers,
>    versioned quotes, trader-override logging, latency on injury feeds.
>
> Collaboration with traders matters: the two-component design lets someone
> challenge Elo or BT separately instead of arguing with a single opaque score.
>
> In-play is a different model family. This system is pregame. Live pricing
> needs score, clock, and possession state.

---

# Act VII — Close and open floor (1:25–1:30)

## Chapter 16 — Closing position (1:25–1:28)

**Do:** Return to `SUMMARY.md`.

**Say:**

> Let me close with the claim I want you to evaluate me on.
>
> The strongest statement is not “every target was beaten.” It is:
>
> > I built a leakage-audited, reproducible pricing model; I selected it on a
> > machine-enforced pre-April path; I got strong probability scores on the
> > metrics that matter for a book; I missed April AUC and reported that without
> > retuning; and I can show you every stage from raw CSV to a fair price.
>
> That is the working style I would bring to quantitative analysis here:
> probability first, information sets explicit, complexity earned, misses
> visible.

## Chapter 17 — Structured Q&A (1:28–1:30+)

**Say:**

> I’m happy to go deeper on any stage — leakage tests, the Elo derivation,
> the grid search, a modify-on-demand feature experiment, or how I’d wire this
> into an overround and risk layer.

**If they go quiet, offer one of these proactively:**

1. “Want me to walk `build_features` line by line?”  
2. “Want me to derive why Elo is gradient ascent on Bradley-Terry?”  
3. “Want me to add `rest_advantage` live and explain what would need re-selecting?”  
4. “Want the overround arithmetic from a 60% fair price to offered odds?”

---

# Appendix A — Timing cheat card

| Clock | Chapter | Must land |
|---|---|---|
| 0:00–0:04 | Opening | Pricing objective + agenda |
| 0:04–0:08 | Metrics | Log loss primary |
| 0:08–0:13 | Data audit | 1230 / 16 cols / reconciliation |
| 0:13–0:18 | Pregame vs postgame | No inventing pace |
| 0:18–0:24 | Information policy | Four leakage layers |
| 0:24–0:32 | Team state | Elo, BT, trend, rest candidates |
| 0:32–0:38 | Ablation story | B6 loses; B7 wins |
| 0:38–0:44 | Elo depth | MOV multiplier + HFA 75 |
| 0:44–0:49 | BT + trend | +1/−1 design; importance |
| 0:49–0:52 | Blend | Correct temperature formula |
| 0:52–0:57 | Declared search | 5 arch × 68,231 |
| 0:57–1:05 | Selection proof | April rows = 0 |
| 1:05–1:12 | Results | AUC miss without retune |
| 1:12–1:20 | Live game | OKC example + TOR miss |
| 1:20–1:25 | Roadmap | Injuries #1 |
| 1:25–1:30 | Close + Q&A | Reproducible honesty |

If they interrupt heavily, **never skip**: leakage (Ch 5), blend formula
(Ch 10), selection proof (Ch 12), AUC miss (Ch 13).

---

# Appendix B — If you are running long or short

**Running long (cut in this order):**

1. Coefficient table detail  
2. February base-rate aside  
3. Second missed-game example  
4. Architecture name tour beyond `hfa_75`  
5. Calibration plot narration  

**Running short (expand in this order):**

1. Line-by-line `build_features`  
2. Hand-compute blend for the OKC row  
3. Show `test_feature_timing.py` failing scenario verbally  
4. Overround numerical example  
5. Modify-on-demand: point at `MODEL_FEATURES`

---

# Appendix C — One-page story spine (memorize)

1. I need a **price**, so I optimize proper scoring rules.  
2. I audit the CSV and separate pregame from postgame.  
3. I build features **before** I update state; same-day batched.  
4. I engineer many candidates; ablation kills the kitchen sink.  
5. I keep Elo (online) + BT/trend (batch + form).  
6. I blend in log-odds and calibrate with temperature/shift.  
7. I select on March with April code-excluded.  
8. April: strong log loss/Brier, miss AUC, no retune.  
9. Production next: injuries, market, multi-season, monitoring.  
10. Claim: reproducible honesty > metric theater.
