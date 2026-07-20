> **Champion correction (read first):** The live model on this branch is a
> **direct L2 logistic** (`k10_hl20`, `C=0.1`) selected on **pre-March** folds.
> Primary April = **frozen March 31**. Parameter ledger:
> `docs/interview/PARAMETER_LEDGER.md`. Ignore any answer text that still
> describes a temperature/shift blend as the champion — that is the v1
> challenger path only (`v1-original-submission`).
>
> Artifact root for current numbers: **`artifacts/current/`**.

---

# Senior Quant Q&A — every question with answers and files to show

Use during the interview when they interrupt. Each answer ends with
**Show:** the file to open. Facts below match the repository (corrections
noted where common prep docs are wrong).

**Critical corrections vs some prep PDFs:**

- Full-sample home-win rate is **~55.4%** (`data_audit.json`), not 60.6%.
- Elo HFA=75 implies **~60.6% only for equal-rated teams**.
- March home-win rate is **~60.3%** (`final_metrics.json`).
- Blend: `p = sigmoid(z / temperature + shift)` — temperature divides **full** `z`.
- Shift is **+0.33 after** temperature scaling (not `0.33/0.59` inside the division).
- Permutation seed **365**; bootstrap seed **2026**.

**Blend formula:**

```text
z = 0.19 * logit(p_elo) + 0.81 * logit(p_rank)
p = sigmoid(z / 0.59 + 0.33)
```

---

## Phase 1 — Opening

### Q1. Summarize your model in 90 seconds.

> A calibrated log-odds blend of margin-of-victory Elo and regularized
> Bradley-Terry with recent trend. Three features: `elo_diff`, `bt_logit`,
> `trend_diff`. Five architectures × ~68k calibrations on March; April
> code-excluded. March beats all four targets. April beats log loss, Brier,
> accuracy; misses AUC — reported without retuning. Sequential and frozen
> outputs both exported.

**Show:** `SUMMARY.md`

### Q2. What is the output?

> Calibrated home-win probability plus fair zero-margin decimal odds
> `1/p` and `1/(1-p)`. A **price**, not a binary classifier label.

**Show:** `predictions/april_predictions.csv` columns
`home_win_probability`, `fair_home_decimal_odds`, `fair_away_decimal_odds`

### Q3. Why not just predict who wins?

> Bet365’s product is a price. 55% vs 65% is 1.82 vs 1.54 odds — and
> liability. Log loss punishes confident wrong prices; accuracy does not.

**Show:** `docs/METHODOLOGY.md` metrics section

### Q4. What metric for a sportsbook?

> Log loss — strictly proper local scoring rule. Brier proper but quadratic.
> AUC/accuracy are not proper scoring rules for calibration.

**Show:** `configs/model.yaml (benchmarks section)`

---

## Phase 2 — Architecture

### Q5. Why two components?

> Elo = online gradient ascent on BT likelihood (adapts game-by-game).
> BT = batch MLE (globally consistent). Different information processing;
> blend is safer at N=1,230.

**Show:** `docs/METHODOLOGY.md`; `src/nba_wp/model.py::fit_base_models`

### Q6. Why not XGBoost / NN?

> Small N, collinear strength features; need direct probs, deterministic
> training, speed, coefficient interpretability, trader-overridable parts.
> Promote nonlinear only after forward proper-score win.

**Show:** `artifacts/current/feature_group_ablation.csv` (B6 loses)

### Q7. Bradley-Terry step by step.

> Each date: design matrix +1 home / −1 away (30 cols); fit
> `LogisticRegression(C=0.15)`; coeffs = θ, intercept = HFA;
> `bt_logit = decision_function(matchup)`.

**Show:** `src/nba_wp/features.py::_fit_bradley_terry`

### Q8. BT vs Elo mathematically.

> BT: logit P = θ_i − θ_j (+ HFA). Elo update R ← R + K(S−E) is one-step
> SGD on that log-likelihood; K = learning rate; (S−E) = gradient.

**Show:** `docs/METHODOLOGY.md` Elo + BT sections

### Q9. Why refit BT daily — expensive?

> Milliseconds at 1,230 games / 30 coeffs. Cached until game count changes.
> Multi-season production → incremental updates.

**Show:** `build_features` cache `cached_bt_game_count`

### Q10. MOV multiplier.

> `log(|margin|+1) * (2.2 / max(0.25, rating_diff*0.001 + 2.2))`.
> Log dampens blowouts; denominator down-weights expected beatdowns.

**Show:** `src/nba_wp/features.py::_elo_multiplier`

### Q11. Why K=10?

> Architectures tried 7.5, 10, 15. Conservative/responsive (7.5/15) had
> **zero** eligible March calibrations. Winner `hfa_75` uses K=10.
> Rough equal-team 10-pt win move ≈ 10·log(11) ≈ 24 rating points with MOV.

**Show:** `artifacts/current/pre_march_selection_results.csv`

### Q12. Why HFA=75?

> Searched 55/65/75. Equal teams: P(home)=1/(1+10^(−75/400))≈60.6%
> **structural**. Full-sample rate is ~55.4%; March ~60.3%. `hfa_75` best
> eligible March LL.

**Show:** `configs/model.yaml` + `march_architecture_results.csv`

### Q13. What is trend_diff?

> short (last 10 margins) − EWMA (half-life 45d); home − away.
> Positive = outrunning own longer form.

**Show:** `features.py::_team_state`, `_ewma`

### Q14. Why Elo weight only 0.19?

> Grid 0–0.35 step 0.005; best eligible = 0.19. Real but secondary.
> Permutation: BT ≫ Elo > trend.

**Show:** `march_tuning_top_candidates.csv`; `permutation_importance.csv`

### Q15. Why log-odds blend?

> Logits are additive evidence. Averaging probs loses magnitude.
> Temperature scales full evidence uniformly on logit scale.

**Show:** `model.py::blend_probabilities`

### Q16. Temperature 0.59?

> `p = sigmoid(z/τ + b)`. τ<1 sharpens. March grid found underconfident
> components. Cautious: March is selection set, 239 games.

**Show:** `search_calibration` + `march_tuning_top_candidates.csv` row 1

### Q17. Shift 0.33?

> Constant added **after** `/τ` on the logit scale — residual home baseline
> adjustment. Selected jointly with w and τ on March.

**Show:** same tuning CSV; do **not** say shift is “0.33/0.59 inside τ”

### Q18. elo_model_c=100 vs rank_model_c=0.1?

> Elo submodel: one feature → near-unregularized OK.
> Rank: two features on top of 30 BT params → need shrinkage.

**Show:** `selected_spec.json` / `configs/model.yaml`

### Q19. Full probability for one game.

> (1) p_elo from elo_diff (2) p_rank from bt_logit, trend_diff
> (3) z = 0.19·logit(p_elo)+0.81·logit(p_rank)
> (4) p = sigmoid(z/0.59+0.33) (5) fair_home=1/p

**Show:** `april_predictions.csv` + `blend_probabilities`

### Q20. Why only three features?

> Ablation: rich challenger B6 worse than Elo alone; blend B7 best.
> Complexity must earn proper-score improvement.

**Show:** `feature_group_ablation.csv`

---

## Phase 3 — Leakage

### Q21. Info set for April 5 LAL vs BOS (or any April game).

> Sequential: all completed games through April 4. Frozen: through March 31
> only. No same-day/future either way.

**Show:** `april_predictions.csv` `state_policy`, `performance_cutoff`

### Q22. TOV/REB/fouls on same row — leakage?

> Read state → write features → then append box advantages to history.
> Test changes points/TOV; features unchanged.

**Show:** `build_features` update-after-write;
`tests/test_feature_timing.py::test_current_game_postgame_values_do_not_change_current_features`

### Q23. Same-day games?

> Batched: all rows written before any updates.
> Test: change game 0 score; game 1 features identical.

**Show:** `test_same_day_games_are_batched`

### Q24. Exact leakage-prevention code.

> Date loop: refresh BT → write all rows → update schedule → maybe skip
> performance if frozen → else Elo/history/prior_games updates.

**Show:** `src/nba_wp/features.py::build_features` (scroll the whole date loop)

### Q25. Frozen: do rest days update?

> Yes. Schedule observable without outcomes. Performance state frozen.

**Show:** schedule update before `if freeze ... continue`

### Q26. Is rest a leak / future schedule?

> Rest from public schedule dates known before tip. Pregame observable.

**Show:** `_team_state` rest computation

### Q27. Are CSV home_wins/losses pregame?

> Yes. `_record_reconciliation` replays season; audit mismatch count 0.

**Show:** `data.py::_record_reconciliation`; `data_audit.json`

### Q28. Zero games — record?

> Beta(4,4): (0+4)/(0+8)=0.5; logit=0. Neutral.

**Show:** `_team_state` smoothed win probability

### Q29. Team two games same day?

> Audit count 0. Even if, batching would give both same pregame state.

**Show:** `data_audit.json` `same_team_multiple_games_same_date_count`

---

## Phase 4 — Selection integrity

### Q30. How do I know you didn’t peek at April?

> Truncate `< 2026-04-01` before features; `run_selection` raises on April;
> proof `april_rows_loaded_during_selection: 0`; validator checks.

**Show:** `python -m nba_wp.cli select`; `selection.py`; `selection_proof.json`

### Q31. But you saw April in development.

> Describe April as **retrospective**. Claim is machine-enforced pre-April
> selection path, not perfect human blindness.

**Show:** `docs/VALIDATION_AND_GOVERNANCE.md` / LIMITATIONS

### Q32. If never seen April?

> Executable selection identical. Exploration history might differ; final
> declared-policy selection is mechanical.

**Show:** `configs/model.yaml`

### Q33. Is March unbiased?

> No — selection set → optimistic champion score.

**Show:** `selected_spec.json` notes

### Q34. What is unbiased performance?

> No truly unbiased estimate with one season. April best available OOS check
> with stated caveats.

**Show:** `final_metrics.json`

### Q35. Why not retune to beat April AUC?

> That is test-set selection / leakage. Reported miss; proper scores can
> improve without fixing ranking.

**Show:** `final_metrics.json` April AUC vs target

### Q36. How many candidates?

> 5 × 71 × 31 × 31 ≈ 341k; ~68,231 per architecture.

**Show:** `configs/model.yaml`; `march_architecture_results.csv` `candidate_count`

### Q37–Q38. March / April targets.

> March: LL<0.509645, Brier<0.167618, AUC>0.831798, Acc>0.7782  
> April: LL<0.468596, Brier<0.150628, AUC>0.868196, Acc>0.8125  
> April: beat LL/Brier/Acc; miss AUC.

**Show:** `configs/model.yaml (benchmarks section)`; `final_metrics.json`

### Q39. sequential vs frozen.

> Sequential: within-month state updates after completed dates.  
> Frozen: performance fixed at month start; schedule still moves.

**Show:** `final_metrics.json` both blocks; `build_features(freeze_date=...)`

### Q40. Why frozen April LL better?

> Ops 0.463 vs frozen 0.459 — within-April updates can overreact on small
> samples. Both reported.

**Show:** `final_metrics.json` `frozen_snapshot_sensitivity.april`

### Q41. Lexicographic rule.

> Eligible first; then min LL, min Brier, max AUC, max Acc, arch name.
> No April metric in the key.

**Show:** `configs/model.yaml`; sort in `selection.py` / `search_calibration`

---

## Phase 5 — Basketball domain

### Q42. NBA home-court advantage.

> Historical ~60–62% often cited. **This file:** full sample ~**55.4%**;
> March ~60.3%. Elo HFA 75 ⇒ ~60.6% for **equal** teams. Drivers: crowd,
> travel, referees, familiarity.

**Show:** `data_audit.json` `home_win_rate`; compute 10^(−75/400)

### Q43. Biggest missing feature?

> Expected player availability / minutes. Priority #1 roadmap. 5–10 pt swings.

**Show:** `docs/LIMITATIONS_AND_ROADMAP.md`

### Q44. Why no pace?

> Need FGA + 0.44·FTA − OREB + TOV. Missing FGA/FTA/OREB/makes.
> Points/constant ⇒ constant ORtg — invalid.

**Show:** `docs/FEATURE_ENGINEERING.md`

### Q45–Q46. B2B / rest?

> Computed (`back_to_back`, games_in_4/6). Not selected — B6 didn’t improve
> proper scores. May matter more with more data / playoffs.

**Show:** `_team_state`; ablation B6

### Q47. Playoffs?

> Dataset is regular season only. Different intensity/rest/strategy —
> needs separate calibration.

**Show:** date range in `data_audit.json`

### Q48. Travel / TZ?

> Not in data. On roadmap.

**Show:** LIMITATIONS

### Q49. Why advantages not raw TOV/REB/fouls?

> Sign convention: positive = good for team
> (opp TOV − own TOV, own REB − opp REB, opp fouls − own fouls).

**Show:** history append in `build_features` after features written

### Q50. Star injured after March 31?

> Frozen doesn’t adjust; operational only via later margins. Limitation #2.

**Show:** LIMITATIONS

---

## Phase 6 — Production / sportsbook

### Q51. Overround.

> Simple: odds = 1/(p·(1+m)). Better: asymmetric by liability/demand.

**Show:** fair odds columns; MODEL_CARD / LIMITATIONS “not customer quote”

### Q52. Blend with market.

> De-vig close → p_blend = w·p_mkt + (1−w)·p_model. Market has injuries.

**Show:** LIMITATIONS roadmap market item

### Q53. Speed.

> BT refit ms; Elo O(1); full pipeline seconds. Multi-league → incremental BT.

**Show:** run validator live

### Q54. Monitoring.

> LL/Brier, calibration-in-large/slope, price bands, drift, freshness,
> injury latency, CLV, overrides, P&L, shadow challengers.

**Show:** LIMITATIONS monitoring list

### Q55. Closing-line value.

> Consistently closer to close than open ⇒ model quality vs market consensus.

### Q56. Trader override.

> Log override; keep model quote; compare over time. Can nudge Elo or BT
> separately.

### Q57. In-play?

> Not this model. Needs score/clock/state. Separate exercise.

### Q58. How know model is wrong?

> Drift on LL/calibration; shadow challengers; governance promotion.

### Q59. Versioning.

> Architecture + calibration + hash on every quote; `trained_model.joblib`.

**Show:** `artifacts/trained_model.joblib`; `selected_spec.json`

### Q60. Production architecture.

> Fundamental model → trader adjustments → open → risk/liability → in-play
> separate stack.

---

## Phase 7 — Statistical theory

### Q61. Derive Elo from first principles.

> One-game BT log-likelihood; ∂/∂θ_home = y−p; ascent with rate K ⇒ Elo.
> MOV reshapes the gradient step.

**Show:** METHODOLOGY + `_elo_multiplier`

### Q62. Why log loss strictly proper?

> Expected LL uniquely minimized at true p; local (depends only on realized
> outcome’s probability).

### Q63. Temperature doesn’t change AUC.

> τ and shift are positive affine on logits → order preserved.
> Search computes AUC once per weight.

**Show:** `search_calibration` docstring

### Q64. Beta(4,4).

> Posterior mean (W+4)/(N+8); 8 neutral pseudo-games; avoids 0/1 early season.

**Show:** `_team_state`

### Q65. AUC not proper.

> Ranking only; can have AUC=1 with terrible calibration (0.51 vs 0.49).

### Q66. Paired bootstrap.

> Resample games; LL_A − LL_B; if 95% CI includes 0, not significant.
> Seeds: bootstrap 2026.

**Show:** `artifacts/paired_bootstrap_vs_*.json`

### Q67. Permutation importance vs coef size.

> Shuffle feature; ΔLL. Captures dependence; better under correlation.
> Seed 365, 100 repeats.

**Show:** `permutation_importance.csv`

### Q68. StandardScaler.

> Pipeline scale + LR. Coef/scale = raw-unit effect. Fair L2 across units.

**Show:** `coefficient_table.csv` `raw_unit_coefficient`

### Q69. AUC miss vs better LL/Brier.

> Calibration/magnitude ≠ ranking. Can sharpen usefully while some pairwise
> orders disagree with benchmark.

**Show:** `final_metrics.json`

---

## Phase 8 — Code fluency

### Q70. Walk `build_features`.

> Sort → init 1500/histories → per date: BT refresh → write all rows →
> schedule update → freeze skip? → Elo/history/prior updates.

**Show:** `features.py::build_features`

### Q71. `_team_state` returns?

> games, record_logit, margins, trend, box advantages, rest, B2B, density.
> Empty history → defaults (rest 7 or from schedule).

**Show:** `_team_state`

### Q72. Show April exclusion.

**Show:** `select_model.py` truncate; `selection.py` ValueError;
`selection_proof.json`

### Q73. Run validator.

```bash
python3 validate_submission.py --root . --data "$NBA_DATA_PATH"
```

Expect PASS, 1230 rows, max date 2026-03-31, April 0, March 239, April 96.

### Q74. Unit tests.

```bash
python3 -m pytest tests/test_feature_timing.py -q
```

Three leakage tests: postgame no self-leak; same-day batch; frozen freeze.

### Q75. Add a feature now?

> Add to `_team_state` / feature row → rerun selection → promote only if
> March proper scores improve under declared rule.

**Show:** `MODEL_FEATURES` in `features.py`

### Q76. Trace one April prediction.

**Show:** `april_predictions.csv` → `engineered_features.csv` same id →
`blend_probabilities`

### Q77. Ablation rows B0–B7.

**Show:** `feature_group_ablation.csv` (narrate B0→B7)

### Q78. Data audit.

**Show:** `data_audit.json` — 1230, 30 teams, 82 each, nulls/dupes/ties 0,
home rate **~55.4%**, record reconcile OK.

### Q79. Correlations.

> elo_diff ↔ bt_logit (same strength, different estimators).
> cumulative ↔ recent margins highly correlated.
> trend least correlated — independent form signal.

**Show:** `artifacts/feature_correlations.csv` /
`figures/feature_correlation_matrix.png`

### Q80. Change random seed?

> BT `random_state=0` (solver path; usually stable with L2).
> Permutation seed 365; bootstrap 2026 — changes those diagnostics only,
> not locked prices if models unchanged.

**Show:** `reporting.py` seeds; `features.py` BT `random_state=0`

---

## Phase 9 — Edge cases

### Q81. Team with zero games.

> Elo 1500; BT shrunk ~0; trend 0; record logit 0 → ~structural HFA price.

### Q82. Only 10 games in dataset.

> BT returns None if games < n_teams; bt_logit=0; Elo barely moved.

**Show:** `_fit_bradley_terry` early return

### Q83. Expansion teams mid-season.

> Start 1500; BT after enough games; early uncertainty high.

### Q84. Overfitting at 1,230?

> 3 champion features; regularized BT; 3-parameter calibration; ablation
> shows richer sets don’t help — complexity constrained by data.

### Q85. Multicollinearity.

> elo_diff and bt_logit in **separate** logistics. Rank: bt vs trend less
> collinear. Scaler helps L2.

### Q86. Schedule format change.

> Uses dates + margins; rest logic may need retuning for new patterns.

### Q87. Overtime.

> Final margin includes OT; simplification vs regulation closeness.

### Q88. Other sports?

> Elo/BT portable; MOV/HFA/trend sport-specific; draws break no-draw assume.

### Q89. Bad data?

> Schema, types, nulls, dupes, same-team, ties, record reconcile →
> `DataValidationError`.

**Show:** `data.py::load_games`

### Q90. Ten seasons?

> Hierarchical dynamic strength with offseason regression + player avail.

**Show:** LIMITATIONS roadmap formula

---

## Phase 10 — Roadmap / behavioral

### Q91–Q93. Top improvements.

> 1) Player availability 2) Multi-season hierarchical 3) Market residuals/CLV

**Show:** LIMITATIONS

### Q94. Four Factors.

> Need real possession ingredients — missing now.

### Q95. Offseason.

> s_pre = mean + φ(s_final − mean), φ<1 from YoY correlation.

### Q96. Risk management.

> Calibration drift, shadow deploy, market compare, overround, versioned
> quotes, stale-state alerts.

### Q97. Why Bet365?

> Probabilistic pricing for real-time markets; trader/dev collaboration;
> Elo/BT are sportsbook-native building blocks.

### Q98. Biggest weakness?

> One season; March selection reuse; no injuries/possessions/market.

### Q99. What learned?

> Information ordering harder than model choice; simple structure hard to
> beat at small N; reporting AUC miss builds credibility.

### Q100. Start over?

> Define information timeline first; write validator and benchmarks before
> optimizing metrics.

---

## Phase 11 — Adversarial / “are you lying?”

### Q101. Set K=20 now.

> Faster, noisier Elo. Responsive K=15 already ineligible on March.
> BT side (weight 0.81) unchanged. Likely hurts eligibility.

**Show:** `march_architecture_results.csv` responsive row

### Q102. Remove trend.

> Rank becomes BT-only logistic; still runs; LL slightly worse
> (importance ~+0.006).

**Show:** `permutation_importance.csv`

### Q103. Temperature = 1.0.

> No sharpening; probs closer to 0.5; LL likely worse; **AUC unchanged**.

**Show:** `blend_probabilities`

### Q104. Pregame records wrong?

> Reconciliation mismatch would flag. Champion features don’t use raw W–L
> columns — only Elo/BT/trend path.

**Show:** `data_audit.json`; `MODEL_FEATURES`

### Q105. Neutral-site HFA=0?

> No neutral flag in data; currently always apply H. Would need schedule flag.

### Q106. p exactly 0 or 1?

> Clips: features logit 1e-9; model/eval/reporting 1e-12 — avoid log(0).

**Show:** `_safe_logit`; `model.py` clips; `reporting.py` clips

### Q107. Team plays itself?

> `DataValidationError` in `load_games`.

### Q108. Next season re-select?

> New Oct–Feb / March / April; re-run declared search; don’t blindly reuse
> old τ,b.

### Q109. EV to Bet365?

> LL reduction vs constant prior; tighter prices; CLV if closer to close —
> needs market data to monetize.

**Show:** ablation B0 vs B7 LL

### Q110. Why hire you vs more complex models?

> Prices not picks; April mechanically excluded; AUC miss reported;
> production-shaped (fast, deterministic, decomposable). Complexity without
> proper-score gain is a liability.

---

## Numbers to know cold (corrected)

| Item | Value |
|---|---|
| Games / teams / per team | 1230 / 30 / 82 |
| Full-sample home rate | **~55.4%** |
| March home rate | **~60.3%** |
| Elo HFA 75 → equal-team P(home) | **~60.6%** |
| March / April N | 239 / 96 |
| Champion | hfa_75; K=10; HFA=75; BT C=0.15; trend 45/10 |
| Calibration | w=0.19, τ=0.59, b=0.33 |
| March LL/Brier/AUC/Acc | 0.4876 / 0.1568 / 0.831798 / 77.82% |
| April seq LL/Brier/AUC/Acc | 0.4634 / 0.1456 / **0.8502** / 83.33% |
| April frozen LL/AUC | 0.4589 / 0.8628 |
| Permutation / bootstrap | seed 365 ×100 / seed 2026 ×2000 |
