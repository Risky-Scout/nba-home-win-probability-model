# Bet365 evaluation matrix prep

This maps the eight dimensions interviewers are likely scoring to exact
moments, files, and answers. Deliver the interview from
`PRESENTATION_SCRIPT_90MIN.md`. Use this file and
`INTERVIEW_QA_CHEATSHEET.md` for drills.

## Critical correction to the attached prep PDF

The PDF writes the blend as if temperature divides only the rank term.
**That is wrong.** The code does:

```python
# nba_wp/model.py::blend_probabilities
blended_logit = (
    calibration.elo_weight * logit(elo_probability)
    + (1.0 - calibration.elo_weight) * logit(rank_probability)
)
return sigmoid(blended_logit / calibration.temperature + calibration.shift)
```

Correct formula:

\[
z = 0.19\,\mathrm{logit}(p_{Elo}) + 0.81\,\mathrm{logit}(p_{rank})
\]

\[
p = \sigma\!\left(\frac{z}{0.59} + 0.33\right)
\]

Also: the **full-season empirical home-win rate is ~55.4%**
(`artifacts/current/data_audit.json`). Elo HFA=75 implies ~**60.6%** only for
**equal-rated** teams. Do not conflate those two numbers.

---

## The eight dimensions (what earns the offer)

| # | Dimension | Tell signal they want | Your proof |
|---|---|---|---|
| 1 | Leakage discipline | Feature-before-update; same-day batch; April exclusion; frozen vs sequential | `features.py::build_features`, `tests/test_feature_timing.py`, `selection_proof.json` |
| 2 | Probabilistic / pricing thinking | Log loss primary; fair odds; temperature as calibration | `model.py::blend_probabilities`, April `fair_*_decimal_odds` columns |
| 3 | Architecture justification | Why Elo + BT; why log-odds blend; why this grid | `METHODOLOGY.md`, ablation B0–B7 |
| 4 | Selection integrity | March honest; April not in choice | `scripts/select_model.py`, `selection_proof.json`, configs/*.json |
| 5 | Statistical depth | Derive Elo-as-gradient, BT-as-logit, temperature | Practice Part below |
| 6 | Code fluency | Navigate, run, modify live | Function map + drills |
| 7 | Honesty / limitations | AUC miss without retune; March = selection set | `final_metrics.json`, `LIMITATIONS_AND_ROADMAP.md` |
| 8 | Domain / sportsbook | HFA, overround, injuries, production | Closing section |

---

## Dimension 1 — Leakage (make-or-break)

### Four layers (say all four)

1. **Feature-before-update** in `build_features`: refresh BT → read states → write all rows for the date → then update performance.
2. **Same-date batching**: no matinee→night leak. Test: `test_same_day_games_are_batched`.
3. **April exclusion**: truncate before April; `run_selection` raises if April present; proof records `april_rows_loaded_during_selection: 0`.
4. **Frozen snapshot**: `freeze_date="2026-04-01"` stops performance updates; schedule/rest still move.

### Exact question they will ask

> Walk me through the information used to price the April 5 Lakers vs Celtics game.

**60s answer:**

> Operational: all completed games through April 4 update Elo, BT, and trend.
> No same-day or future result is visible. Frozen: only through March 31;
> no April outcome updates April performance state. Both are reported because
> they answer different questions — operational is live-book style; frozen is
> the strict month-start sensitivity.

### Trap

> Turnovers/fouls/rebounds are on the same row — how do you not leak?

> They update history **after** the feature row is written. Same-game values
> never enter that row or any same-date row. Proven by
> `test_current_game_postgame_values_do_not_change_current_features`.

### Subtle

> In frozen mode, do rest days still update?

> Yes. Schedule is observable without outcomes. Only performance state freezes.

---

## Dimension 2 — Pricing, not classification

### Lead with price language

Say: “the model **prices** home at 65%,” not “predicts the home team wins.”

### Why log loss primary

- Strictly proper local scoring rule.
- Wrong-and-confident (0.99) is catastrophic for liability; wrong-and-cautious (0.51) is not.
- Accuracy treats 51% and 99% the same if both are ≥ 0.5.

### Temperature 0.59

> March grid found components underconfident. Dividing the **full** blended
> logit by 0.59 sharpens. Shift 0.33 adjusts residual home baseline. Treat
> cautiously: March is 239 games and also the selection set.

### Fair odds

`fair_home = 1/p`, `fair_away = 1/(1-p)` — zero-margin mathematical odds, not
a customer quote (no overround / liability / trader override).

---

## Dimension 3 — Architecture

### Why two components

| Component | Role |
|---|---|
| MOV Elo | Online / sequential strength tracker |
| BT + trend | Batch global ranking + recent-vs-long form |

Kiraly & Qian: Elo ≈ online gradient ascent on BT log-likelihood. You combine
online + batch estimators, then calibrate.

### Why Elo weight only 0.19

Grid searched 0–0.35. Best eligible was 0.19 — Elo useful but secondary.
Permutation importance: **BT ≫ Elo > trend**.

### Why not XGBoost

Small N, collinear strength features, need deterministic probs + interpretability
+ trader-overridable components. Promote nonlinear only after forward proper-score win.

### MOV multiplier (know cold)

```text
m = log(|margin|+1) * (2.2 / max(0.25, rating_diff*0.001 + 2.2))
```

Log dampens blowouts; denominator down-weights expected beatdowns.

### C values

- `elo_model_c = 100` — one feature (`elo_diff`); near-unregularized is fine.
- `rank_model_c = 0.1` — two features with BT strengths behind them; needs shrinkage.
- `bt_c = 0.15` — ridge on the 30 team coefficients.

---

## Dimension 4 — Selection integrity

### Three proofs

1. **Code**: truncate `< 2026-04-01`; raise if April present.
2. **Config-as-code**: architectures, grid, benchmarks in `configs/`.
3. **Generated spec**: `selected_spec.json` written by search, loaded by scorer.

### Honesty line (strength, not weakness)

> March is the selection set, so its champion score is optimistic. April is
> retrospective: code cannot load it during selection, but I do not claim
> perfect human blindness across the whole project.

---

## Dimension 5 — Seven derivations to practice aloud

1. **Elo as gradient ascent:** \(R \leftarrow R + K m (S-E)\); residual \(S-E\) is the BT log-likelihood gradient.
2. **BT as logistic:** design matrix +1 home / −1 away; intercept = HFA; coeffs = team strengths.
3. **Log-odds blend + temperature on full \(z\)** (see correction above).
4. **EWMA half-life 45:** \(w = 0.5^{\mathrm{age}/45}\); trend = short10 − EWMA.
5. **Beta(4,4):** \((W+4)/(N+8)\); early-season shrinkage (candidate feature, not champion).
6. **AUC vs proper scores:** ranking vs calibration; explains April AUC miss.
7. **Paired bootstrap:** game-level resample of log-loss differences; 2,000 reps; seed 365.

---

## Dimension 6 — Code fluency map

| Open this | Hook |
|---|---|
| `data.py::load_games` / `audit_games` | game_id as string; record reconciliation |
| `features.py::build_features` | date loop; leakage control |
| `features.py::_team_state` | Beta(4,4), EWMA, rest |
| `features.py::_fit_bradley_terry` | +1/−1 design |
| `features.py::_elo_multiplier` | log MOV |
| `model.py::fit_base_models` | separate Elo vs rank pipelines |
| `model.py::blend_probabilities` | log-odds blend (correct formula) |
| `model.py::search_calibration` | 68,231 / architecture; eligibility |
| `selection.py::run_selection` | reject April; lexicographic rule |
| `reporting.py::ablation_table` | B0–B7 |
| `scripts/select_model.py` | hard truncate |
| `validate_submission.py` | quality gate |

### Live drills

1. `python validate_submission.py --root . --data "$NBA_DATA_PATH"` → PASS
2. Trace one April row: features → component probs → blend → fair odds
3. `python -m pytest tests/ -q`
4. Ablation CSV + April AUC miss in `final_metrics.json`
5. Optional modify: add `rest_advantage` to `MODEL_FEATURES` and explain re-selection needed

---

## Dimension 7 — Honesty

### April scoreboard (memorize)

| Metric | Model | Target | Result |
|---|---:|---:|---|
| Log loss | 0.4634 | 0.4686 | Beat |
| Brier | 0.1456 | 0.1506 | Beat |
| AUC | 0.8502 | 0.8682 | **Miss** |
| Accuracy | 83.33% | 81.25% | Beat |

### If they say “why not retune?”

> That would be selection on the test set. Calibration can improve proper
> scores without fixing ranking. Frozen April AUC is 0.8628 — closer, still
> short. I reported the miss.

### If they say “is AUC miss a failure?”

> For a book, calibration drives price quality and liability. Ranking matters,
> but I will not hide a miss or quietly chase it.

---

## Dimension 8 — Sportsbook / NBA

### HFA = 75 Elo points

Equal teams: \(P(home)=1/(1+10^{-75/400})\approx 0.606\). That is the **structural**
HFA embedded in Elo, not the full-sample win rate (~55.4%).

### Overround sketch

Fair \(p=0.60\) → odds 1.667. With 5% overround roughly:
offered home \(0.60\times1.05=0.63\) → 1.587; away \(0.40\times1.05=0.42\) → 2.381.

### Production priority order

1. Player availability / expected minutes  
2. Market-implied probs for benchmarking / residual blend  
3. Multi-season hierarchical dynamic strength  
4. Valid possessions / Four Factors (cannot invent from points)  
5. Monitoring: calibration drift, CLV, trader overrides, shadow challengers  

### In-play?

Not this model. Pregame only. In-play needs score, clock, possession state.

---

## Self-scorecard (practice until green)

| Dimension | Can you do this without notes? | ✓ |
|---|---|---|
| 1 | List 4 leakage layers + answer April 5 information set | |
| 2 | Explain log loss vs accuracy with a liability example | |
| 3 | Write the **correct** blend formula from memory | |
| 4 | Show April exclusion in code + proof JSON | |
| 5 | Derive Elo-as-gradient and BT +1/−1 design | |
| 6 | Trace one April prediction end-to-end live | |
| 7 | Lead with AUC miss; refuse to retune | |
| 8 | Convert fair p → overround odds; name injury as #1 gap | |

---

## 60-second summary (memorize)

> A calibrated log-odds blend of margin-of-victory Elo and regularized
> Bradley-Terry with recent trend. Three features: `elo_diff`, `bt_logit`,
> `trend_diff`. Five architectures × ~68k calibrations on March; April
> code-excluded from selection. March beats all four rounded targets. April
> beats log loss, Brier, and accuracy, misses AUC — reported without retuning.
> Both sequential and frozen outputs are exported because they answer different
> operational questions.

## Behavioral rules from the rubric

1. Lead every explanation with the information timeline.
2. Frame numbers as **prices**.
3. Present the AUC miss early.
4. Know what one season cannot identify.
5. Resist complexity that did not earn a proper-score win.
6. Never be defensive about transparent simplicity.
