# 90-minute presentation guide

Design: ~45–50 min prepared explanation, 10–15 min live demo, 25–30 min questions.
Numbers quoted below come from `reports/metrics.json` / `reports/model_report.md`;
re-check them there if you regenerate the pipeline.

**Opening statement (memorize):**

> "I treated this as a pricing problem, not a classification contest. The
> deliverable is a calibrated home-win probability for each April game using
> only October-through-March information. I'll show you the leakage controls,
> the pre-March selection protocol, one locked March test, and the frozen
> April forecast — and for every number, the file that proves where it came
> from."

---

## 0–5 min — Decision problem

- **Objective:** frame the task as probability pricing.
- **Talking points:** target `home_win`; product is \(\hat p\) and fair decimal
  odds \(1/\hat p\); log loss is the pricing-relevant proper score; accuracy is
  a by-product.
- **Show:** `README.md` main-result table; `predictions/april_predictions.csv`.
- **Likely question:** "Why log loss and not accuracy?"
- **Answer:** a book is punished by the magnitude of a mispriced probability,
  not by a 0/1 pick; log loss is the proper score that pays exactly that
  penalty; accuracy discards the price.
- **Mistake to avoid:** describing the model as a "winner picker."

## 5–12 min — Dataset and leakage risk

- **Objective:** show you audited before modeling.
- **Talking points:** 1,230 games, 30 teams × 82; wins/losses are *pregame*
  (verified — 0 reconciliation mismatches by replaying every result);
  points/turnovers/fouls/rebounds are *postgame* and would leak if used
  same-game; home rate 55.4%.
- **Show:** `artifacts/current/data_audit.json`; `src/nba_wp/data.py` loader
  guards (duplicates, ties, missing).
- **Likely question:** "How do you know the wins column is pregame?"
- **Answer:** replay the season updating records after each date and compare —
  zero mismatches across all 1,230 rows; the audit JSON stores it.
- **Mistake:** hand-waving "I assumed the docs were right."

## 12–20 min — Temporal protocol

- **Objective:** the governance story.
- **Talking points:** development Oct–Feb; expanding folds train<Jan→Jan and
  train<Feb→Feb; every knob frozen in `configs/model.yaml` before March;
  March scored **once**; April refit through Mar 31, frozen state.
- **Show:** `configs/model.yaml`; `artifacts/current/pre_march_selection_proof.json`
  (0 March, 0 April rows); `tests/test_temporal_protocol.py`.
- **Likely question:** "Could you have tuned on March by accident?"
- **Answer:** the selection entry point truncates at Mar 1 and a guard raises
  on any March-or-later row; tests cover both; the proof JSON records the max
  selection date (2026-02-28).
- **Honesty line:** April was viewed earlier in this project's life; the
  executable pipeline excludes it, but I don't claim historical
  preregistration.
- **Mistake:** claiming April is a pristine holdout.

## 20–30 min — Feature construction

- **Objective:** three features, fully derivable, leakage-safe.
- **Talking points:**
  - `elo_diff` = (R_home − R_away + 65)/400; K=10; log margin-of-victory;
    ratings update *after* the feature row; same-date games batched.
  - `bt_logit` = regularized Bradley–Terry strength difference fit on prior
    games only.
  - `trend_diff` = short-window form (last 10) minus EWMA (half-life 20 d).
  - Rest/turnover/rebound candidates were built and **rejected** on pre-March
    folds (see governance CSV).
- **Show:** `src/nba_wp/features.py`; `artifacts/current/feature_governance.csv`;
  `reports/feature_drift_monthly.csv` (April within training envelope).
- **Likely question:** "Why HFA 65?"
- **Answer:** fixed search default on the Elo/400 scale (≈59% for equal
  teams); it was not tuned on March or April; sensitivity belongs to the
  declared K/half-life/C grid.
- **Mistake:** implying every constant was "optimized" — say fitted / selected
  / fixed-by-design and show the ledger.

## 30–38 min — Baselines and the selected model

- **Objective:** justify the model against simpler alternatives.
- **Talking points (March locked / April frozen):**
  - constant 55%: LL 0.681 / 0.679 — crushed (CI excludes 0).
  - record-difference logistic: LL 0.545 / 0.511 — beaten (CI excludes 0).
  - **Elo-only logistic: LL 0.508 / 0.467 — statistically indistinguishable**
    (Δ per-game LL CI includes 0 on both periods).
  - selected 3-feature logistic: LL 0.510 / 0.469.
- **Show:** `reports/model_report.md` baseline tables;
  `artifacts/current/model_coefficients.json` (β: elo 0.73, bt 0.16, trend
  0.06 standardized).
- **Likely question:** "Why not just ship Elo-only then?"
- **Answer:** the declared pre-March search selected the 3-feature form; a
  post-hoc switch now would be informed by locked-test results — exactly the
  sin the protocol exists to prevent. Elo-only is recorded as the
  pre-registered simpler challenger for the next season.
- **Mistake:** claiming the extra features "add signal" — the data don't
  resolve that.

## 38–47 min — Validation and locked March test

- **Objective:** one locked test, honestly reported.
- **Talking points:** March scored once: LL 0.5103, Brier 0.1672, 182/239
  correct; blend challenger (v1 model) correlates 0.989 with selected and is
  slightly *worse* (Δ CI excludes 0 in selected's favor); benchmark floats are
  retrospective context only.
- **Show:** `artifacts/current/final_metrics.json → locked_march_test`;
  `docs/BENCHMARK_PROVENANCE.md`.
- **Likely question:** "You looked at March twice, surely?"
- **Answer:** the committed artifacts derive from one scored run of the locked
  spec; the validator recomputes them bit-for-bit from raw data
  (`--recompute`, max |Δp| = 0.0).
- **Mistake:** describing rounding-level AUC gaps versus references as wins.

## 47–55 min — Calibration and April predictions

- **Objective:** the deliverable.
- **Talking points:** frozen Mar-31 April: LL 0.4695, Brier 0.1511, AUC 0.862,
  74/96; date-block bootstrap 90% interval for LL ≈ [0.43, 0.51]; calibration
  slope ≈1.44 with ECE ≈0.11 — probabilities modestly under-dispersed, a
  diagnostic not a disaster at n=96; rolling-daily April is a separate
  descriptive file, *not* the assignment result.
- **Show:** `predictions/april_predictions.csv`;
  `reports/figures/april_calibration.png`;
  `artifacts/current/date_block_bootstrap_summary.json`.
- **Likely question:** "Would you recalibrate?"
- **Answer:** not on April — that's the scoring period. With more data,
  Platt-recalibrate on a later validation block and monitor slope drift.
- **Mistake:** recalibrating on the test period, or quoting fair odds as
  offer prices.

## 55–67 min — Live demonstration

Exact sequence (rehearse):

```bash
# 1. Everything green from a clean state
uv run pytest -q

# 2. Regenerate the April deliverable
uv run python -m nba_wp.cli predict \
  --data data/nba-win-probability-data.csv \
  --config configs/model.yaml \
  --output predictions/april_predictions.csv

# 3. Full bit-for-bit verification against raw data
uv run python validate_submission.py --root . \
  --data data/nba-win-probability-data.csv --recompute
```

Then open one row of `predictions/april_predictions.csv`, read
p → fair odds = 1/p, and open `outputs/engineered_features.csv` filtered to the
same `game_id` to show its three feature values.

- **Likely question:** "Change something and show the guard fails."
- **Answer:** run `pytest tests/test_temporal_protocol.py -q`, or demo the
  guard in a Python shell by passing a March row to
  `assert_pre_march_selection_frame` — it raises.

## 67–75 min — Production architecture and monitoring

- **Talking points:** nightly state update job; freeze-and-tag model per
  release; score-then-monitor (log loss vs baseline, calibration slope,
  probability-band hit-rates); rollback = repoint tag; market prices join for
  de-vig comparison *before* any pricing claim; add lineup/injury feeds.
- **Show:** `reports/model_report.md` extensions section;
  `docs/MARKET_PRICING_LIMITATIONS.md`.
- **Mistake:** claiming this repo is production-ready — it's a prototype with
  production hygiene.

## 75–90 min — Questions

### Ten most likely technical questions

1. **Why is log loss the right objective for a sportsbook?** Proper score;
   penalizes mispriced probability magnitude; Brier as quadratic cross-check.
2. **Prove the wins column is pregame.** Replay reconciliation, 0 mismatches
   (`data_audit.json`).
3. **How exactly does Elo update, and when?** K=10, log-MOV multiplier,
   HFA 65 inside the expected-score logistic; update after features are
   written; same-date batching.
4. **Why these three features and not more?** Declared candidates; ablation +
   pre-March folds; richer sets didn't win; governance CSV shows
   built-vs-rejected.
5. **Isn't Elo-only just as good?** Statistically unresolved (CI includes 0);
   switching post-hoc would be locked-test leakage; recorded as challenger.
6. **What does C=0.1 do?** L2 shrinkage on standardized coefficients; chosen
   on the folds, from a 7-point log grid.
7. **Slope 1.44 — your probabilities are underconfident?** Modestly
   under-dispersed, n=96, CI wide; monitor, don't overfit a fix on the test
   set.
8. **What breaks this in production?** Injuries/rest/lineups absent; late-season
   tanking and rotation changes; one season only.
9. **How do I know the CSV I'm looking at came from this code?** Validator
   recomputes predictions from raw data and compares (max |Δp| 0.0);
   manifest hashes every artifact.
10. **What would you do with bookmaker odds?** De-vig, compare log loss vs
    market, CLV study; only then discuss edge — none is claimed here.

### Honest-disclosure list (say before they find it)

- April seen earlier in the project → retrospective, not pristine.
- Elo-only equivalence unresolved.
- Calibration slope diagnostic on a small sample.
- December-fold protocol sensitivity (config comment + model report):
  including a thin Oct–Nov-trained fold flips C to 0.01 and scores worse
  out-of-sample; disclosed rather than absorbed.

---

## 12-slide skeleton

1. Decision problem — price, not pick
2. Data and target — 1,230 games, pregame vs postgame columns
3. Information timing — feature-before-update, same-date batching
4. Temporal split — Oct–Feb dev / Jan–Feb folds / March locked / April frozen
5. Pregame state — Elo, Bradley–Terry, trend definitions
6. Baselines — constant / record / Elo-only
7. Candidate model — direct L2 logistic, coefficients
8. Rolling validation — fold table, 72 candidates, winner
9. Locked March — 0.5103 LL, 182/239, blend challenger comparison
10. Calibration + April — 0.4695 LL, 74/96, reliability, bootstrap CI
11. Production architecture — feeds, monitoring, rollback
12. Limitations and next steps — the four honest disclosures

Appendix slides: benchmark provenance; drift table; ensemble correlation;
December-fold sensitivity; parameter ledger.
