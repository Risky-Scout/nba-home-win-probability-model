# 90-minute presentation script (remediated champion)

**How to use.** Read **Say** almost verbatim. Do **Do**. When you see
**Show**, leave that file on screen. Keep `docs/SENIOR_QUANT_QA.md` open for
interrupts.

**Parameter rule (repeat often):**

> Every number is **Fitted**, **Selected** on pre-March folds, or **Fixed by
> design**. I will show the file that proves which.

**Champion (do not recite the old blend):**

\[
\operatorname{logit}\hat p=\hat\beta_0+\hat\beta_1\Delta\mathrm{Elo}
+\hat\beta_2\Delta\mathrm{BT}+\hat\beta_3\Delta\mathrm{Trend}
\]

Selected: \(K=10\), HFA \(=65\), half-life \(=20\) days, logistic \(C=0.1\).  
Primary April: **frozen March 31** (`outputs/april_predictions_frozen_snapshot.csv`).

**Fact traps:**

- Full-sample home rate ≈ **55.4%** (`data_audit.json`) — not 60.6%.
- Elo HFA 65 is a **search default**, not “60.6% overall home rate.”
- March was **not** used for selection on this branch.
- Do not claim sportsbook alpha.

**Setup:** `START_HERE.md` + `docs/CURSOR_PRESENTATION_SETUP.md`.

**Tabs:** see `docs/interview/FILE_INDEX.md`.

---

# Act I — Frame (0:00–0:10)

## 1. Opening (0:00–0:04)

**Do:** `SUMMARY.md` then `START_HERE.md`.

**Say:**

> Thanks for the time. I’ll walk the model chronologically: data controls,
> features, selection, locked tests, and April scoring — with every parameter
> traceable to a file.
>
> The assignment asks for home-win probabilities for April using October through
> March information. I treat this as a **probability pricing** prototype: the
> product is a calibrated \(\hat p\), and fair decimal odds \(1/\hat p\), not a
> pick’em contest.
>
> The champion is a **direct L2 logistic** on three leakage-controlled features:
> Elo difference, Bradley-Terry logit difference, and trend difference.
> Architecture and regularization were chosen on **January and February**
> expanding folds only. March is a locked test. The **primary April result**
> freezes state at March 31.
>
> Please interrupt anytime. When you ask “where does that number come from?”,
> I’ll open the ledger or the artifact.

**Invite questions.**

## 2. Objective and metrics (0:04–0:10)

**Do:** `docs/METHODOLOGY.md`, `configs/selection_policy.json`,
`configs/benchmarks.json`.

**Say:**

> Target \(Y=1\) if home points exceed away points. Primary proper score is
> **log loss**. Brier is secondary. AUC and accuracy are descriptive.
>
> External floats in `benchmarks.json` are **retrospective references only**.
> They do **not** gate selection — that mistake was removed in remediation.
> Provenance: `docs/BENCHMARK_PROVENANCE.md`.

**Show:** `docs/interview/PARAMETER_LEDGER.md` section 1–2.

---

# Act II — Data and leakage (0:10–0:22)

## 3. Data audit (0:10–0:14)

**Do:** `nba_wp/data.py`, `artifacts/current/data_audit.json`.

**Say:**

> 1,230 games, Oct 21 2025 through Apr 12 2026. Wins/losses columns are
> **pregame** records; points/turnovers/fouls/rebounds are **postgame** box
> scores. Using postgame totals as raw predictors would leak. Strength features
> are built only from history available before the game.

**Show:** home rate ≈ 0.554 in `data_audit.json`.

## 4. Feature-before-update (0:14–0:22)

**Do:** `nba_wp/features.py`, `docs/FEATURE_ENGINEERING.md`,
`artifacts/current/feature_governance.csv`.

**Say:**

> Three engineered signals:
>
> 1. **Elo** with log margin-of-victory and home-court constant HFA — ratings
>    update after the feature row is written; same-day games are batched.
> 2. **Bradley-Terry** strengths fit on prior games only → `bt_logit`.
> 3. **Trend**: short 10-game form minus EWMA with selected half-life →
>    `trend_diff`.
>
> I also built richer candidates (rest, turnovers, rebounds). Ablation and
> pre-March selection did not promote them into the champion. Governance table:
> `feature_governance.csv`.

---

# Act III — Model form (0:22–0:40)

## 5. Why direct logistic (0:22–0:30)

**Do:** `nba_wp/model.py` (`fit_direct_logistic`),
`docs/AUDIT_RESPONSE.md`.

**Say:**

> An earlier blend mixed two logistics in log-odds space. Algebraically that
> collapses to a linear logit in the same three features, while inviting a huge
> calibration grid. The remediated champion estimates that linear logit
> **directly** with L2 regularization.
>
> The old blend remains a **challenger** in the 72-candidate search. It did not
> win on pre-March mean log loss, so it was not promoted.

**Show:** formula on `START_HERE.md`.

## 6. Coefficients — fitted, not guessed (0:30–0:40)

**Do:** `artifacts/current/model_coefficients.json`.

**Say:**

> After locking the architecture, we fit sklearn logistic regression with
> standardization. I’ll read the standardized coefficients:
>
> - intercept ≈ 0.238
> - elo_diff ≈ 0.732
> - bt_logit ≈ 0.156
> - trend_diff ≈ 0.062
>
> Elo carries most of the signal; BT and trend are smaller but signed the way
> basketball intuition expects. Raw-unit coefficients are also in the JSON so
> you can reconstruct \(\operatorname{logit}\hat p\) from unscaled features.

**Show work:** leave coefficients on screen; point to training mean/scale.

---

# Act IV — Selection governance (0:40–0:58)

## 7. Pre-March folds and search budget (0:40–0:50)

**Do:** `configs/architecture_candidates.json`,
`scripts/select_model.py`,
`artifacts/current/pre_march_selection_proof.json`,
`artifacts/current/pre_march_selection_results.csv`.

**Say:**

> Selection uses only games before 1 March 2026. Two expanding folds:
> train→January validate; train→February validate. Objective: minimize mean
> validation log loss.
>
> Search size: 3×3×7 = **63** direct logistics, plus **nine** architecture-matched
> blend challengers → **72** total — not hundreds of thousands of temperature
> settings. Proof JSON shows zero March and zero April rows in selection.
>
> Winner: `k10_hl20`, \(C=0.1\), direct logistic.

**Honesty line:**

> This is a **reconstructed governance path**. April was viewed earlier in the
> project, so I do not claim historical preregistration — only that the
> executable pipeline now excludes March/April from selection.

## 8. Locked March and frozen April (0:50–0:58)

**Do:** `artifacts/current/final_metrics.json`,
`outputs/april_predictions_frozen_snapshot.csv`,
`outputs/april_predictions.csv`.

**Say:**

> March is locked after the spec freezes: **182 / 239** correct. I do not call
> tiny AUC/accuracy gaps versus retrospective reference floats a “win.”
>
> Primary April result freezes performance state at March 31 — matching the
> assignment’s “use October through March.” Sequential daily April scoring is
> kept only as sensitivity; it is **not** the headline assignment result.
>
> Frozen April: log loss ≈ 0.469, Brier ≈ 0.151, AUC ≈ 0.862, **74 / 96** correct.

---

# Act V — Uncertainty, calibration, claims (0:58–1:15)

## 9. Date-block bootstrap and calibration (0:58–1:08)

**Do:** `artifacts/current/date_block_bootstrap_summary.json`,
`artifacts/current/calibration_diagnostics.json`,
`figures/april_calibration.png`.

**Say:**

> Uncertainty uses **paired date-block** bootstrap on frozen April — games on
> the same date move together. Intervals **condition on the locked
> specification**; I did not re-run the 72-candidate search inside each
> replicate.
>
> Important comparison claim: date-block differences versus Elo are small and
> include zero. So: the direct model **won the declared pre-March process**,
> but its incremental April value over Elo remains **statistically unresolved**.
>
> Calibration intercept/slope/ECE are diagnostics, not proof that calibration
> is “solved.” Slope above one can mean probabilities are too compressed toward
> 0.5.

## 10. Market language and production claim (1:08–1:15)

**Do:** `docs/MARKET_PRICING_LIMITATIONS.md`, JD PDF in `docs/assignment/`.

**Say:**

> Fair odds equal one over probability. That is a zero-margin transform — not a
> traded price. Without time-stamped bookmaker odds I cannot claim CLV, edge, or
> profitability. The role is about pricing algorithms and trader collaboration;
> this dataset supports a forecasting prototype, not a betting P&L study.
>
> Readiness claim: **research / interview prototype**, not a deployable
> sportsbook system.

---

# Act VI — Live demo and close (1:15–1:30)

## 11. Price one April game live (1:15–1:22)

**Do:** `outputs/april_predictions_frozen_snapshot.csv` — pick one row.

**Say:**

> Here is game_id … home … away … \(\hat p=\) … fair decimal odds \(1/p=\).
> Features for that row are in `engineered_features.csv` under the same
> `game_id`. I can recompute log loss for the April slice from the CSV with the
> validator if you want a live check.

**Do (optional):**

```bash
python3 validate_submission.py --root . --data data/nba-win-probability-data.csv --recompute
```

## 12. Close (1:22–1:30)

**Do:** `docs/LIMITATIONS_AND_ROADMAP.md`, `docs/AUDIT_RESPONSE.md`.

**Say:**

> Three limitations I will not disguise: April is retrospective, not pristine;
> no market prices means no alpha claim; one season is not deployment evidence.
>
> What I will defend: leakage-controlled features, pre-March selection with a
> small declared search, locked March test, frozen March 31 April scoring,
> checkable coefficients, and honest uncertainty.
>
> Happy to go deeper on Elo updates, BT fitting, or any coefficient.

---

# Appendix — quick parameter table

| Number | Class | Value | File |
|---|---|---|---|
| Elo start / scale | Fixed | 1500 / 400 | `features.py` |
| Elo \(K\) | Selected | 10 | `selected_spec_pre_march.json` |
| HFA | Fixed in search | 65 | `architecture_candidates.json` |
| Half-life | Selected | 20d | same |
| Logistic \(C\) | Selected | 0.1 | same |
| \(\hat\beta\)s | Fitted | see ledger | `model_coefficients.json` |

Full detail: `docs/interview/PARAMETER_LEDGER.md`.
