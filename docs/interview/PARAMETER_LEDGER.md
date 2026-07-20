# Parameter ledger — every number in the model

**Rule for the interview:** every number is **Fitted**, **Selected**, or **Fixed by design**.  
If you cannot point to a file, do not invent a story.

Selected model: **Elo-only L2 logistic** — architecture `k5_hfa80_hl20`
(half-life irrelevant for a trend-free set), feature set `elo`,
`logistic_c = 0.03`. Winner of a **672-candidate prequential search**
(K × HFA × half-life × 5 nested feature sets × C) judged on pooled per-game
log loss over all 399 January–February games. Richer feature sets were in the
declared ladder and lost.

> Historical note: earlier iterations selected a three-feature logistic
> (`k10_hl20`, C=0.1) under a two-fold estimator. The prequential estimator
> (~200× more validation points than fold means) supersedes it; the
> three-feature model remains a documented challenger in
> `reports/metrics.json`. Sections below marked (superseded) describe it.

---

## 1. Fixed by design (conventions)

| Symbol / name | Value | Why | Code |
|---|---|---|---|
| Elo initial rating | 1500 | Standard Elo centre | `src/nba_wp/features.py` (`EloState`) |
| Elo logistic scale | 400 | Classic Elo denominator | `expected_score` / `elo_diff` |
| Margin-of-victory transform | \(\log(\lvert m\rvert+1)\) scaled | Smooth MOV; selected mode `log` | `features.py` MOV helper |
| Bradley-Terry ridge-ish \(C\) default | 0.15 | Stabilizes BT MLE | `configs/model.yaml` → `bt_c` |
| Trend short window | 10 games | Short form contrast | same → `trend_short_games` |
| Elo HFA (search default) | 65 | ≈ equal-team home edge under Elo scale | same → `elo_hfa` |
| Component logistic \(C\) (blend path) | elo 1.0 / rank 0.1 | Only used if blend challenger wins | same |
| Selection metric | mean validation log loss | Pricing proper score | `configs/model.yaml` |
| Selection cutoff | `< 2026-03-01` | March locked test | `src/nba_wp/selection.py` `SELECTION_CUTOFF` |
| Primary April policy | frozen at 2026-03-31 | Literal assignment | `score_final` / `reporting.py` |

---

## 2. Selected on pre-March folds (Jan / Feb)

Search grid (`configs/model.yaml`):

- \(K \in \{10,20,30\}\)
- half-life \(\in \{20,45,90\}\) days
- logistic \(C \in \{0.01,0.03,0.1,0.3,1,3,10\}\)
- + 9 blend challengers → **72** candidates

Winner (`artifacts/current/selected_spec_pre_march.json`):

| Parameter | Selected value | Evidence |
|---|---|---|
| Elo \(K\) | **10** | `architecture.elo_k` |
| Trend half-life | **20** days | `architecture.trend_half_life_days` |
| Logistic \(C\) | **0.1** | `logistic_c` |
| Model type | **direct_logistic** | `model_type` |
| Pre-March mean LL | 0.6319 | `pre_march_validation_metrics` |

Proof of no March/April in selection:

`artifacts/current/pre_march_selection_proof.json`

Fold results:

`artifacts/current/pre_march_fold_results.csv`  
`artifacts/current/pre_march_selection_results.csv`

---

## 3. Fitted from data (coefficients)

Final fit for April scoring trains on games through March 31 with selected architecture.

Standardized logistic (sklearn `StandardScaler` + `LogisticRegression`):

\[
\operatorname{logit}\hat p
=
\hat\beta_0
+
\hat\beta_{\mathrm{Elo}}\,z(\Delta\mathrm{Elo})
+
\hat\beta_{\mathrm{BT}}\,z(\Delta\mathrm{BT})
+
\hat\beta_{\mathrm{Tr}}\,z(\Delta\mathrm{Trend})
\]

| Feature | \(\hat\beta\) (standardized) | Training mean | Training scale | Raw-unit \(\hat\beta/\mathrm{scale}\) |
|---|---|---|---|---|
| intercept | 0.23848778829614511 | — | — | — |
| elo_diff | 0.7320943728053962 | 0.16685920352598477 | 0.3390078610479993 | 2.1595203442841133 |
| bt_logit | 0.15601597382186272 | 0.25373418704799144 | 0.5323095844808791 | 0.29309255059537054 |
| trend_diff | 0.06168440358453506 | −0.026387827320736235 | 3.563245223639699 | 0.017311299030249493 |

**Source file:** `artifacts/current/model_coefficients.json`  
**Also CSV:** `artifacts/current/coefficient_table.csv`

These numbers are **fitted**, not hand-tuned. Re-fit with `make score` after `make select`.

---

## 4. Feature definitions (how inputs are calculated)

### Elo difference

\[
\Delta\mathrm{Elo}_g
=
\frac{R_{\mathrm{home}}-R_{\mathrm{away}}+\mathrm{HFA}}{400}
\]

- Ratings update **after** features are written for that game (feature-before-update).
- Same calendar date: batch games, then update all.

**Code:** `src/nba_wp/features.py` Elo block.

### Bradley-Terry logit

Fit team strengths \(\theta\) on **prior** games only (regularized), then:

\[
\Delta\mathrm{BT}_g = \theta_{\mathrm{home}}-\theta_{\mathrm{away}}
\quad(+ \text{any intercept convention in code})
\]

Exposed as `bt_logit`. **Code:** `_fit_bradley_terry` in `features.py`.

### Trend difference

Short-window mean margin (last 10 games) minus EWMA of margins with half-life \(h\):

\[
\mathrm{trend}_i = \mathrm{short10}_i - \mathrm{EWMA}_{h}(\mathrm{margins}_i)
\]

\[
\Delta\mathrm{Trend}_g = \mathrm{trend}_{\mathrm{home}}-\mathrm{trend}_{\mathrm{away}}
\]

**Code:** `_ewma`, trend construction in `features.py`.

---

## 5. What was *not* selected for the champion

| Idea | Status | Why |
|---|---|---|
| Dense temperature/shift grid | Rejected | Monotonic in \(z\); does not create ranking models |
| Four-target March gate | Removed | Not in assignment; circular with reporting |
| Log-odds blend of Elo + rank logistics | Challenger only | Algebraically ≈ linear logit; lost pre-March LL race |
| Causal/identity continuous IDs | N/A here | Not in this NBA take-home feature set |

Blend challenger remains in search for comparison; it did **not** win.

---

## 6. Evaluation numbers (report, do not “select” from them)

| Period | Role | Key file |
|---|---|---|
| Jan / Feb | Selection folds | `pre_march_*` |
| March | Locked test (182/239 correct) | `final_metrics.json` → `locked_march_test` |
| April frozen | **Primary assignment** (74/96) | `april_predictions_frozen_snapshot.csv` |
| April sequential | Sensitivity only | `april_predictions.csv` |

Primary frozen April metrics (approx):

- Log loss ≈ 0.469  
- Brier ≈ 0.151  
- AUC ≈ 0.862  
- Accuracy 74/96  

Date-block uncertainty: `artifacts/current/date_block_bootstrap_summary.json`  
(Conditional on locked specification — say this out loud.)

---

## 7. Fair odds (transform only)

\[
\text{Fair decimal odds} = 1 / \hat p
\]

No overround, no book comparison. See `docs/MARKET_PRICING_LIMITATIONS.md`.
