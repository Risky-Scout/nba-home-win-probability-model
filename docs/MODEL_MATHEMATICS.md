---
title: "NBA Home-Win Probability Model — Mathematics"
author: "GitHub main · architecture hfa_75"
geometry: margin=0.85in
fontsize: 11pt
header-includes:
  - \usepackage{amsmath,amssymb}
  - \usepackage{booktabs}
  - \usepackage{enumitem}
  - \setlist{itemsep=0.2em,topsep=0.3em}
  - \setlength{\parskip}{0.45em}
  - \setlength{\parindent}{0pt}
---

# NBA Home-Win Probability Model

## Mathematical summary

**Champion family.** Logistic-stacked blend of Elo and Bradley–Terry with recent-form trend.

**Selected architecture.** `hfa_75`

| Symbol | Meaning | Selected value |
|:------:|:--------|:---------------|
| $H$ | Elo home-field advantage (rating points) | $75$ |
| $K$ | Elo learning rate | $10$ |
| — | Elo MOV mode | $\log$ |
| $C_{\mathrm{BT}}$ | Bradley–Terry $\ell_2$ inverse-regularization | $0.15$ |
| $h$ | Trend EWMA half-life (days) | $45$ |
| $n_S$ | Short-form window (games) | $10$ |
| $C_E$ | Elo calibrator inverse-regularization | $100$ |
| $C_R$ | Rank calibrator inverse-regularization | $0.1$ |

**Fitted stacker (March).** $a = 0.5796,\quad b = 0.9838,\quad c = 0.3154$.

---

## 1. Target

For game $g$ with home points $P_h$ and away points $P_a$,

$$
Y_g =
\begin{cases}
1 & \text{if } P_h > P_a,\\[0.25em]
0 & \text{otherwise}.
\end{cases}
$$

The model delivers a probability price
$$
\hat p_g \;=\; \widehat{\Pr}(Y_g = 1).
$$

---

## 2. Information timing (no leakage)

Let $\mathcal{H}_{<d}$ be all outcomes and box scores from dates **strictly before** game date $d$. For every game $g$ on date $d$,

$$
X_g \;=\; f\!\bigl(\mathcal{H}_{<d}\bigr).
$$

**Same-day protocol.** On date $d$: (1) refresh Bradley–Terry from prior games; (2) write feature rows for **all** games on $d$; (3) **then** update Elo / histories from results. No same-date result enters another same-date feature row.

---

## 3. Link functions

$$
\sigma(z) \;=\; \frac{1}{1+e^{-z}},
\qquad
\operatorname{logit}(p) \;=\; \log\!\frac{p}{1-p}.
$$

---

## 4. Elo component

### 4.1 Raw Elo probability

Each team starts at rating $R_i = 1500$. For home $h$ and away $a$,

$$
p^{\mathrm{Elo,raw}}_g
\;=\;
\frac{1}{1 + 10^{-(R_h - R_a + H)/400}}.
$$

Equal teams ($R_h = R_a$) with $H = 75$ give
$$
p^{\mathrm{Elo,raw}} \;=\; \frac{1}{1+10^{-75/400}} \;\approx\; 0.606.
$$

The scaled Elo feature used by the calibrator is
$$
x^{\mathrm{Elo}}_g \;=\; \frac{R_h - R_a + H}{400}.
$$

### 4.2 Rating update (after the date completes)

Home margin $m_g = P_h - P_a$. Margin-of-victory multiplier (log mode):

$$
M_g
\;=\;
\log\bigl(|m_g| + 1\bigr)
\cdot
\frac{2.2}{\max\!\bigl(0.25,\; 0.001\,(R_h - R_a) + 2.2\bigr)}.
$$

Updates:

\begin{align*}
R_h' &= R_h + K\, M_g \bigl(Y_g - p^{\mathrm{Elo,raw}}_g\bigr),\\[0.35em]
R_a' &= R_a - K\, M_g \bigl(Y_g - p^{\mathrm{Elo,raw}}_g\bigr),
\end{align*}

with $K = 10$.

### 4.3 Elo calibrator

An $\ell_2$-logistic regression (sklearn `LogisticRegression`, features standardized) maps $x^{\mathrm{Elo}}$ to a calibrated component probability $p_E$:

$$
p_E
\;=\;
\sigma\!\bigl(
\beta_{E,0} + \beta_{E,1}\, \widetilde{x}^{\mathrm{Elo}}
\bigr),
$$

where $\widetilde{x}^{\mathrm{Elo}}$ is the standardized version of $x^{\mathrm{Elo}}$, and the inverse-regularization strength is $C_E = 100$.

---

## 5. Bradley–Terry component

From all completed games before date $d$, fit team strengths $q_i$ and home intercept $\alpha$ by $\ell_2$-regularized logistic regression on design rows with $+1$ for home, $-1$ for away:

$$
\Pr(h \text{ beats } a)
\;=\;
\sigma\!\bigl(\alpha + q_h - q_a\bigr),
\qquad
C_{\mathrm{BT}} = 0.15.
$$

The matchup feature is the decision value
$$
x^{\mathrm{BT}}_g \;=\; \hat\alpha + \hat q_h - \hat q_a.
$$

---

## 6. Recent-form trend

For team $i$, let $m_{ik}$ be that team’s point margin in prior game $k$, and let $\Delta\mathrm{days}_{kd}$ be the age of that game in days at date $d$.

**Long form** (exponentially weighted, half-life $h = 45$ days):
$$
L_{i,d}
\;=\;
\frac{
\sum_{k < d} 2^{-\Delta\mathrm{days}_{kd}/h}\, m_{ik}
}{
\sum_{k < d} 2^{-\Delta\mathrm{days}_{kd}/h}
}.
$$

**Short form** (mean of the last $n_S = 10$ games, or fewer if unavailable):
$$
S_{i,d}
\;=\;
\frac{1}{n_i}\sum_{\text{last }n_i} m_{ik},
\qquad
n_i = \min(10,\, \#\{\text{prior games}\}).
$$

**Team trend and matchup feature:**
$$
T_{i,d} \;=\; S_{i,d} - L_{i,d},
\qquad
x^{\mathrm{trend}}_g \;=\; T_{h,d} - T_{a,d}.
$$

---

## 7. Rank calibrator (BT + trend)

A second $\ell_2$-logistic maps $(x^{\mathrm{BT}},\, x^{\mathrm{trend}})$ to the rank-component probability $p_R$:

$$
p_R
\;=\;
\sigma\!\bigl(
\beta_{R,0}
+
\beta_{R,1}\, \widetilde{x}^{\mathrm{BT}}
+
\beta_{R,2}\, \widetilde{x}^{\mathrm{trend}}
\bigr),
$$

with inverse-regularization $C_R = 0.1$ and standardized inputs.

---

## 8. Logistic stacking (final price)

Let $p_E$ and $p_R$ be the two component probabilities. The final model is a logistic regression on their logits, fitted by penalized maximum likelihood on **March** component logits:

$$
\hat p_g
\;=\;
\sigma\!\bigl(
a\,\operatorname{logit}(p_E)
+
b\,\operatorname{logit}(p_R)
+
c
\bigr).
$$

**March fit (architecture `hfa_75`):**
$$
a = 0.5796,\qquad
b = 0.9838,\qquad
c = 0.3154.
$$

**Equivalent $(w,\tau,s)$ form** (same mapping):
\begin{align*}
w &= \frac{a}{a+b} \approx 0.371,\\[0.25em]
\tau &= \frac{1}{a+b} \approx 0.640,\\[0.25em]
s &= c \approx 0.315,
\end{align*}
$$
\hat p_g
\;=\;
\sigma\!\Bigl(
\frac{w\,\operatorname{logit}(p_E) + (1-w)\,\operatorname{logit}(p_R)}{\tau}
+ s
\Bigr).
$$

---

## 9. Selection rule

Among architecture candidates, choose the one that **minimizes March sequential log loss**. April rows are **not** used in selection ($0$ April rows loaded; selection data max date $2026$-$03$-$31$).

Base component coefficients for March scoring are trained on games through February; March state updates sequentially by date. Final April scoring refits base coefficients through March under the locked architecture and stacker policy documented in `artifacts/selected_spec.json`.

---

## 10. Evaluation metrics

**Log loss**
$$
\mathrm{LL}
\;=\;
-\frac{1}{N}\sum_g
\Bigl[
Y_g\log \hat p_g
+
(1-Y_g)\log(1-\hat p_g)
\Bigr].
$$

**Brier score**
$$
\mathrm{BS}
\;=\;
\frac{1}{N}\sum_g \bigl(\hat p_g - Y_g\bigr)^2.
$$

**Accuracy** uses the fixed threshold $\hat p_g \ge 0.5$. AUC is reported for ranking.

**Headline sequential-daily results (committed artifacts):**

| Period | Log loss | Accuracy @ $0.5$ |
|:-------|:--------:|:----------------:|
| March (selection) | $\approx 0.488$ | $188/239$ ($\approx 78.7\%$) |
| April (holdout) | $\approx 0.459$ | $79/96$ ($\approx 82.3\%$) |

---

## 11. Fair decimal odds

Zero-margin (no overround) fair odds implied by the price:
$$
O_{\mathrm{home}} \;=\; \frac{1}{\hat p_g},
\qquad
O_{\mathrm{away}} \;=\; \frac{1}{1-\hat p_g}.
$$

These are mathematical fair odds, not a production sportsbook quote.

---

## 12. End-to-end map

$$
\begin{aligned}
&\text{CSV}
\;\xrightarrow{\;\text{audit}\;}
X_g = \bigl(x^{\mathrm{Elo}},\, x^{\mathrm{BT}},\, x^{\mathrm{trend}}\bigr)\\[0.4em]
&\xrightarrow{\;\text{calibrators}\;}
(p_E,\, p_R)
\xrightarrow{\;\text{stacker}\;}
\hat p_g = \sigma\!\bigl(a\,\operatorname{logit}(p_E)+b\,\operatorname{logit}(p_R)+c\bigr).
\end{aligned}
$$

**Code anchors.** Features: `nba_wp/features.py`. Calibrators & stacker: `nba_wp/model.py`. Selection: `nba_wp/selection.py`. Locked numbers: `artifacts/selected_spec.json`.
