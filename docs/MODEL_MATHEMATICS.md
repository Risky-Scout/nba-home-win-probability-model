---
title: "NBA Home-Win Probability Model — Mathematics"
subtitle: "Champion architecture `hfa_75` (GitHub main)"
author: "Joseph Shackelford"
geometry: margin=0.85in
fontsize: 11pt
header-includes:
  - \usepackage{amsmath,amssymb,booktabs}
  - \usepackage{enumitem}
  - \setlist{nosep,leftmargin=1.2em}
  - \setlength{\parskip}{0.45em}
  - \setlength{\parindent}{0pt}
---

# Overview

The deliverable is a **price**: for each game $g$, estimate
\[
p_g \;=\; P(Y_g = 1 \mid \text{pregame information}),
\]
where $Y_g = 1$ if the home team wins and $0$ otherwise.

**Champion form.** Two calibrated component probabilities (Elo and Bradley–Terry + recent trend) are combined by a logistic stacker:
\[
\boxed{
p_g
=
\sigma\!\Bigl(
a\,\operatorname{logit}(p_g^{E})
+
b\,\operatorname{logit}(p_g^{R})
+
c
\Bigr)
}
\]
with **deployed** stacker coefficients (temperature-floored to $\tau\ge 1$)
\[
a = 0.3593,\qquad
b = 0.6407,\qquad
c = 0.3254,
\]
where $a+b=1$. The unconstrained MLE fit was $a=0.5696$, $b=1.0159$,
$c=0.3132$ ($\tau=1/(a+b)\approx 0.63<1$); it is kept for audit only. See §7.

**Selected hyperparameters** (`hfa_75`):
$K=10$, $H=75$, MOV $=$ log, BT regularization $C_{\mathrm{BT}}=0.15$,
trend half-life $45$ days, short window $10$ games,
Elo-logistic $C_E=100$, rank-logistic $C_R=0.1$.

**Notation.**
$\sigma(z)=1/(1+e^{-z})$,
$\operatorname{logit}(p)=\log\!\bigl(p/(1-p)\bigr)$.

---

# 1. Target

\[
Y_g
=
\begin{cases}
1, & \text{home points}_g > \text{away points}_g,\\[0.25em]
0, & \text{otherwise}.
\end{cases}
\]

---

# 2. Information / no leakage

For game date $d$, every feature for games on $d$ is a function of history
strictly before $d$:
\[
X_g = f(\mathcal{H}_{<d}).
\]
All same-date rows are written **before** any result on $d$ updates ratings,
Bradley–Terry strengths, or trend histories.

---

# 3. Elo component

## 3.1 Raw Elo probability

Each team starts at $R_i = 1500$. For home $h$ and away $a$,
\[
p_g^{\mathrm{Elo,raw}}
=
\frac{1}{1 + 10^{-(R_h - R_a + H)/400}},
\qquad H = 75.
\]
Equal teams ($R_h=R_a$) imply
\[
p^{\mathrm{home}}
=
\frac{1}{1+10^{-75/400}}
\approx 0.606.
\]

The Elo feature used by the calibrated logistic is the scaled rating gap:
\[
x_g^{E}
=
\frac{R_h - R_a + H}{400}.
\]

## 3.2 Margin-of-victory update

After date $d$ is complete, with home margin $m_g$ (home points $-$ away points)
and $K=10$:
\[
R_h' = R_h + K\, M_g\, (Y_g - p_g^{\mathrm{Elo,raw}}),
\qquad
R_a' = R_a - K\, M_g\, (Y_g - p_g^{\mathrm{Elo,raw}}),
\]
where the MOV multiplier uses the **winner-minus-loser** rating difference:
\[
M_g
=
\log\!\big(|m_g|+1\big)
\cdot
\frac{2.2}{\max\!\bigl(0.25,\; 0.001(R_{\mathrm{win}}-R_{\mathrm{lose}})+2.2\bigr)}.
\]
Here $R_{\mathrm{win}},R_{\mathrm{lose}}$ are the pregame ratings of the winning
and losing team (matching the FiveThirtyEight form). This makes an upset move
ratings *more* than an expected result and keeps the update team-swap symmetric
at zero home advantage. (The $\max$ guard is in code; ratings exclude $H$.)

## 3.3 Calibrated Elo probability

An L2-logistic model (with feature standardization) maps $x_g^{E}$ to
\[
p_g^{E}
=
\sigma\!\bigl(\beta_0^{E} + \beta_1^{E}\, \tilde x_g^{E}\bigr),
\]
where $\tilde x$ is the standardized feature and regularization uses
$C_E = 100$.

---

# 4. Bradley–Terry component

From all prior games (before date $d$), fit team strengths $q_i$ and home intercept $\alpha$ by L2-logistic regression on design rows
\[
\mathbf{x}_g^{\mathrm{BT}}
=
\mathbf{e}_h - \mathbf{e}_a
\quad\text{(home $+1$, away $-1$)},
\]
\[
P(Y_g=1)
=
\sigma\!\bigl(\alpha + q_h - q_a\bigr),
\qquad C_{\mathrm{BT}} = 0.15.
\]
The pregame feature is the decision value
\[
x_g^{\mathrm{BT}}
=
\hat\alpha + \hat q_h - \hat q_a.
\]

---

# 5. Recent-form trend

For team $i$ on date $d$, let $m_{ik}$ be that team’s point margin in prior game $k$,
and $\Delta_{kd}$ the age of that game in days.

**Long form** (exponentially weighted, half-life $45$ days):
\[
L_{i,d}
=
\frac{
\sum_{k:\,\mathrm{date}_k < d}
2^{-\Delta_{kd}/45}\, m_{ik}
}{
\sum_{k:\,\mathrm{date}_k < d}
2^{-\Delta_{kd}/45}
}.
\]

**Short form** (mean of the last $\min(10,n_i)$ games):
\[
S_{i,d}
=
\frac{1}{n_i^{\mathrm{short}}}
\sum_{\text{last } n_i^{\mathrm{short}}} m_{ik},
\qquad
n_i^{\mathrm{short}}=\min(10,n_i).
\]

**Trend and matchup feature:**
\[
T_{i,d} = S_{i,d} - L_{i,d},
\qquad
x_g^{\mathrm{trend}} = T_{h,d} - T_{a,d}.
\]

---

# 6. Rank (BT + trend) component

A second L2-logistic model maps $(x_g^{\mathrm{BT}},\, x_g^{\mathrm{trend}})$ to
\[
p_g^{R}
=
\sigma\!\bigl(
\beta_0^{R}
+
\beta_{\mathrm{BT}}\, \tilde x_g^{\mathrm{BT}}
+
\beta_{\mathrm{tr}}\, \tilde x_g^{\mathrm{trend}}
\bigr),
\qquad C_R = 0.1.
\]

---

# 7. Logistic stack (final price)

Component logits are blended by a third logistic regression fitted on **March**
one-step-ahead component probabilities (penalized MLE, $C=1$), then projected
onto a **temperature floor** $\tau\ge 1$ so the deployed blend never sharpens:
\[
\boxed{
p_g
=
\sigma\!\Bigl(
a\,\operatorname{logit}(p_g^{E})
+
b\,\operatorname{logit}(p_g^{R})
+
c
\Bigr)
}
\]
\[
a = 0.3593,\quad
b = 0.6407,\quad
c = 0.3254 \qquad (\text{deploy}, \; a+b=1).
\]

The two component logits are near-duplicates ($\rho\approx 0.97$), so the
unconstrained MLE learns $a+b=1.586>1$, i.e. $\tau=1/(a+b)\approx 0.63<1$, which
sharpens the blend toward $0$/$1$ and produced extreme out-of-sample prices.
Deployment preserves the Elo weight $w=a/(a+b)=0.3593$, sets $a+b=1$
($a=w$, $b=1-w$), and refits the intercept to $c=0.3254$. Both sets of
coefficients are stored in `artifacts/selected_spec.json`; the floor is pinned
by `tests/test_stacker_temperature_floor.py`.

**Equivalent $(w,\tau,s)$ form** (same mapping, deploy values):
\[
w = \frac{a}{a+b} = 0.3593,\qquad
\tau = \frac{1}{a+b} = 1.000,\qquad
s = c = 0.3254,
\]
\[
p_g
=
\sigma\!\Biggl(
\frac{
w\,\operatorname{logit}(p_g^{E})
+
(1-w)\,\operatorname{logit}(p_g^{R})
}{\tau}
+
s
\Biggr).
\]

---

# 8. Selection rule

Architectures are scored by **March** sequential (daily) log loss using the
**unconstrained** stacker fit. April rows are **not** used in selection
($N_{\mathrm{April}}^{\mathrm{selection}}=0$, max selection date $2026$-$03$-$31$).
Chosen architecture: **`hfa_75`**. The temperature floor is applied only to the
deployed model, after selection. Because March is used both to pick the
architecture and to fit the stacker, March metrics are **in-sample for the
blend** and are not an unbiased holdout.

---

# 9. Scoring metrics

**Log loss** (primary):
\[
\mathrm{LL}
=
-\frac{1}{N}\sum_{g=1}^{N}
\Bigl[
Y_g\log p_g
+
(1-Y_g)\log(1-p_g)
\Bigr].
\]

**Brier score:**
\[
\mathrm{BS}
=
\frac{1}{N}\sum_{g=1}^{N}(p_g - Y_g)^2.
\]

**Accuracy** uses the fixed cutoff $p_g \ge 1/2$. AUC ranks games by $p_g$.

Champion results. **Primary April holdout (frozen pre-April):**
$\mathrm{LL}=0.4844$, Brier $=0.1558$, AUC $=0.8628$, accuracy $81.25\%$
($78/96$). March selection surface (unconstrained stacker): $\mathrm{LL}=0.4880$;
the deployed (floored) stacker scores March $\mathrm{LL}=0.5084$. Optional April
sequential backtest (live-update simulation): $\mathrm{LL}=0.4745$.

---

# 10. Fair odds (zero-margin)

\[
O_g^{\mathrm{home}} = \frac{1}{p_g},
\qquad
O_g^{\mathrm{away}} = \frac{1}{1-p_g}.
\]
These are mathematical fair decimal odds (no overround).

---

# 11. End-to-end map

\[
\begin{aligned}
&(R_h,R_a,H)
\;\rightarrow\;
x^{E}
\;\rightarrow\;
p^{E}
\\
&(q,\alpha)
\;\rightarrow\;
x^{\mathrm{BT}}
\\
&(S,L)
\;\rightarrow\;
x^{\mathrm{trend}}
\\
&(x^{\mathrm{BT}},x^{\mathrm{trend}})
\;\rightarrow\;
p^{R}
\\
&(p^{E},p^{R})
\;\xrightarrow{\;a,\,b,\,c\;}
p
\end{aligned}
\]

**Sources in code:** `nba_wp/features.py`, `nba_wp/model.py`, `nba_wp/selection.py`;
fitted values in `artifacts/selected_spec.json`.
