---
title: "NBA Home-Win Probability Model — Mathematics"
subtitle: "Deployed champion: Elo-only (architecture `conservative`)"
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

**Champion form (deployed).** The deployed champion is **Elo-only**: a single
L2-logistic map on the standardized margin-of-victory Elo rating differential,
\[
\boxed{
p_g
=
\sigma\!\bigl(c + w\,\tilde x_g^{E}\bigr)
},
\qquad
\tilde x_g^{E} = \frac{x_g^{E}-\mu}{s},
\]
with **deployed** coefficients (fit on all games through March 31)
\[
w = 0.9272,\qquad
c = 0.2415,\qquad
\mu = 0.1408,\quad s = 0.2690
\]
(raw-unit weight $w/s \approx 3.446$ per unit of $x^{E}$). This is the model in
§3.

**Rejected challenger.** An Elo $+$ Bradley–Terry/recent-trend **logistic
stacker** (§7) was implemented and then rejected: under honest nested
rolling-origin validation it does not beat Elo-only out-of-sample on log loss or
Brier, and it is worse calibrated. It is retained only as a challenger in the
`challenger` block of `artifacts/selected_spec.json`. See §7–§9.

**Selected hyperparameters** (`conservative`, Elo-only):
$K=7.5$, $H=55$, MOV $=$ log (winner$-$loser), Elo-logistic $C_E=10$.
The challenger additionally uses BT regularization $C_{\mathrm{BT}}=0.1$,
trend half-life $60$ days, short window $12$ games, rank-logistic $C_R=0.1$.

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
\qquad H = 55.
\]
Equal teams ($R_h=R_a$) imply
\[
p^{\mathrm{home}}
=
\frac{1}{1+10^{-55/400}}
\approx 0.578.
\]

The Elo feature used by the calibrated logistic is the scaled rating gap:
\[
x_g^{E}
=
\frac{R_h - R_a + H}{400}.
\]

## 3.2 Margin-of-victory update

After date $d$ is complete, with home margin $m_g$ (home points $-$ away points)
and $K=7.5$:
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
$C_E = 10$. **This is the deployed champion**: fit on all games through
March 31, $\beta_0^{E}=c=0.2415$ and $\beta_1^{E}=w=0.9272$ (standardized;
raw-unit $w/s\approx 3.446$), so $p_g = p_g^{E}$. April performance state is
frozen at March 31.

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

# 7. Logistic stack (rejected challenger)

The challenger combines the two component logits with a convex logistic stacker,
fit on **inner out-of-fold** component probabilities and deployed under a
**temperature floor** $\tau\ge 1$ (a genuine convex blend: $a,b\ge 0$,
$a+b\le 1$, so it never sharpens):
\[
p_g^{\mathrm{blend}}
=
\sigma\!\Bigl(
a\,\operatorname{logit}(p_g^{E})
+
b\,\operatorname{logit}(p_g^{R})
+
c
\Bigr),
\qquad a,b\ge 0,\; a+b\le 1.
\]
For architecture `conservative` the challenger weights are $a\approx0.398$,
$b\approx0.602$ ($a+b=1$, $\tau=1$); exact coefficients are in the `challenger`
block of `artifacts/selected_spec.json`. Convexity is pinned by
`tests/test_stacker_temperature_floor.py::test_stacker_weights_are_convex_when_stacker_is_used`.

**This blend is not deployed.** Under nested rolling-origin validation (§8) it is
worse than Elo-only on both proper scores and worse calibrated
(slope $\beta\approx1.8$ vs. Elo-only $\beta\approx1.35$), so the champion is the
Elo-only model of §3.

---

# 8. Selection and honest validation

**Deployed selection (April-blind).** Each procedure (Elo-only, rank-only,
blend) selects its **own** architecture by its **own** March log loss (Brier
tie-break). April rows are **not** used
($N_{\mathrm{April}}^{\mathrm{selection}}=0$, max selection date
$2026$-$03$-$31$). The Elo-only winner is **`conservative`**; its probability map
is then refit on all rows through March 31. Because March is used to select, it
is in-sample for selection and is not a pristine holdout.

**Nested rolling-origin audit** (`scripts/nested_validation.py`, 11 weekly outer
folds, 501 out-of-sample games) validates the whole procedure under two
information policies — *frozen-block* (state frozen at each outer origin) and
*daily-sequential* (results strictly before date $t$). Each procedure picks its
own architecture by its own inner OOF score; the stacker trains on inner
out-of-fold predictions. Pooled out-of-sample:
\[
\begin{array}{lcc}
\text{candidate} & \mathrm{LL} & \mathrm{Brier}\\\hline
\text{Elo-only (champion)} & 0.532 & 0.177\\
\text{rank-only} & 0.550 & 0.184\\
\text{blend} & 0.548 & 0.183\\
\text{constant} & 0.688 & 0.247
\end{array}
\]
Block-bootstrap blend$-$Elo: $\Delta\mathrm{LL}=+0.017$ (95% CI
$[+0.010,+0.023]$), $\Delta\mathrm{Brier}=+0.006$ (95% CI $[+0.004,+0.009]$);
**0 of 4{,}000** week-block replicates favored the blend. Champion–challenger
decision under both policies: **keep Elo-only**. Elo-only calibration:
$\alpha\approx-0.05$, $\beta\approx1.37$ (95% CI $[1.22,1.57]$), ECE $\approx
0.059$.

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

Champion results (Elo-only). **Primary April holdout (frozen pre-April):**
$\mathrm{LL}=0.4644$, Brier $=0.1498$, AUC $=0.8668$, accuracy $78.13\%$
($75/96$). March selection period (Elo-only, base fit through February):
$\mathrm{LL}=0.5066$. Optional April sequential backtest (live-update
simulation): $\mathrm{LL}=0.4646$. The rejected blend on the same frozen April
window scores $\mathrm{LL}=0.4687$, Brier $=0.1505$ — worse on both proper
scores.

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

Deployed champion path (Elo-only):
\[
(R_h,R_a,H)
\;\rightarrow\;
x^{E}
\;\xrightarrow{\;c,\,w\;}\;
p^{E} = p_g .
\]
Rejected challenger path (blend), for reference:
\[
\begin{aligned}
&(q,\alpha)
\;\rightarrow\;
x^{\mathrm{BT}},\quad
(S,L)
\;\rightarrow\;
x^{\mathrm{trend}}
\\
&(x^{\mathrm{BT}},x^{\mathrm{trend}})
\;\rightarrow\;
p^{R},\quad
(p^{E},p^{R})
\;\xrightarrow{\;a,\,b,\,c\;}
p^{\mathrm{blend}} .
\end{aligned}
\]

**Sources in code:** `nba_wp/features.py`, `nba_wp/model.py`, `nba_wp/selection.py`;
fitted values in `artifacts/selected_spec.json`.
