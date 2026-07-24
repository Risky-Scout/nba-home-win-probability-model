# Methodology

## 1. Target

For game \(g\),

\[
Y_g =
\begin{cases}
1, & \text{home points} > \text{away points},\\
0, & \text{otherwise}.
\end{cases}
\]

The submitted quantity is \(P(Y_g=1)\), not merely a binary winner.

## 2. Feature timing

For every game date \(d\), the feature engine performs a read-before-update
operation:

\[
X_g = f(\mathcal{H}_{<d}),
\]

where \(\mathcal{H}_{<d}\) contains outcomes and box scores only from dates
strictly before \(d\). Every game on date \(d\) is assigned a feature row before
any result on \(d\) updates state.

This is why same-game points and other box-score outcomes can be used to update
future team state without leaking into their own game.

## 3. Elo component

Each team begins at 1,500. For home team \(h\) and away team \(a\),

\[
p^{Elo}_g =
\frac{1}{
1 + 10^{-(R_h-R_a+H)/400}
}.
\]

The deployed (`conservative`) architecture uses \(H=55\) Elo points. After the
date is complete,

\[
R_h' = R_h + K M_g (Y_g-p^{Elo}_g),
\]

\[
R_a' = R_a - K M_g (Y_g-p^{Elo}_g),
\]

where \(K=7.5\) and the margin-of-victory multiplier follows the
FiveThirtyEight log form, evaluated on the **winner-minus-loser** rating
difference:

\[
M_g =
\log(|m_g|+1)
\frac{2.2}{
0.001\,(R_{\text{win}}-R_{\text{lose}})+2.2
},
\]

with \(m_g\) the home point margin, and \(R_{\text{win}}, R_{\text{lose}}\) the
pregame ratings of the winning and losing team. Using the winner's rating
difference (rather than a fixed home-minus-away difference) is what makes an
upset — a low-rated team beating a high-rated team — move the ratings *more*
than an expected result, and keeps the update team-swap symmetric at zero home
advantage. The denominator is guarded in code against pathological values, and
the behaviour is pinned by `tests/test_elo_mov_winner_diff.py`.

The multiplier offset (the `2.2`) and slope (`0.001`) are **data-driven** rather
than merely borrowed from FiveThirtyEight: the offset is profiled on the
frozen-policy rolling out-of-sample surface over the grid
\(\{1.6, 2.0, 2.2, 2.6, 3.0\}\), and `2.2` is kept because it is within one
standard error of the nominal best (`3.0`), so it is now empirically confirmed
on our data. Both `mov_offset` and `mov_slope` are exposed as tunable fields;
see `mov_offset_selection` in `artifacts/selected_spec.json`.

An Elo cold-start refinement — a provisional-K warmup for a team's first few
games — was also implemented and profiled on the same surface, but kept **off**
(`warmup_games = 0`) because every warmup configuration worsened out-of-sample
log loss (`cold_start_selection` in `selected_spec.json`). A same-season record
prior was intentionally *not* added, being redundant with Elo, which already
incorporates same-season results.

The raw Elo feature (called `elo_diff`) is

\[
\text{elo\_diff}_g = \frac{R_h-R_a+H}{400}.
\]

## 4. Deployed champion — Elo-only logistic

**The deployed model is Elo-only.** It is a single logistic map on the
standardized Elo rating differential — there is no Bradley-Terry component, no
recent-trend component, and no stacker or temperature floor in the deployed
price:

\[
p^{home}_g = \sigma\!\bigl(c + w\, z(\text{elo\_diff}_g)\bigr),
\qquad
z(\text{elo\_diff}) = \frac{\text{elo\_diff} - \mu}{s},
\]

where \(z\) standardizes `elo_diff` by the training mean \(\mu\) and scale \(s\).
The coefficients are fitted by an L2-logistic model (\(C=10\)) on **all games
through March 31**; the April performance state is then frozen at March 31. The
deployed values are

\[
w = 0.9271823510,\quad
c = 0.2415428879,\quad
\mu = 0.1408146877,\quad
s = 0.2690252334,
\]

equivalent to a raw-unit weight of \(3.4464512465\) per unit of `elo_diff`
(\(w/s\)). All values are stored in `artifacts/selected_spec.json` under
`elo_model`, and the primary April prices recompute from them (pinned by
`tests/test_champion_promotion.py`). The raw form is also exposed directly:
alongside `raw_unit_coefficient` the spec now carries `raw_intercept`
\(= c - w\mu/s = -0.2437680682\), the correct raw-space intercept. When pricing
from the raw-unit coefficient, pair it with `raw_intercept`, **not** with the
standardized `intercept` \(c\) — `raw_intercept` already absorbs the centering
shift (pinned by `tests/test_workbook_reconstruction.py`).

Elo-only is the champion because the nested rolling-origin audit (see
`docs/VALIDATION_AND_GOVERNANCE.md`) rejects the Elo + rank blend: the blend
does not beat Elo-only out-of-sample on either log loss or Brier. The blend is
retained only as a clearly-labelled **rejected challenger**, described in
Sections 5–7 below.

## 5. Rejected challenger — Bradley-Terry component

The following three sections describe the **rejected challenger blend**, which
is implemented and validated but is *not* deployed. For each date, team
strengths are fitted from earlier game outcomes:

\[
P(h \text{ beats } a)
=
\sigma(
\alpha + q_h-q_a
),
\]

where \(\alpha\) is the home intercept and \(q_i\) is team \(i\)'s fitted
strength. L2 regularization controls the 30 team coefficients. The challenger
architecture uses `C = 0.1`.

The model feature is the Bradley-Terry decision value:

\[
x^{BT}_g = \hat \alpha + \hat q_h-\hat q_a.
\]

## 6. Rejected challenger — recent trend

For team \(i\), define point margin \(m_{ik}\) in each prior game from the
team's perspective. Long-form margin is exponentially weighted:

\[
L_{i,d}
=
\frac{
\sum_{k<d}
2^{-\Delta days_{kd}/60} m_{ik}
}{
\sum_{k<d}
2^{-\Delta days_{kd}/60}
}.
\]

Short-form margin is the mean over the most recent twelve games:

\[
S_{i,d}
=
\frac{1}{n_i}
\sum_{\text{last } \min(12,n_i)} m_{ik}.
\]

Trend is the change relative to long-form strength:

\[
T_{i,d}=S_{i,d}-L_{i,d}.
\]

The matchup feature is

\[
x^{trend}_g=T_{h,d}-T_{a,d}.
\]

A second L2-logistic model maps
\((x^{BT}_g, x^{trend}_g)\) to a rank-component probability.

## 7. Rejected challenger — log-odds blend (logistic stacking)

The stacker and temperature floor below belong **only to the rejected
challenger blend**. They are *not* part of the deployed champion (Section 4),
which is Elo-only and applies no stacking or temperature floor.

Let \(p_E\) and \(p_R\) be the two component probabilities. The blend is a
logistic regression on their logits, fitted by penalized maximum likelihood:

\[
p_g =
\sigma\bigl(
a\operatorname{logit}(p_E)
+
b\operatorname{logit}(p_R)
+
c
\bigr).
\]

The unconstrained March fit for the challenger produced:

\[
a=0.6076,\qquad
b=0.9204,\qquad
c=0.3279,
\]

equivalent to \((w, \tau, s)\) via \(w=a/(a+b)\), \(\tau=1/(a+b)\),
\(s=c\). Because the two component logits are near-duplicates
(\(\rho\approx 0.97\)), the unconstrained fit learns \(a+b>1\), i.e. temperature
\(\tau<1\), which **sharpens** the blend and produced extreme prices out of
sample.

The challenger is therefore built with a **temperature floor** \(\tau\ge 1\)
(a genuine convex logit blend: \(0\le a,b\le 1\), \(a+b=1\), \(\tau\ge 1\),
pinned by
`tests/test_stacker_temperature_floor.py::test_stacker_weights_are_convex_when_stacker_is_used`).
When the unconstrained fit sharpens, the Elo/rank weight \(w\) is preserved, the
coefficients are projected onto \(a+b=1\), and the intercept is refit:

\[
a=0.3977,\qquad
b=0.6023,\qquad
c=0.3348,\qquad
\tau=1.0.
\]

These are the challenger's deploy-form coefficients, stored under the
`challenger` block of `artifacts/selected_spec.json` (`status: "rejected"`).
They score `outputs/challenger_blend_april_predictions.csv`, which is a
reference only — the challenger is **not** the deployed model. On the frozen
April window the blend scores log loss 0.468725 / Brier 0.150465, worse than
Elo-only on both proper scores.

## 8. Metrics

Log loss:

\[
LL =
-\frac{1}{N}
\sum_g
\left[
Y_g\log(p_g)
+
(1-Y_g)\log(1-p_g)
\right].
\]

Brier score:

\[
BS =
\frac{1}{N}
\sum_g
(p_g-Y_g)^2.
\]

AUC measures ranking. Accuracy uses the fixed threshold \(p_g \ge 0.5\). Log
loss and Brier are primary because the deliverable is a probability price.

## 9. Fair odds

The output also reports zero-margin decimal odds:

\[
O_{home} = \frac{1}{p_g},
\qquad
O_{away} = \frac{1}{1-p_g}.
\]

These are mathematical fair odds, not a production sportsbook quote. They
contain no overround, risk adjustment, liability response, or trader override.
