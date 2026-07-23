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

The selected architecture uses \(H=75\) Elo points. After the date is complete,

\[
R_h' = R_h + K M_g (Y_g-p^{Elo}_g),
\]

\[
R_a' = R_a - K M_g (Y_g-p^{Elo}_g),
\]

where \(K=10\) and the margin-of-victory multiplier follows the
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

The raw Elo feature is

\[
x^{Elo}_g = \frac{R_h-R_a+H}{400}.
\]

An L2-logistic calibration model is fitted to historical \(x^{Elo}\).

## 4. Bradley-Terry component

For each date, team strengths are fitted from earlier game outcomes:

\[
P(h \text{ beats } a)
=
\sigma(
\alpha + q_h-q_a
),
\]

where \(\alpha\) is the home intercept and \(q_i\) is team \(i\)'s fitted
strength. L2 regularization controls the 30 team coefficients. The selected
regularization is `C = 0.15`.

The model feature is the Bradley-Terry decision value:

\[
x^{BT}_g = \hat \alpha + \hat q_h-\hat q_a.
\]

## 5. Recent trend

For team \(i\), define point margin \(m_{ik}\) in each prior game from the
team's perspective. Long-form margin is exponentially weighted:

\[
L_{i,d}
=
\frac{
\sum_{k<d}
2^{-\Delta days_{kd}/45} m_{ik}
}{
\sum_{k<d}
2^{-\Delta days_{kd}/45}
}.
\]

Short-form margin is the mean over the most recent ten games:

\[
S_{i,d}
=
\frac{1}{n_i}
\sum_{\text{last } \min(10,n_i)} m_{ik}.
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

## 6. Log-odds blend (logistic stacking)

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

The unconstrained March fit produced:

\[
a=0.5696,\qquad
b=1.0159,\qquad
c=0.3132,
\]

equivalent to \((w, \tau, s)\) via \(w=a/(a+b)=0.359\),
\(\tau=1/(a+b)=0.631\), \(s=c=0.313\). Because the two component logits are
near-duplicates (\(\rho\approx 0.97\)), the unconstrained fit learns
\(a+b>1\), i.e. temperature \(\tau<1\), which **sharpens** the blend and
produced extreme (99%+) prices out of sample.

The model is therefore deployed with a **temperature floor** \(\tau\ge 1\).
When the unconstrained fit sharpens, the Elo/rank weight \(w\) is preserved but
the coefficients are projected onto \(a+b=1\) (a convex logit blend, no
sharpening) and the intercept is refit:

\[
a=0.3593,\qquad
b=0.6407,\qquad
c=0.3254,\qquad
\tau=1.0.
\]

These deploy coefficients are what score April. The floor is enforced in code
and pinned by `tests/test_stacker_temperature_floor.py`; both the unconstrained
and deployed coefficients are stored in `artifacts/selected_spec.json`.

## 7. Metrics

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

## 8. Fair odds

The output also reports zero-margin decimal odds:

\[
O_{home} = \frac{1}{p_g},
\qquad
O_{away} = \frac{1}{1-p_g}.
\]

These are mathematical fair odds, not a production sportsbook quote. They
contain no overround, risk adjustment, liability response, or trader override.
