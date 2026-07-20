# Optimality statement — what "optimal given the constraints" means here

## Claim

The shipped model is the **log-loss minimizer over a declared, exhaustive-for-
this-dataset candidate space, chosen by the strongest selection estimator the
data allow, without ever touching March or April**. That is the strongest
optimality claim this assignment's data can support.

## The three components of the claim

### 1. Candidate space (declared in `configs/model.yaml` before selection ran)

Everything derivable from the supplied CSV without leakage:

- Sequential Elo: K ∈ {5, 10, 20, 30}, home advantage ∈ {50, 65, 80},
  log margin-of-victory
- Regularized Bradley–Terry strengths (prior games only)
- Form trend (short window − EWMA, half-life ∈ {20, 45})
- Rest / back-to-back / schedule-density advantages
- Lagged record, margin, turnover, rebound, foul differentials
- Five **nested feature sets** from Elo-only to all of the above
- L2 strength C ∈ {0.01 … 10}

672 candidates after deduplicating the half-life axis for trend-free sets.

### 2. Selection estimator

**Prequential daily expanding validation**: for every date d in January–
February, fit on all games strictly before d and score the games on d. Pooled
per-game log loss over **399 validation games** — every pre-March game after
the burn-in period is an out-of-sample point exactly once, under exactly the
deployment condition (predict tomorrow from everything before it). This
dominates fold-mean estimators (2 points) in precision and matches the
April use case in form.

### 3. Governance

Selection input ends 2026-02-28 (guard raises otherwise; proof JSON records 0
March / 0 April rows). The winner was locked before March was scored once.
April is scored from the frozen 2026-03-31 state.

## The result

The winner is the **simplest candidate**: Elo-only logistic (K=5, HFA=80,
C=0.03), prequential log loss 0.6309. The nested ladder was free to choose
Bradley–Terry, trend, rest, and box-score differentials — **they all lost**.
On 399 validation games, the extra features add noise faster than signal.
The tie-break preferred fewer features, but the Elo-only set won outright.

Locked March (scored once): log loss 0.5135, 184/239. Frozen April: log loss
0.4746, Brier 0.1530, AUC 0.866, 75/96.

## What this optimality claim does NOT cover

1. **Feature classes the data cannot express** — injuries, lineups, travel,
   minutes, market prices. No model trained on this CSV can incorporate them.
2. **Model families outside the declared space** — e.g. gradient boosting.
   With ~900 training games and weak features beyond Elo, tree ensembles are
   expected to overfit; they were excluded a priori, not evaluated and hidden.
3. **Estimation noise at n=399** — a candidate within ~0.005 log loss of the
   winner is statistically indistinguishable from it. Concretely: the same
   Elo-only feature at default regularization (C=1.0) trailed C=0.03 by only
   0.0012 prequential log loss, and on the locked test and April it scores
   ~0.01 *better* (paired CI narrowly excludes zero). That is the measured
   cost of honest selection noise. Switching to C=1.0 now would be test
   leakage, so the declared winner ships and this is disclosed instead. The
   claim is "minimizer of the declared search under the best available
   estimator," not "provably better than every near-tie."
4. **History** — April was viewed during the wider project. The executable
   pipeline provably excludes it from selection, but this remains a
   reconstructed governance path, not preregistration.

## Why the "simpler model won" outcome is the credible one

The earlier three-feature model was selected by a 2-point estimator (two fold
means). The prequential estimator uses ~200× more validation information. When
the better estimator disagrees with the weaker one, the better one governs.
The three-feature model and v1 blend remain documented challengers in
`reports/metrics.json`; on frozen April the Elo-only model is also the best
performer among them (LL 0.4746 vs 0.4906 for the challenger scored in
metrics.json — reported for transparency, never used for selection).
