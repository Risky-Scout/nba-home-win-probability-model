# Audit response

## Preservation

- Immutable tag: `v1-original-submission`
- Remediation branch: `cursor/audit-remediation-7b85`

The original submission is preserved. This branch addresses statistical-governance
concerns without rewriting that history.

Authoritative outputs live in `artifacts/current/`. Legacy v1 search artifacts
are quarantined under `artifacts/v1_legacy/`.

## What changed

1. **Selection governance** — expanding-window Jan/Feb folds; no March/April rows.
2. **Objective** — mean validation log loss; no four-target benchmark gate.
3. **Search budget** — 63 direct-logistic candidates plus nine architecture-matched
   blend challengers (**72 total**), not ~341k temperature/shift settings.
4. **Champion form** — direct L2 logistic on `elo_diff`, `bt_logit`, `trend_diff`
   when it wins pre-March selection; original blend retained as challenger.
5. **April primary result** — frozen March 31 snapshot.
6. **Uncertainty** — date-block bootstrap (conditional on locked specification);
   calibration intercept/slope/ECE reported as diagnostics.
7. **Claims** — no sportsbook alpha; prototype readiness only; April retrospective.

## Governance honesty

`pre_march_selection_proof.json` proves what the current executable path consumes.
It does **not** prove that this architecture grid was historically specified
before April was viewed. Describe this as a **reconstructed governance path**,
not genuine historical preregistration.

## Model comparison honesty

The direct logistic won the declared pre-March validation process, but its
incremental April value over Elo remains **statistically unresolved**. Date-block
proper-score differences versus Elo are small and include zero.

## Calibration honesty

Frozen-April diagnostics (ECE ≈ 0.113, intercept ≈ 0.256, slope ≈ 1.44) are
reported as findings. A slope above one suggests probabilities may be too close
to 0.50. Calibration is **not** claimed as solved.

## What cannot be fixed with this dataset

- April cannot be made pristine again after prior viewing.
- Sportsbook alpha cannot be established without market prices.
- Deployment readiness cannot be demonstrated from one season alone.

## Final conclusion

The repository demonstrates a strong leakage-controlled and reproducible
probability-modeling prototype. After remediation, model selection is governed
using pre-March chronological validation, March is a locked test, frozen
March 31 scoring is the primary April result, and uncertainty is reported using
date blocks. The model is not presented as proof of sportsbook alpha or as
production-ready.
