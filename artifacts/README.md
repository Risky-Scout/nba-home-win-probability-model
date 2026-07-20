# Artifacts layout

| Directory | Contents |
|---|---|
| `artifacts/current/` | Authoritative remediation outputs (pre-March selection, frozen April metrics, date-block bootstrap, calibration). |
| `artifacts/v1_legacy/` | Original submission-era files retained only for comparison. The immutable source of truth for v1 is git tag `v1-original-submission`. |

Do not mix metrics across directories when reviewing results. Prefer:

1. `current/pre_march_selection_proof.json`
2. `current/final_metrics.json`
3. `current/date_block_bootstrap_summary.json`
4. `outputs/april_predictions_frozen_snapshot.csv`
