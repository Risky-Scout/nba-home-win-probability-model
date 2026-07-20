# Benchmark provenance

## Status

The numerical values stored in `configs/benchmarks.json` are **retrospective
reference values**, not official assignment acceptance criteria.

The written assignment asks for April home-win probabilities using
October–March information. It does **not** specify numerical thresholds for
log loss, Brier, AUC, or accuracy.

## What is known

| Field | Statement |
|---|---|
| Source in assignment PDF | Not present |
| First recorded in this repository | Present in the original submission tag `v1-original-submission` |
| Used historically | Original `v1` selection required beating all four March values |
| Current role after remediation | Display-only context in comparison tables |

## What cannot be established

The exact external origin of the specific floats (for example `0.509645` and
`0.831798`) is **not documented** in the assignment materials available to
this repository. When provenance cannot be established, this document states
that explicitly rather than inventing a source.

## Remediation policy

Model selection on the audit-remediation branch:

- does **not** require beating these values;
- optimizes mean expanding-window validation log loss on pre-March folds;
- may still print benchmark deltas for reader context.

## March accuracy / AUC reporting

Even as reference comparisons, March accuracy at the displayed precision can
be an effective tie (186/239 ≈ 77.8243% vs a rounded 77.82%). Tiny AUC
differences at the 1e-7 scale are not treated as meaningful wins.
