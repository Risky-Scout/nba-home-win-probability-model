# File index — keep these tabs ready

Paths are relative to the repo root. Authoritative metrics live under **`artifacts/current/`**.

## Speak / navigate

| Tab | File |
|---|---|
| Landing | `START_HERE.md` |
| 2-min summary | `SUMMARY.md` |
| **Script** | `docs/interview/PRESENTATION_SCRIPT_90MIN.md` |
| Parameter ledger | `docs/interview/PARAMETER_LEDGER.md` |
| Q&A (Q1–Q110) | `docs/SENIOR_QUANT_QA.md` |
| Cheatsheet | `docs/INTERVIEW_QA_CHEATSHEET.md` |
| One-pager | `docs/PRESENTATION_ONE_PAGER.md` |
| Setup | `docs/CURSOR_PRESENTATION_SETUP.md` |

## Assignment originals

| File |
|---|
| `docs/assignment/nba-win-probability-instructions.pdf` |
| `docs/assignment/Quantitative_Analyst_Sports_JD.pdf` |

## Source code (chronological story)

| Step | File |
|---|---|
| Load + audit | `src/nba_wp/data.py` |
| Elo / BT / trend features | `src/nba_wp/features.py` |
| Direct logistic + metrics + bootstrap | `src/nba_wp/model.py` |
| Pre-March selection | `src/nba_wp/selection.py` |
| Select CLI | `python -m nba_wp.cli select` |
| Score CLI | `scripts/score_final.py` |
| Reporting / frozen April | `src/nba_wp/reporting.py` |
| Validator | `validate_submission.py` |

## Configs (declared search)

| File | Contents |
|---|---|
| `configs/model.yaml` | Jan/Feb folds; log-loss primary |
| `configs/model.yaml` | 72-candidate budget |
| `configs/model.yaml (benchmarks section)` | Retrospective reference only |

## Artifacts to show on screen

| Claim | File |
|---|---|
| Home rate / audit | `artifacts/current/data_audit.json` |
| No March/April in selection | `artifacts/current/pre_march_selection_proof.json` |
| Winner spec | `artifacts/current/selected_spec_pre_march.json` |
| Fold LL table | `artifacts/current/pre_march_fold_results.csv` |
| Full candidate table | `artifacts/current/pre_march_selection_results.csv` |
| Coefficients | `artifacts/current/model_coefficients.json` |
| Final metrics | `artifacts/current/final_metrics.json` |
| Calibration | `artifacts/current/calibration_diagnostics.json` |
| Uncertainty | `artifacts/current/date_block_bootstrap_summary.json` |
| Feature governance | `artifacts/current/feature_governance.csv` |
| Ablation | `artifacts/current/feature_group_ablation.csv` |
| Importance | `artifacts/current/permutation_importance.csv` |

## Predictions

| Role | File |
|---|---|
| **Primary April** | `predictions/april_predictions.csv` |
| Sequential April (sensitivity) | `outputs/april_predictions.csv` |
| Locked March | `outputs/march_predictions.csv` |
| Engineered features | `outputs/engineered_features.csv` |

## Figures

| File |
|---|
| `figures/march_calibration.png` |
| `figures/april_calibration.png` |
| `figures/permutation_importance.png` |
| `figures/ablation_log_loss.png` |

## Governance / honesty docs

| Topic | File |
|---|---|
| Audit response | `docs/AUDIT_RESPONSE.md` |
| Benchmark provenance | `docs/BENCHMARK_PROVENANCE.md` |
| Market limitations | `docs/MARKET_PRICING_LIMITATIONS.md` |
| Validation policy | `docs/VALIDATION_AND_GOVERNANCE.md` |
| Results language | `docs/RESULTS_AND_INTERPRETATION.md` |
| Methodology | `docs/METHODOLOGY.md` |
| Feature engineering | `docs/FEATURE_ENGINEERING.md` |

## Legacy (do not present as current)

`artifacts/v1_legacy/` — original March-search / old bootstrap files.  
Immutable tag: `v1-original-submission`.
