# Repository audit

Audit performed before restructuring, on branch `cursor/submission-restructure-7b85`
(parent: remediated `cursor/audit-remediation-7b85`, original preserved at tag
`v1-original-submission`). Every reported metric was re-derived from
`nba-win-probability-data.csv` before any edit; all matched committed artifacts
(frozen April LL 0.469453 / Brier 0.151060 / AUC 0.862348 / 74/96; locked March
182/239).

Severity: Critical / High / Medium / Low.
Status: `fixed` (this branch), `already-fixed` (remediation branch), `accepted` (documented, intentionally kept), `open` (disclosed limitation).

| Severity | Category | File or function | Issue | Evidence | Consequence | Recommended fix | Status |
|---|---|---|---|---|---|---|---|
| Critical | Statistical validity | v1 `selection.py` (tag) | March used for architecture + calibration selection, then reported as "test" | v1 tag: selection loops score March directly | March metrics were post-selection, not a test | Pre-March expanding-window selection; March locked | already-fixed |
| Critical | Statistical validity | v1 `configs/benchmarks` | Four March benchmark floats acted as a hard promotion gate with unknown provenance | v1 selection required beating all four | Circular selection; unverifiable acceptance criteria | Benchmarks context-only; provenance doc | already-fixed |
| Critical | Statistical validity | v1 tuning grid | 341,155 correlated temperature/shift settings vs 239 March games | v1 `march_tuning_top_candidates.csv` | Severe selection multiplicity; \(p=\sigma(z/T+b)\) monotone in \(z\) so most "models" identical in ranking | 72-candidate declared budget; Platt calibration | already-fixed |
| High | Statistical validity | April headline | Sequential (rolling) April was headlined although the task says use Oct–Mar | v1 README | Overstated assignment result | Frozen March-31 snapshot is primary; rolling kept as scenario file | already-fixed / fixed (separate files) |
| High | Statistical validity | selection protocol | Only 2 expanding folds (Jan, Feb); preferred protocol also validates December | `configs/selection_policy.json` | Weaker selection evidence; December unused for validation | Add fold train→Nov / validate Dec; rerun selection | fixed |
| High | Statistical validity | baselines | No committed comparison against constant-rate and record-difference baselines with paired uncertainty | absent artifact | Cannot demonstrate skill over trivial models | `evaluation.py`: 4 baselines, paired per-game ΔLL + bootstrap CI | fixed |
| High | Engineering | packaging/CLI | No single obvious command; no `python -m nba_wp.cli predict`; no lockfile; CI single-OS | repo root | Reviewer friction; unpinned installs; no Windows check | src layout, `cli.py`, `uv.lock`, Ubuntu+Windows CI | fixed |
| High | Engineering | predictions schema | April CSV lacks away probability, fair odds, model_version, information_cutoff | `outputs/april_predictions*.csv` headers | Not a self-describing price sheet | `predictions/april_predictions.csv` with full schema | fixed |
| Medium | Statistical validity | ensemble (blend challenger) | Blend vs direct-logistic difference never quantified with correlation + paired CI | absent | Cannot justify keeping/dropping challenger | Component correlation + Δ log loss + CI in `reports/metrics.json` | fixed |
| Medium | Statistical validity | feature drift | Cumulative-season features' monthly drift not committed as a diagnostic | absent artifact | Extrapolation risk into April unquantified | Monthly mean/std/quantiles + max standardized distance artifact | fixed |
| Medium | Statistical validity | calibration language | Slope ≈1.44 / ECE ≈0.11 on 96 games must not be described as "solved" | `calibration_diagnostics.json` | Overclaim risk | Report as diagnostic with sample-size caveat | already-fixed |
| Medium | Statistical validity | April status | April was viewed during the wider project before this protocol | project history | April is retrospective, not pristine | State plainly in README/report | already-fixed / restated |
| Medium | Engineering | configs | Three JSON configs; no single `model.yaml`; duplication risk | `configs/*.json` | Config sprawl | Single `configs/model.yaml` consumed by CLI + scripts | fixed |
| Medium | Engineering | committed artifacts | Generated artifacts (figures, CSVs, joblib) committed | `artifacts/`, `figures/` | Repo bloat; drift vs code risk | Keep (interview needs offline evidence) but validator recomputes; documented | accepted |
| Medium | Engineering | docs duplication | Metrics hand-typed in several docs | docs/*.md | Numbers drift from code | One machine-readable `reports/metrics.json`; reports generated from it | fixed |
| Medium | Sports validity | rest/travel/B2B | Rest days / travel / back-to-backs derivable from schedule but not in champion | `feature_governance.csv` | Possible unused signal | Documented as evaluated-and-rejected (rest) / future extensions (travel) | accepted |
| Low | Engineering | naming | "champion/production" language in places | docs | Overpromise | Neutral naming: baseline / candidate / selected_model / challenger / locked_test | fixed |
| Low | Engineering | CI file refs | CI compiled `run_submission.py` etc. by hard path; silent skip if file absent not verified | `.github/workflows/tests.yml` | Weak guarantee | CI runs CLI smoke + schema verification that fail hard | fixed |
| Low | Statistical validity | AUC/accuracy emphasis | Rounding-level AUC/accuracy differences previously described as wins | v1 README | Misleading precision | Effective-tie language; exact correct-game counts | already-fixed |
| Low | Engineering | Windows | No PowerShell instructions | README | Windows reviewer friction | PowerShell section in README | fixed |

## Statistical validity — detailed answers

- **Target**: `home_win = 1{home_points > away_points}`; ties rejected by loader (0 in data). Defined once in `data.py`.
- **Pregame cutoff**: wins/losses columns are pregame (verified: 0 reconciliation mismatches across all 1,230 games); points/turnovers/fouls/rebounds are postgame and are **never** used same-game; they enter only via lagged team state.
- **Same-game leakage**: feature-before-update ordering; same-date games batched (state updates only after all games that date). Tests: `test_feature_timing.py`.
- **Future-game leakage**: features at game *g* built from games strictly before *g*'s date; appending later games leaves earlier rows unchanged (tested).
- **Duplicates / missing / record consistency**: loader raises on duplicate `game_id`, missing values, ties, self-play; full-season reconciliation of the win/loss columns has 0 mismatches.
- **Train/valid/test**: Oct–Nov→Dec, Oct–Dec→Jan, Oct–Jan→Feb expanding folds (selection); March locked test scored once; April frozen forecast. Hyperparameters, model family, calibration policy, tie-breaks all frozen in `configs/model.yaml` before March.
- **March untouched?** On this protocol, yes — `pre_march_selection_proof.json` (0 March rows, 0 April rows) plus tests that raise if a March+ row enters selection. Historically (v1) March was used; that is why v1 is superseded.
- **April used for decisions?** No April rows in the executable selection path (proof + tests). April was *seen* earlier in the project's life, so we do not call it pristine.
- **Log loss / Brier**: `sklearn.metrics.log_loss(labels=[0,1])`, `brier_score_loss`; verified by hand-recomputation in the walkthrough notebook.
- **AUC/accuracy limits**: reported as descriptive; 96-game April sample ⇒ wide intervals (date-block bootstrap CIs committed).
- **Extreme probabilities**: min/max predicted probabilities and extreme-bin audit committed; no clipping beyond numerical epsilon.
- **Collinearity**: `elo_diff` vs `bt_logit` correlation reported in `feature_correlations.csv` (both encode strength; ridge \(C=0.1\) handles it; ablation quantifies marginal value).
- **Refit logic**: selected pipeline refit once on Oct–Mar for April; no April information enters the fit (frozen-state test).

## Sports-modeling validity — feature availability

| Feature | In supplied data? | Derivable without leakage? | Used in selected model? |
|---|---|---|---|
| Wins/losses (pregame) | Yes | Yes (given) | Via BT/Elo history; record-diff baseline |
| Point differential | Postgame only | Yes, lagged | Yes (Elo MOV, trend) |
| Turnovers/fouls/rebounds | Postgame only | Yes, lagged | Evaluated; rejected (noise) |
| Recent form | — | Yes (EWMA of prior margins) | Yes (`trend_diff`) |
| Home advantage | — | Yes (constant HFA in Elo) | Yes (HFA=65) |
| Strength of schedule | — | Yes (implicit in Elo/BT opponents) | Yes (implicit) |
| Rest / B2B | — | Yes (schedule dates) | Evaluated; rejected pre-March |
| Travel | Not in data | Needs venue coords | Future extension |
| Injuries / lineups | Not in data | No | Future extension (documented) |
| Late-season rotation | Not in data | Weak proxy only | Limitation (April risk) |

## Engineering quality — findings

- Structure: flat `nba_wp/` moved to `src/nba_wp/`; CLI added; single entry command.
- Dead/stale: v1 search artifacts quarantined in `artifacts/v1_legacy/`; none referenced by code.
- Determinism: fixed seeds for bootstrap; sklearn deterministic given inputs; loader sorts by (date, game_id) so input row order cannot change predictions (tested).
- CI: now Ubuntu + Windows, lockfile install, ruff, pytest, CLI smoke on fixture, schema/bounds verification. A missing required file fails loudly (smoke test reads real paths).
- Paths: `pathlib` everywhere; no shell-specific separators in library code.
