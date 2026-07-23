"""Rebuild NBA_Model_Fully_Formulated.xlsx to reconcile with the corrected model.

The workbook is a fully live-formula "twin" of the Python model: the Elo chain,
Bradley-Terry gradient proof, trend, the on-sheet Newton fits (base models and
stacker) and the frozen-April predictions are all computed by cell formulas, and
every stage reconciles against the committed artifacts.

This transformer updates the committed workbook so it matches the current model:

  1. Elo MOV multiplier now uses the winner-minus-loser rating difference
     (FiveThirtyEight form) instead of home-minus-away. This is applied to the
     live Elo-chain formula (sheet 3, column J), which then cascades through the
     whole live fit chain automatically.
  2. Every baked reconciliation reference (blue) value is refreshed from the
     regenerated artifacts / outputs.
  3. The logistic stacker is deployed with a temperature floor (T >= 1): the
     unconstrained Newton fit is kept, and a projection block computes the
     deployed a, b, c (a + b = 1, intercept refit by a 1-D Newton). The April
     sheet and the dashboard use the deployed coefficients.
  4. The primary frozen-April sheet now scores with the February-trained
     selection base models (sheet 7) and the deployed stacker, matching
     nba_wp/reporting.py.

Usage (from repo root, venv active):
  python -m scripts.rebuild_full_workbook
"""

from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.styles import Font

ROOT = Path(__file__).resolve().parents[1]
WB_PATH = ROOT / "NBA_Model_Fully_Formulated.xlsx"

BLUE = Font(color="1F4E78")
BOLD = Font(bold=True)


def _num(cell_value: object) -> float:
    return float(cell_value)


def main() -> None:
    spec = json.loads((ROOT / "artifacts" / "selected_spec.json").read_text())
    cal = spec["calibration"]
    metrics = json.loads((ROOT / "artifacts" / "final_metrics.json").read_text())
    primary = metrics["primary_holdout"]["april"]

    # Deployed (floored) and unconstrained stacker coefficients.
    dep_a = float(cal["coef_elo_logit"])
    dep_b = float(cal["coef_rank_logit"])
    dep_c = float(cal["intercept"])
    unc_a = float(cal["unconstrained_coef_elo_logit"])
    unc_b = float(cal["unconstrained_coef_rank_logit"])
    unc_c = float(cal["unconstrained_intercept"])

    # Final (through-March) base-model coefficients for the sheet 9 reconcile.
    coef = pd.read_csv(ROOT / "artifacts" / "coefficient_table.csv")

    def coef_row(component: str, feature: str) -> pd.Series:
        return coef[(coef["component"] == component) & (coef["feature"] == feature)].iloc[0]

    elo_std = coef_row("elo", "elo_diff")
    elo_int = coef_row("elo", "(intercept)")
    bt_std = coef_row("rank", "bt_logit")
    tr_std = coef_row("rank", "trend_diff")
    rank_int = coef_row("rank", "(intercept)")

    eng = pd.read_csv(ROOT / "outputs" / "engineered_features.csv")
    fa = pd.read_csv(ROOT / "outputs" / "april_predictions_frozen_snapshot.csv")

    wb = openpyxl.load_workbook(WB_PATH)

    # ---- Sheet 3: Elo chain MOV fix + refreshed elo_diff references ----------
    s3 = wb["3_Elo_Chain"]
    for r in range(3, 1233):
        s3[f"J{r}"] = (
            f"=LN(ABS(E{r})+1)*('2_Params'!$B$8/"
            f"MAX('2_Params'!$B$10,(IF(L{r}=1,F{r}-G{r},G{r}-F{r}))"
            f"*'2_Params'!$B$9+'2_Params'!$B$8))"
        )
    elo_diff = eng["elo_diff"].to_numpy()
    for i, val in enumerate(elo_diff):
        c = s3[f"M{i + 3}"]
        c.value = float(val)
        c.font = BLUE

    # ---- Sheet 4: refreshed training bt_logit / trend_diff imports -----------
    s4 = wb["4_Seq_Features"]
    bt = eng["bt_logit"].to_numpy()
    td = eng["trend_diff"].to_numpy()
    for i in range(len(eng)):
        gc = s4[f"G{i + 3}"]
        hc = s4[f"H{i + 3}"]
        gc.value = float(bt[i])
        hc.value = float(td[i])
        gc.font = BLUE
        hc.font = BLUE

    # ---- Sheet 8: unconstrained repo refs + deploy (temperature floor) -------
    s8 = wb["8_Fit_Stacker"]
    s8["H20"] = "RECONCILIATION vs selected_spec.json (unconstrained MLE)"
    s8["J22"] = unc_a
    s8["J23"] = unc_b
    s8["J24"] = unc_c
    for coord in ("J22", "J23", "J24"):
        s8[coord].font = BLUE

    s8["H26"] = "TEMPERATURE FLOOR -> DEPLOY (model.py fit_logit_stacker, T>=1)"
    s8["H26"].font = BOLD
    s8["H27"], s8["I27"] = "a_unc (=Q5)", "=$Q$5"
    s8["H28"], s8["I28"] = "b_unc (=Q6)", "=$Q$6"
    s8["H29"], s8["I29"] = "c_unc (=Q7)", "=$Q$7"
    s8["H30"], s8["I30"] = "total = a+b", "=I27+I28"
    s8["H31"], s8["I31"] = "w = a/total", "=I27/I30"
    s8["H32"], s8["I32"] = "T = 1/total", "=1/I30"
    s8["H33"], s8["I33"] = "floor applied?", '=IF(I32<1,"yes","no")'
    s8["H34"], s8["I34"] = "a deploy", "=IF(I32<1,I31,I27)"
    s8["H35"], s8["I35"] = "b deploy", "=IF(I32<1,1-I31,I28)"
    s8["H36"], s8["I36"] = "c deploy", "=IF(I32<1,N40,I29)"

    # 1-D Newton refit of the intercept with a_deploy, b_deploy fixed (col N).
    s8["M26"] = "c refit Newton (a_deploy,b_deploy fixed); start c0=c_unc"
    s8["N27"] = "=I29"
    for r in range(28, 41):
        prev = r - 1
        sig = (
            f"(1/(1+EXP(-($I$34*$D$5:$D$243+$I$35*$E$5:$E$243+N{prev}))))"
        )
        s8[f"N{r}"] = (
            f"=N{prev}-"
            f"(SUMPRODUCT({sig}-$C$5:$C$243))/"
            f"(SUMPRODUCT({sig}*(1-{sig})))"
        )

    s8["H38"] = "DEPLOY vs selected_spec.json"
    s8["H38"].font = BOLD
    s8["H39"], s8["I39"], s8["J39"], s8["K39"] = "a", "=I34", dep_a, "=ABS(I39-J39)"
    s8["H40"], s8["I40"], s8["J40"], s8["K40"] = "b", "=I35", dep_b, "=ABS(I40-J40)"
    s8["H41"], s8["I41"], s8["J41"], s8["K41"] = "c", "=I36", dep_c, "=ABS(I41-J41)"
    for coord in ("J39", "J40", "J41"):
        s8[coord].font = BLUE

    # ---- Sheet 9: refreshed final base-model coefficient references ----------
    s9 = wb["9_Fit_Final_Models"]
    s9["V18"] = float(elo_std["standardized_coefficient"])
    s9["V19"] = float(elo_int["standardized_coefficient"])
    s9["V20"] = float(elo_std["training_mean"])
    s9["V21"] = float(elo_std["training_scale"])
    s9["BE21"] = float(bt_std["standardized_coefficient"])
    s9["BE22"] = float(tr_std["standardized_coefficient"])
    s9["BE23"] = float(rank_int["standardized_coefficient"])
    s9["BE24"] = float(bt_std["training_mean"])
    s9["BE25"] = float(bt_std["training_scale"])
    s9["BE26"] = float(tr_std["training_mean"])
    s9["BE27"] = float(tr_std["training_scale"])
    for coord in ("V18", "V19", "V20", "V21", "BE21", "BE22", "BE23", "BE24", "BE25", "BE26", "BE27"):
        s9[coord].font = BLUE

    # ---- Sheet 10: February selection models + deploy stacker + refs ---------
    s10 = wb["10_April_Frozen"]
    s10["A1"] = (
        "APRIL - FROZEN PRE-APRIL (PRIMARY): team state frozen at March 31; "
        "February-trained selection base models (sheet 7) + deploy (T>=1) "
        "stacker (sheet 8)"
    )
    s10["A2"] = (
        "Frozen elo_diff sums ONLY pre-April deltas from sheet 3. bt from sheet "
        "5 strengths. trend from sheet 6. All reconciliation references (blue) "
        "from outputs/april_predictions_frozen_snapshot.csv (== april_predictions.csv)."
    )
    for r in range(5, 101):
        s10[f"I{r}"] = (
            f"=1/(1+EXP(-('7_Fit_Selection_Models'!$AC$12*"
            f"(F{r}-'7_Fit_Selection_Models'!$U$5)/'7_Fit_Selection_Models'!$U$6"
            f"+'7_Fit_Selection_Models'!$AC$13)))"
        )
        s10[f"J{r}"] = (
            f"=1/(1+EXP(-('7_Fit_Selection_Models'!$BL$14*"
            f"(G{r}-'7_Fit_Selection_Models'!$BD$5)/'7_Fit_Selection_Models'!$BD$6"
            f"+'7_Fit_Selection_Models'!$BL$15*"
            f"(H{r}-'7_Fit_Selection_Models'!$BD$7)/'7_Fit_Selection_Models'!$BD$8"
            f"+'7_Fit_Selection_Models'!$BL$16)))"
        )
        s10[f"K{r}"] = (
            f"=1/(1+EXP(-('8_Fit_Stacker'!$I$34*LN(I{r}/(1-I{r}))"
            f"+'8_Fit_Stacker'!$I$35*LN(J{r}/(1-J{r}))"
            f"+'8_Fit_Stacker'!$I$36)))"
        )
    for i in range(len(fa)):
        r = i + 5
        for col, name in (("L", "home_win_probability"), ("N", "elo_diff"),
                          ("O", "bt_logit"), ("P", "trend_diff")):
            c = s10[f"{col}{r}"]
            c.value = float(fa[name].iloc[i])
            c.font = BLUE
    s10["A102"] = (
        "METRICS - computed by formula, reconciled vs "
        "artifacts/final_metrics.json (primary_holdout.april)"
    )
    s10["C104"] = float(primary["log_loss"])
    s10["C105"] = float(primary["brier"])
    s10["C106"] = float(primary["accuracy"])
    s10["C107"] = float(primary["auc"])
    for coord in ("C104", "C105", "C106", "C107"):
        s10[coord].font = BLUE

    # ---- Sheet 11: dashboard points at deploy stacker cells ------------------
    s11 = wb["11_Reconcile_Dashboard"]
    s11["A7"] = "Stacker a (deploy) vs selected_spec.json"
    s11["B7"], s11["D7"] = "='8_Fit_Stacker'!$I$34", "='8_Fit_Stacker'!K39"
    s11["A8"] = "Stacker b (deploy) vs selected_spec.json"
    s11["B8"], s11["D8"] = "='8_Fit_Stacker'!$I$35", "='8_Fit_Stacker'!K40"
    s11["A9"] = "Stacker c (deploy) vs selected_spec.json"
    s11["B9"], s11["D9"] = "='8_Fit_Stacker'!$I$36", "='8_Fit_Stacker'!K41"

    # ---- Sheet 0: README text refresh ---------------------------------------
    s0 = wb["0_README"]
    s0["B2"] = (
        "Source of truth: github.com/Risky-Scout/nba-home-win-probability-model  "
        "·  main (corrected model: Elo MOV winner-loser fix + stacker temperature "
        "floor T>=1 + frozen pre-April primary)"
    )
    s0["B11"] = (
        "Two base signals are computed for every game strictly from prior "
        "information: (1) an Elo rating difference (K=10, home advantage 75, log "
        "margin-of-victory multiplier evaluated on the WINNER-minus-LOSER rating "
        "difference) and (2) a rank signal combining regularized Bradley-Terry "
        "strengths with a schedule-weighted momentum trend. Each signal is turned "
        "into a probability by a standardized, L2-penalized logistic regression "
        "fitted through February; a logistic stacker is fitted on March component "
        "log-odds and then DEPLOYED WITH A TEMPERATURE FLOOR (T>=1) so correlated "
        "signals cannot sharpen into extreme prices. Primary April scoring uses "
        "the February-trained base models under the frozen pre-April policy: team "
        "state frozen at March 31, no April outcome leaks into any April "
        "prediction."
    )
    s0["B21"] = (
        "8_Fit_Stacker - March component log-odds from the selection models; "
        "3-parameter Newton fit gives the unconstrained a, b, c; a temperature-"
        "floor block then projects to the deployed a, b, c (a+b=1, intercept "
        "refit), reconciled to artifacts/selected_spec.json "
        f"(deploy {dep_a:.5f} / {dep_b:.5f} / {dep_c:.5f})."
    )
    s0["B23"] = (
        "10_April_Frozen - 96 games priced with the February selection base "
        "models (sheet 7) and the deploy stacker (sheet 8): frozen features on-"
        "sheet, component probabilities, stacked probability, per-game "
        "reconciliation vs outputs/april_predictions.csv, and Log Loss / Brier / "
        "Accuracy / AUC by formula vs artifacts/final_metrics.json."
    )

    wb.save(WB_PATH)
    print(f"Saved {WB_PATH}")


if __name__ == "__main__":
    main()
