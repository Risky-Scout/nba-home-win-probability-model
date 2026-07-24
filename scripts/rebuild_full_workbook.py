"""Rebuild NBA_Model_Fully_Formulated.xlsx to reconcile with the Elo-only champion.

The workbook is a fully live-formula "twin" of the model: sheet 2 holds the
architecture parameters, and every downstream sheet (Elo chain, Bradley-Terry,
trend, the on-sheet Newton fits, and the frozen-April prices) is computed by cell
formulas that reference those parameters. Opening the file in Excel recalculates
the entire chain; the blue reconciliation cells carry the committed artifact
values so on-sheet == blue.

This transformer updates the workbook to the promoted model:

  1. Parameters (sheet 2) switch to the selected architecture ``conservative``
     (Elo-only champion): K=7.5, home advantage 55, Elo logistic C=10. The
     margin-of-victory multiplier uses the winner-minus-loser rating difference.
  2. The DEPLOYED price is Elo-only. Sheet 10's final column is the calibrated
     Elo probability from the through-March fit (sheet 9), reconciled to
     ``outputs/april_predictions.csv``.
  3. The Elo + Bradley-Terry/recent-trend blend is retained only as a clearly
     labelled REJECTED CHALLENGER (sheet 8); it is not deployed.

Usage (from repo root, venv active):
  python -m scripts.rebuild_full_workbook
"""

from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.styles import Font

from nba_wp.data import load_games
from nba_wp.features import Architecture, build_features
from nba_wp.model import fit_base_models, standardized_coefficient_rows

ROOT = Path(__file__).resolve().parents[1]
WB_PATH = ROOT / "NBA_Model_Fully_Formulated.xlsx"

BLUE = Font(color="1F4E78")
BOLD = Font(bold=True)


def _rank_rows(games: pd.DataFrame, arch: Architecture, max_date: str) -> dict[str, pd.Series]:
    feats = build_features(games[games["game_date"] < "2026-04-01"].copy(), arch)
    train = feats[feats["game_date"] < max_date].copy()
    models = fit_base_models(train, arch)
    rows = pd.DataFrame(standardized_coefficient_rows(models))
    out = {}
    for feature in ["bt_logit", "trend_diff", "(intercept)"]:
        out[feature] = rows[(rows["component"] == "rank") & (rows["feature"] == feature)].iloc[0]
    return out


def main() -> None:
    spec = json.loads((ROOT / "artifacts" / "selected_spec.json").read_text())
    arch = Architecture.from_dict(spec["architecture"])
    elo = spec["elo_model"]
    cal = spec["challenger"]["calibration"]  # rejected challenger stacker
    metrics = json.loads((ROOT / "artifacts" / "final_metrics.json").read_text())
    primary = metrics["primary_holdout"]["april"]

    dep_a = float(cal["coef_elo_logit"])
    dep_b = float(cal["coef_rank_logit"])
    dep_c = float(cal["intercept"])
    unc_a = float(cal["unconstrained_coef_elo_logit"])
    unc_b = float(cal["unconstrained_coef_rank_logit"])
    unc_c = float(cal["unconstrained_intercept"])

    coef = pd.read_csv(ROOT / "artifacts" / "coefficient_table.csv")

    def elo_row(feature: str) -> pd.Series:
        return coef[(coef["component"] == "elo") & (coef["feature"] == feature)].iloc[0]

    elo_std = elo_row("elo_diff")
    elo_int = elo_row("(intercept)")

    games = load_games(ROOT / "data" / "nba-win-probability-data.csv")
    rank_march = _rank_rows(games, arch, "2026-04-01")  # through-March challenger rank
    bt_std = rank_march["bt_logit"]
    tr_std = rank_march["trend_diff"]
    rank_int = rank_march["(intercept)"]

    eng = pd.read_csv(ROOT / "outputs" / "engineered_features.csv")
    fa = pd.read_csv(ROOT / "outputs" / "april_predictions_frozen_snapshot.csv")
    ch = pd.read_csv(ROOT / "outputs" / "challenger_blend_april_predictions.csv")

    wb = openpyxl.load_workbook(WB_PATH)

    # ---- Sheet 2: architecture parameters -> conservative (Elo-only champion) -
    s2 = wb["2_Params"]
    s2["A1"] = (
        'ARCHITECTURE PARAMETERS — provenance: artifacts/selected_spec.json '
        '(Elo-only champion, architecture "conservative"). Fields below the Elo '
        'block parameterize the rejected challenger blend only.'
    )
    s2["B5"] = float(arch.elo_k)      # 7.5
    s2["B6"] = float(arch.elo_hfa)    # 55
    s2["B11"] = float(arch.bt_c)      # 0.1 (challenger)
    s2["B12"] = float(arch.trend_half_life_days)  # 60 (challenger)
    s2["B13"] = int(arch.trend_short_games)       # 12 (challenger)
    s2["B14"] = float(arch.elo_model_c)           # 10
    s2["B15"] = float(arch.rank_model_c)          # 0.1 (challenger)

    # ---- Sheet 3: Elo chain MOV fix (winner-loser) + refreshed elo_diff refs --
    s3 = wb["3_Elo_Chain"]
    for r in range(3, 1233):
        s3[f"J{r}"] = (
            f"=LN(ABS(E{r})+1)*('2_Params'!$B$8/"
            f"MAX('2_Params'!$B$10,(IF(L{r}=1,F{r}-G{r},G{r}-F{r}))"
            f"*'2_Params'!$B$9+'2_Params'!$B$8))"
        )
    for i, val in enumerate(eng["elo_diff"].to_numpy()):
        c = s3[f"M{i + 3}"]
        c.value = float(val)
        c.font = BLUE

    # ---- Sheet 4: refreshed training bt_logit / trend_diff imports -----------
    s4 = wb["4_Seq_Features"]
    bt = eng["bt_logit"].to_numpy()
    td = eng["trend_diff"].to_numpy()
    for i in range(len(eng)):
        gc, hc = s4[f"G{i + 3}"], s4[f"H{i + 3}"]
        gc.value, hc.value = float(bt[i]), float(td[i])
        gc.font = hc.font = BLUE

    # ---- Sheet 8: REJECTED CHALLENGER stacker (not deployed) ------------------
    s8 = wb["8_Fit_Stacker"]
    s8["A1"] = (
        "REJECTED CHALLENGER — Elo + Bradley-Terry/recent-trend logistic stack. "
        "NOT DEPLOYED: the nested rolling-origin audit shows it is worse than "
        "Elo-only out-of-sample (see artifacts/nested_*_summary.json). Shown for "
        "transparency and to demonstrate the convex temperature-floor (T>=1) fit."
    )
    s8["H20"] = "RECONCILIATION vs selected_spec.json challenger (unconstrained MLE)"
    s8["J22"], s8["J23"], s8["J24"] = unc_a, unc_b, unc_c
    for coord in ("J22", "J23", "J24"):
        s8[coord].font = BLUE

    s8["H26"] = "CONVEX + TEMPERATURE FLOOR -> DEPLOY-FORM (model.py, T>=1, w in [0,1])"
    s8["H26"].font = BOLD
    s8["H27"], s8["I27"] = "a_unc (=Q5)", "=$Q$5"
    s8["H28"], s8["I28"] = "b_unc (=Q6)", "=$Q$6"
    s8["H29"], s8["I29"] = "c_unc (=Q7)", "=$Q$7"
    s8["H30"], s8["I30"] = "a+ = max(a,0)", "=MAX(I27,0)"
    s8["H31"], s8["I31"] = "b+ = max(b,0)", "=MAX(I28,0)"
    s8["H32"], s8["I32"] = "total+ = a+ + b+", "=I30+I31"
    s8["H33"], s8["I33"] = "w = a+/total+", "=IF(I32>0,I30/I32,0.5)"
    s8["H34"], s8["I34"] = "T = max(1/total+, 1)", "=MAX(IF(I32>0,1/I32,1),1)"
    s8["H35"], s8["I35"] = "a deploy = w/T", "=I33/I34"
    s8["H36"], s8["I36"] = "b deploy = (1-w)/T", "=(1-I33)/I34"

    s8["M26"] = "c refit Newton (a_deploy,b_deploy fixed); start c0=c_unc"
    s8["N27"] = "=I29"
    for r in range(28, 41):
        prev = r - 1
        sig = f"(1/(1+EXP(-($I$35*$D$5:$D$243+$I$36*$E$5:$E$243+N{prev}))))"
        s8[f"N{r}"] = (
            f"=N{prev}-(SUMPRODUCT({sig}-$C$5:$C$243))/(SUMPRODUCT({sig}*(1-{sig})))"
        )
    s8["H37"], s8["I37"] = "c deploy", "=N40"

    s8["H38"] = "DEPLOY-FORM vs selected_spec.json challenger"
    s8["H38"].font = BOLD
    s8["H39"], s8["I39"], s8["J39"], s8["K39"] = "a", "=I35", dep_a, "=ABS(I39-J39)"
    s8["H40"], s8["I40"], s8["J40"], s8["K40"] = "b", "=I36", dep_b, "=ABS(I40-J40)"
    s8["H41"], s8["I41"], s8["J41"], s8["K41"] = "c", "=I37", dep_c, "=ABS(I41-J41)"
    for coord in ("J39", "J40", "J41"):
        s8[coord].font = BLUE

    # ---- Sheet 9: refreshed final base-model coefficient references -----------
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

    # ---- Sheet 10: DEPLOYED Elo-only champion (through-March fit, frozen April)-
    s10 = wb["10_April_Frozen"]
    s10["A1"] = (
        "APRIL - FROZEN PRE-APRIL (PRIMARY, DEPLOYED CHAMPION = ELO-ONLY): team "
        "state frozen at March 31; price = calibrated Elo probability from the "
        "through-March fit (sheet 9)."
    )
    s10["A2"] = (
        "Frozen elo_diff sums ONLY pre-April deltas from sheet 3. Columns I/J "
        "(Feb-selection components) and the sheet-8 stacker are the REJECTED "
        "challenger, shown for transparency. Reconciliation references (blue) "
        "from outputs/april_predictions.csv (== frozen snapshot)."
    )
    s10["I4"] = "p_elo (Feb, challenger input)"
    s10["J4"] = "p_rank (Feb, challenger input)"
    s10["K4"] = "p_final = Elo-only champion (through-Mar)"
    for r in range(5, 101):
        # Champion deployed price: calibrated Elo (through-March, sheet 9) on the
        # frozen elo_diff (column F).
        s10[f"K{r}"] = (
            f"=1/(1+EXP(-('9_Fit_Final_Models'!$AC$12*"
            f"(F{r}-'9_Fit_Final_Models'!$U$5)/'9_Fit_Final_Models'!$U$6"
            f"+'9_Fit_Final_Models'!$AC$13)))"
        )
    for i in range(len(fa)):
        r = i + 5
        for col, name in (("L", "home_win_probability"), ("N", "elo_diff"),
                          ("O", "bt_logit"), ("P", "trend_diff")):
            c = s10[f"{col}{r}"]
            c.value = float(fa[name].iloc[i])
            c.font = BLUE
    # Rejected challenger blend display + reconcile (columns V/W/X).
    s10["V4"], s10["W4"], s10["X4"] = (
        "p_blend (REJECTED)", "repo p_blend", "|delta|",
    )
    for r in range(5, 101):
        s10[f"V{r}"] = (
            f"=1/(1+EXP(-('8_Fit_Stacker'!$I$35*LN(I{r}/(1-I{r}))"
            f"+'8_Fit_Stacker'!$I$36*LN(J{r}/(1-J{r}))+'8_Fit_Stacker'!$I$37)))"
        )
        s10[f"X{r}"] = f"=ABS(V{r}-W{r})"
    for i in range(len(ch)):
        c = s10[f"W{i + 5}"]
        c.value = float(ch["home_win_probability"].iloc[i])
        c.font = BLUE

    s10["A102"] = (
        "METRICS - Elo-only champion, computed by formula, reconciled vs "
        "artifacts/final_metrics.json (primary_holdout.april)"
    )
    s10["C104"] = float(primary["log_loss"])
    s10["C105"] = float(primary["brier"])
    s10["C106"] = float(primary["accuracy"])
    s10["C107"] = float(primary["auc"])
    for coord in ("C104", "C105", "C106", "C107"):
        s10[coord].font = BLUE

    # ---- Sheet 11: dashboard -> champion Elo-only reconcile -------------------
    s11 = wb["11_Reconcile_Dashboard"]
    s11["A7"] = "Champion Elo std coef vs coefficient_table.csv"
    s11["B7"], s11["D7"] = "='9_Fit_Final_Models'!$AC$12", float(elo_std["standardized_coefficient"])
    s11["A8"] = "Champion Elo intercept vs coefficient_table.csv"
    s11["B8"], s11["D8"] = "='9_Fit_Final_Models'!$AC$13", float(elo_int["standardized_coefficient"])
    s11["A9"] = "Challenger stacker a (deploy) vs selected_spec.json [REJECTED]"
    s11["B9"], s11["D9"] = "='8_Fit_Stacker'!$I$35", "='8_Fit_Stacker'!K39"
    for coord in ("D7", "D8"):
        s11[coord].font = BLUE

    # ---- Sheet 0: README text refresh ---------------------------------------
    s0 = wb["0_README"]
    s0["B2"] = (
        "Source of truth: github.com/Risky-Scout/nba-home-win-probability-model  "
        "·  main (DEPLOYED CHAMPION = Elo-only; Elo+rank blend implemented and "
        "REJECTED by nested rolling-origin validation)"
    )
    s0["B11"] = (
        "The deployed champion is Elo-only: a margin-of-victory Elo rating "
        "difference (K=7.5, home advantage 55, log MOV multiplier on the "
        "WINNER-minus-LOSER rating difference) turned into a probability by a "
        "standardized, L2-penalized logistic regression fitted on all games "
        "through March 31. April team state is frozen at March 31, so no April "
        "outcome leaks into any April prediction. An Elo + Bradley-Terry/recent-"
        "trend logistic-stack blend was also built and then REJECTED: under "
        "nested rolling-origin validation it is worse than Elo-only on log loss "
        "and Brier and is worse calibrated."
    )
    s0["B21"] = (
        "8_Fit_Stacker - REJECTED CHALLENGER. March component log-odds from the "
        "selection models; 3-parameter Newton fit gives the unconstrained a,b,c; "
        "a convex + temperature-floor block (w in [0,1], T>=1) projects to the "
        "deploy-form a,b,c, reconciled to the challenger block of "
        f"selected_spec.json (deploy {dep_a:.5f} / {dep_b:.5f} / {dep_c:.5f}). "
        "This blend is not deployed."
    )
    s0["B23"] = (
        "10_April_Frozen - 96 games priced by the DEPLOYED Elo-only champion "
        "(through-March Elo fit, sheet 9) on frozen pre-April features, with "
        "per-game reconciliation vs outputs/april_predictions.csv and Log Loss / "
        "Brier / Accuracy / AUC by formula vs artifacts/final_metrics.json. "
        "Columns V-X show the rejected blend for comparison."
    )

    wb.save(WB_PATH)
    print(f"Saved {WB_PATH}")


if __name__ == "__main__":
    main()
