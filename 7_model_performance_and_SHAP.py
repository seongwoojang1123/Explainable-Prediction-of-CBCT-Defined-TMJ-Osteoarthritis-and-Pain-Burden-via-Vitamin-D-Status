"""
7. Model Performance Summary & SHAP Table Formatting
=====================================================
Formats Table 4 and the SHAP importance table from the outputs of
`8_final_holdout_ml_evaluation.py`. No performance values are entered
manually: this script only reads and formats the computed results, so
the repository tables always match the executed analysis.

Run order:
    python 8_final_holdout_ml_evaluation.py   # computes everything
    python 7_model_performance_and_SHAP.py    # formats the tables
"""

import pandas as pd
from config import OUTDIR

PERF = OUTDIR / "Table4_model_performance.csv"
SHAP = OUTDIR / "TableS4_shap_importance.csv"


def format_table4():
    if not PERF.exists():
        raise SystemExit(
            f"{PERF} not found. Run 8_final_holdout_ml_evaluation.py first.")
    df = pd.read_csv(PERF)
    df["AUROC (95% CI)"] = df.apply(
        lambda r: f"{r['AUROC']:.3f} ({r['AUROC_CI_low']:.3f}"
                  f"\u2013{r['AUROC_CI_high']:.3f})", axis=1)
    cols = ["Algorithm", "Model", "AUROC (95% CI)", "AUPRC", "Brier_score",
            "Calibration_intercept", "Calibration_slope",
            "Sensitivity", "Specificity"]
    out = df[cols].round(3)
    path = OUTDIR / "Table4_formatted.csv"
    out.to_csv(path, index=False)
    print(f"Table 4 (formatted) saved \u2192 {path}")
    print(out.to_string(index=False))


def format_shap():
    if not SHAP.exists():
        print(f"{SHAP} not found \u2014 run script 8 with shap installed "
              f"to generate the SHAP tables.")
        return
    imp = pd.read_csv(SHAP).round(3)
    path = OUTDIR / "TableS4_formatted.csv"
    imp.to_csv(path, index=False)
    print(f"\nSHAP importance table saved \u2192 {path}")
    print(imp.to_string(index=False))


if __name__ == "__main__":
    format_table4()
    format_shap()
