"""
7. Model Performance Summary & SHAP Placeholder
================================================
Table 4: Manually entered model-performance metrics from the
         cross-validated Elastic Net and XGBoost experiments.
Figure 3: ROC / calibration curve images are maintained separately.
Figure 4: SHAP summary – placeholder instructions for when the
         raw SHAP plotting table becomes available.

Usage
-----
    python 7_model_performance_and_SHAP.py

Outputs
-------
    outputs/Table4_model_performance_summary.csv
    outputs/Figure4_SHAP_placeholder.txt
"""

import pandas as pd
from config import OUTDIR


# ── Table 4 ──────────────────────────────────────────────────────────────────

def write_table4():
    rows = [
        ["Elastic Net", "Clinical only",                        0.515, 0.573, 0.253,  0.078, 0.246, 0.581, 0.457],
        ["Elastic Net", "Clinical + labs without Vitamin D",     0.543, 0.595, 0.253,  0.069, 0.309, 0.581, 0.448],
        ["Elastic Net", "Clinical + labs + Vitamin D",           0.729, 0.752, 0.216, -0.002, 0.868, 0.735, 0.552],
        ["Elastic Net", "Clinical + labs + Vitamin D + GSI",     0.740, 0.753, 0.211, -0.003, 0.851, 0.744, 0.571],
        ["XGBoost",     "Clinical only",                        0.493, 0.511, 0.253,  0.131,-0.188, 0.701, 0.267],
        ["XGBoost",     "Clinical + labs without Vitamin D",     0.518, 0.542, 0.251,  0.068, 0.265, 0.752, 0.314],
        ["XGBoost",     "Clinical + labs + Vitamin D",           0.727, 0.766, 0.206, -0.060, 1.724, 0.573, 0.705],
        ["XGBoost",     "Clinical + labs + Vitamin D + GSI",     0.755, 0.798, 0.204, -0.033, 1.755, 0.632, 0.781],
    ]
    cols = [
        "Algorithm", "Model", "AUROC", "AUPRC", "Brier_score",
        "Calibration_intercept", "Calibration_slope",
        "Sensitivity", "Specificity",
    ]
    table4 = pd.DataFrame(rows, columns=cols)
    out_path = OUTDIR / "Table4_model_performance_summary.csv"
    table4.to_csv(out_path, index=False)
    print(f"Table 4 saved → {out_path}")


# ── Figure 4 placeholder ────────────────────────────────────────────────────

def write_shap_placeholder():
    txt = """\
Figure 4 – SHAP Visualization Placeholder
------------------------------------------
Required files to reproduce SHAP figures programmatically:

1. Long-format SHAP plotting table (CSV):
   - feature, shap_value, feature_value

2. Optional feature-importance table (CSV):
   - feature, mean_abs_shap

Suggested outputs:
   - Figure4_SHAP_summary.png
   - Figure4_SHAP_violin.png

Note: The current workspace contains final SHAP images
but not the raw plotting table.  When available, add the
SHAP CSV to data/ and extend this script accordingly.
"""
    out_path = OUTDIR / "Figure4_SHAP_placeholder.txt"
    out_path.write_text(txt)
    print(f"SHAP placeholder saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    write_table4()
    write_shap_placeholder()
