"""
3. Tables 2 & 3 – Regression Analyses
======================================
Table 2: Association of vitamin D with TMJ OA presence (binary logistic)
        and severity (ordinal logistic).
Table 3: Association of vitamin D with pain intensity VAS
        (OLS with HC3 robust SE).

Usage
-----
    python 3_table2_table3_regression_analysis.py

Outputs
-------
    outputs/Table2_vitaminD_TMJOA_presence_severity.csv
    outputs/Table3_vitaminD_and_VAS.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

from config import OUTDIR, load_and_prepare_data, ensure_numeric


# ── Table 2 ──────────────────────────────────────────────────────────────────

def build_table2(df):
    use_cols = [
        "TMJ_OA", "TMJ_OA_grade", "VitaminD", "VitaminD_deficiency01",
        "Age", "Sex_female_bin", "Symptom_onset_log1p", "ESR", "RF_elevated01",
        "Zinc", "GSI", "TMJ_noise", "Muscle_stiffness", "Jaw_locking", "Bruxism",
    ]
    dat = ensure_numeric(df.copy(), use_cols).dropna()

    covars = [
        "Age", "Sex_female_bin", "Symptom_onset_log1p", "ESR", "RF_elevated01",
        "Zinc", "GSI", "TMJ_noise", "Muscle_stiffness", "Jaw_locking", "Bruxism",
    ]
    results = []

    # ── Binary logistic: TMJ OA presence ─────────────────────────────────
    for exposure in ["VitaminD", "VitaminD_deficiency01"]:
        tmp = dat.copy()
        if exposure == "VitaminD":
            tmp["VitaminD_per10"] = tmp["VitaminD"] / 10
            exp_col = "VitaminD_per10"
        else:
            exp_col = exposure

        X = sm.add_constant(tmp[[exp_col] + covars])
        model = sm.Logit(tmp["TMJ_OA"], X).fit(disp=False)
        beta = model.params[exp_col]
        se   = model.bse[exp_col]
        p    = model.pvalues[exp_col]
        results.append({
            "Outcome":  "TMJ OA presence",
            "Exposure": "Vitamin D per 10 ng/mL" if exposure == "VitaminD" else "Vitamin D deficiency",
            "Model":    "Adjusted logistic regression",
            "Beta": beta, "OR": np.exp(beta),
            "CI_low": np.exp(beta - 1.96 * se),
            "CI_high": np.exp(beta + 1.96 * se),
            "p_value": p,
        })

    # ── Ordinal logistic: TMJ OA severity ────────────────────────────────
    for exposure in ["VitaminD", "VitaminD_deficiency01"]:
        tmp = dat.copy()
        if exposure == "VitaminD":
            tmp["VitaminD_per10"] = tmp["VitaminD"] / 10
            exp_col = "VitaminD_per10"
        else:
            exp_col = exposure

        y = tmp["TMJ_OA_grade"].astype(int)
        X = tmp[[exp_col] + covars]
        fit = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False)
        beta = fit.params[exp_col]
        se   = fit.bse[exp_col]
        p    = fit.pvalues[exp_col]
        results.append({
            "Outcome":  "TMJ OA severity",
            "Exposure": "Vitamin D per 10 ng/mL" if exposure == "VitaminD" else "Vitamin D deficiency",
            "Model":    "Adjusted ordinal logistic regression",
            "Beta": beta, "OR": np.exp(beta),
            "CI_low": np.exp(beta - 1.96 * se),
            "CI_high": np.exp(beta + 1.96 * se),
            "p_value": p,
        })

    out = pd.DataFrame(results)
    out_path = OUTDIR / "Table2_vitaminD_TMJOA_presence_severity.csv"
    out.to_csv(out_path, index=False)
    print(f"Table 2 saved → {out_path}")
    return out


# ── Table 3 ──────────────────────────────────────────────────────────────────

def build_table3(df):
    use_cols = [
        "VAS", "VitaminD", "VitaminD_deficiency01", "TMJ_OA_grade", "Age",
        "Sex_female_bin", "ESR", "RF_elevated01", "Zinc", "GSI",
        "Muscle_stiffness", "Jaw_locking", "Bruxism",
    ]
    dat = ensure_numeric(df.copy(), use_cols).dropna()

    covars = [
        "Age", "Sex_female_bin", "TMJ_OA_grade", "ESR", "RF_elevated01",
        "Zinc", "GSI", "Muscle_stiffness", "Jaw_locking", "Bruxism",
    ]
    rows = []

    # Vitamin D continuous (per 10 ng/mL)
    tmp = dat.copy()
    tmp["VitaminD_per10"] = tmp["VitaminD"] / 10
    X   = sm.add_constant(tmp[["VitaminD_per10"] + covars])
    fit = sm.OLS(tmp["VAS"], X).fit(cov_type="HC3")
    beta, se, p = fit.params["VitaminD_per10"], fit.bse["VitaminD_per10"], fit.pvalues["VitaminD_per10"]
    rows.append({
        "Predictor": "Vitamin D per 10 ng/mL",
        "Beta": beta, "CI_low": beta - 1.96 * se, "CI_high": beta + 1.96 * se,
        "p_value": p,
    })

    # Vitamin D deficiency (binary)
    X   = sm.add_constant(dat[["VitaminD_deficiency01"] + covars])
    fit = sm.OLS(dat["VAS"], X).fit(cov_type="HC3")
    beta, se, p = fit.params["VitaminD_deficiency01"], fit.bse["VitaminD_deficiency01"], fit.pvalues["VitaminD_deficiency01"]
    rows.append({
        "Predictor": "Vitamin D deficiency",
        "Beta": beta, "CI_low": beta - 1.96 * se, "CI_high": beta + 1.96 * se,
        "p_value": p,
    })

    table3 = pd.DataFrame(rows)
    out_path = OUTDIR / "Table3_vitaminD_and_VAS.csv"
    table3.to_csv(out_path, index=False)
    print(f"Table 3 saved → {out_path}")
    return table3


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_prepare_data()
    build_table2(df)
    build_table3(df)
