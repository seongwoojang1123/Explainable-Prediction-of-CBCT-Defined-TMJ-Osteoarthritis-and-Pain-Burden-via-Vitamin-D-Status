"""
2. Table 1 – Baseline Characteristics by TMJ OA Status
======================================================
Compares continuous and categorical variables between
Non-TMJ OA and TMJ OA groups, with Bonferroni correction.

Usage
-----
    python 2_table1_baseline_characteristics.py

Output
------
    outputs/Table1_baseline_characteristics.csv
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from statsmodels.stats.multitest import multipletests

from config import (
    OUTDIR, load_and_prepare_data,
    mean_sd, count_pct, mann_whitney_p, chi2_p, fisher_p,
)


def build_table1(df):
    group0 = df[df["TMJ_OA"] == 0]
    group1 = df[df["TMJ_OA"] == 1]

    # ── Define variable sets ─────────────────────────────────────────────
    continuous = [
        ("Age, years",                "Age"),
        ("Pain intensity, VAS",       "VAS"),
        ("Symptom duration",          "Symptom_onset"),
        ("Vitamin D, ng/mL",          "VitaminD"),
        ("RF",                        "RF"),
        ("ESR",                       "ESR"),
        ("Zinc",                      "Zinc"),
        ("Global Severity Index, GSI","GSI"),
    ]

    categorical = [
        ("Sex, female",          "Sex_female_bin",         1, "chi2"),
        ("Vitamin D deficiency", "VitaminD_deficiency01",  1, "chi2"),
        ("Elevated RF",          "RF_elevated01",          1, "fisher"),
        ("TMJ noise",            "TMJ_noise",              1, "chi2"),
        ("Muscle stiffness",     "Muscle_stiffness",       1, "chi2"),
        ("Jaw locking",          "Jaw_locking",            1, "fisher"),
        ("Bruxism",              "Bruxism",                1, "fisher"),
    ]

    rows, raw_ps = [], []

    # ── Continuous variables (Mann–Whitney U) ────────────────────────────
    for label, col in continuous:
        p = mann_whitney_p(group0[col], group1[col])
        raw_ps.append(p)
        rows.append({
            "Variable":   label,
            "Overall":    mean_sd(df[col]),
            "Non-TMJ OA": mean_sd(group0[col]),
            "TMJ OA":     mean_sd(group1[col]),
            "p_value":    p,
        })

    # ── Categorical variables (Chi-squared / Fisher) ─────────────────────
    for label, col, value, test_kind in categorical:
        p = chi2_p(df["TMJ_OA"], df[col]) if test_kind == "chi2" \
            else fisher_p(df["TMJ_OA"], df[col])
        raw_ps.append(p)
        rows.append({
            "Variable":   label,
            "Overall":    count_pct(df[col], value),
            "Non-TMJ OA": count_pct(group0[col], value),
            "TMJ OA":     count_pct(group1[col], value),
            "p_value":    p,
        })

    # ── Bonferroni adjustment ────────────────────────────────────────────
    adj = multipletests(raw_ps, method="bonferroni")[1]
    for r, p_adj in zip(rows, adj):
        r["Bonferroni_adjusted_p_value"] = p_adj

    table1 = pd.DataFrame(rows)
    out_path = OUTDIR / "Table1_baseline_characteristics.csv"
    table1.to_csv(out_path, index=False)
    print(f"Table 1 saved → {out_path}")
    return table1


if __name__ == "__main__":
    df = load_and_prepare_data()
    build_table1(df)
