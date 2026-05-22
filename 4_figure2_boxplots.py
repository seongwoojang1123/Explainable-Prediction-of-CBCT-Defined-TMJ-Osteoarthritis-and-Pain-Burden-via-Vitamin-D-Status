"""
4. Figure 2 – Vitamin D, ESR, GSI by TMJ OA Status / Grade
===========================================================
Six-panel box-and-scatter figure (3 × 2 grid).
Left column: Non-TMJ OA vs TMJ OA (Mann–Whitney U).
Right column: Grade 0 vs 1 vs 2 (Kruskal–Wallis).

Usage
-----
    python 4_figure2_boxplots.py

Output
------
    outputs/Figure2_VitD_ESR_GSI.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import OUTDIR, load_and_prepare_data, mann_whitney_p, simple_box_scatter


def make_figure2(df):
    fig = plt.figure(figsize=(12, 12))

    variables = [
        ("VitaminD", "Vitamin D (ng/mL)",
         "Vitamin D according to TMJ OA status",
         "Vitamin D according to TMJ OA grade"),
        ("ESR", "ESR (mm/hr)",
         "ESR according to TMJ OA status",
         "ESR according to TMJ OA grade"),
        ("GSI", "GSI",
         "GSI according to TMJ OA status",
         "GSI according to TMJ OA grade"),
    ]

    panel_labels = ["a", "b", "c", "d", "e", "f"]
    panel_idx = 0

    for row_idx, (var, ylabel, ttl1, ttl2) in enumerate(variables):
        # ── Left: status (2 groups) ──────────────────────────────────────
        ax1 = fig.add_subplot(3, 2, row_idx * 2 + 1)
        g0 = pd.to_numeric(df.loc[df["TMJ_OA"] == 0, var], errors="coerce").dropna().values
        g1 = pd.to_numeric(df.loc[df["TMJ_OA"] == 1, var], errors="coerce").dropna().values
        p = mann_whitney_p(g0, g1)
        simple_box_scatter(ax1, [g0, g1], ["Non-TMJ OA", "TMJ OA"],
                           ylabel=ylabel, title=ttl1, p_text=f"p = {p:.4g}")
        ax1.text(-0.08, 1.03, panel_labels[panel_idx],
                 transform=ax1.transAxes, fontsize=18, fontweight="bold")
        panel_idx += 1

        # ── Right: grade (3 groups) ──────────────────────────────────────
        ax2 = fig.add_subplot(3, 2, row_idx * 2 + 2)
        g_grade = [
            pd.to_numeric(df.loc[df["TMJ_OA_grade"] == g, var], errors="coerce").dropna().values
            for g in [0, 1, 2]
        ]
        p_kw = stats.kruskal(*g_grade).pvalue
        simple_box_scatter(ax2, g_grade, ["Grade 0", "Grade 1", "Grade 2"],
                           ylabel=ylabel, title=ttl2)
        ax2.text(-0.08, 1.03, panel_labels[panel_idx],
                 transform=ax2.transAxes, fontsize=18, fontweight="bold")
        ax2.text(0.02, 0.96, f"Kruskal–Wallis p = {p_kw:.4g}",
                 transform=ax2.transAxes, fontsize=10)
        panel_idx += 1

    fig.tight_layout()
    out_path = OUTDIR / "Figure2_VitD_ESR_GSI.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 saved → {out_path}")


if __name__ == "__main__":
    df = load_and_prepare_data()
    make_figure2(df)
