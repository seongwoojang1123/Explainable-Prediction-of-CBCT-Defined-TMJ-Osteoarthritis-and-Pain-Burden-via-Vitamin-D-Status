"""
5. Figure 5 – Vitamin D and Pain Intensity (VAS)
=================================================
Three-panel figure:
  A. VAS by vitamin D deficiency status (box-scatter)
  B. Vitamin D level by VAS group ≤5 vs ≥6 (box-scatter)
  C. Vitamin D vs VAS scatter plot with Spearman correlation

"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import OUTDIR, load_and_prepare_data, mann_whitney_p, simple_box_scatter


def make_figure5(df):
    fig = plt.figure(figsize=(14, 5))

    # ── A. VAS by vitamin D deficiency ───────────────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    g_non = pd.to_numeric(df.loc[df["VitaminD_deficiency01"] == 0, "VAS"], errors="coerce").dropna().values
    g_def = pd.to_numeric(df.loc[df["VitaminD_deficiency01"] == 1, "VAS"], errors="coerce").dropna().values
    p = mann_whitney_p(g_non, g_def)
    simple_box_scatter(ax1, [g_non, g_def], ["Non-deficient", "Deficient"],
                       ylabel="Pain intensity (VAS)", p_text=f"p = {p:.4g}")
    ax1.set_xlabel("Vitamin D deficiency status")
    ax1.text(-0.18, 1.03, "A", transform=ax1.transAxes, fontsize=16, fontweight="bold")

    # ── B. Vitamin D by VAS group ────────────────────────────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    tmp = df[["VitaminD", "VAS"]].dropna().copy()
    tmp["VAS_group"] = np.where(tmp["VAS"] <= 5, "VAS ≤ 5", "VAS ≥ 6")
    g_low  = tmp.loc[tmp["VAS_group"] == "VAS ≤ 5", "VitaminD"].values
    g_high = tmp.loc[tmp["VAS_group"] == "VAS ≥ 6", "VitaminD"].values
    p = mann_whitney_p(g_low, g_high)
    simple_box_scatter(ax2, [g_low, g_high], ["VAS ≤ 5", "VAS ≥ 6"],
                       ylabel="Vitamin D (ng/mL)", p_text=f"p = {p:.4g}")
    ax2.set_xlabel("Pain intensity group")
    ax2.text(-0.18, 1.03, "B", transform=ax2.transAxes, fontsize=16, fontweight="bold")

    # ── C. Scatter: Vitamin D vs VAS ─────────────────────────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    tmp = df[["VitaminD", "VAS"]].dropna().copy()
    x, y = tmp["VitaminD"].values, tmp["VAS"].values
    ax3.scatter(x, y, alpha=0.7, s=28)

    coef = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax3.plot(x_line, coef[0] * x_line + coef[1], linewidth=2)

    rho, p = stats.spearmanr(x, y)
    ax3.text(0.03, 0.97, f"Spearman rho = {rho:.3f}\np = {p:.4f}",
             transform=ax3.transAxes, va="top", ha="left", fontsize=10)
    ax3.set_xlabel("Vitamin D (ng/mL)")
    ax3.set_ylabel("Pain intensity (VAS)")
    ax3.grid(axis="y", alpha=0.15)
    ax3.text(-0.18, 1.03, "C", transform=ax3.transAxes, fontsize=16, fontweight="bold")

    fig.tight_layout()
    out_path = OUTDIR / "Figure5_VitaminD_and_VAS.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 5 saved → {out_path}")


if __name__ == "__main__":
    df = load_and_prepare_data()
    make_figure5(df)
