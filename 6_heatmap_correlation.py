"""
6. Correlation Heat Maps
========================
Generates Spearman-based heat maps for:
  (a) VAS – factors significantly correlated with pain intensity
  (b) TMJ OA severity – factors significantly correlated with OA grade

Each map includes only variables that pass p < 0.05 screening.

Usage
-----
    python 6_heatmap_correlation.py

Outputs
-------
    outputs/Heatmap_VAS_significant_factors.png
    outputs/Heatmap_VAS_significant_factors.screening.csv
    outputs/Heatmap_VAS_significant_factors.corr.csv
    outputs/Heatmap_VAS_significant_factors.pval.csv
    outputs/Heatmap_TMJOA_severity_significant_factors.png   (+ .screening / .corr / .pval)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import OUTDIR, PRETTY, load_and_prepare_data, add_sig_stars


# ── Core heatmap builder ─────────────────────────────────────────────────────

def _heatmap_from_target(df, target, candidates, title, outfile):
    """Screen variables, then build and save a correlation heat map."""

    # Step 1 – screen for significant correlations with the target
    rows = []
    for var in candidates:
        if var == target:
            continue
        sub = df[[target, var]].dropna()
        if len(sub) < 20:
            continue
        rho, p = stats.spearmanr(sub[target], sub[var])
        rows.append({
            "Variable": var,
            "Pretty":   PRETTY.get(var, var),
            "n": len(sub), "rho": rho, "p_value": p,
        })

    screen = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    sig = screen.loc[screen["p_value"] < 0.05].copy()

    # Step 2 – full pairwise correlation matrix
    heat_vars   = [target] + sig["Variable"].tolist()
    heat_labels = [PRETTY.get(v, v) for v in heat_vars]

    corr_mat = pd.DataFrame(index=heat_labels, columns=heat_labels, dtype=float)
    pval_mat = pd.DataFrame(index=heat_labels, columns=heat_labels, dtype=float)

    for i, v1 in enumerate(heat_vars):
        for j, v2 in enumerate(heat_vars):
            sub = df[[v1, v2]].dropna()
            if v1 == v2:
                rho, p = 1.0, 0.0
            else:
                rho, p = stats.spearmanr(sub[v1], sub[v2])
            corr_mat.iloc[i, j] = rho
            pval_mat.iloc[i, j] = p

    # Step 3 – plot
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_mat.values.astype(float), vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(heat_labels)))
    ax.set_yticks(np.arange(len(heat_labels)))
    ax.set_xticklabels(heat_labels, rotation=45, ha="right")
    ax.set_yticklabels(heat_labels)

    for i in range(len(heat_labels)):
        for j in range(len(heat_labels)):
            val = corr_mat.iloc[i, j]
            p   = pval_mat.iloc[i, j]
            stars = "" if i == j else add_sig_stars(p)
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}{stars}", ha="center", va="center",
                    color=color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman correlation coefficient")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Step 4 – save companion CSVs
    screen.to_csv(outfile.with_suffix(".screening.csv"), index=False)
    corr_mat.to_csv(outfile.with_suffix(".corr.csv"))
    pval_mat.to_csv(outfile.with_suffix(".pval.csv"))
    print(f"Heatmap saved → {outfile}")


# ── Public entry point ───────────────────────────────────────────────────────

def make_heatmaps(df):
    candidates_vas = [
        "Age", "Sex_female_bin", "Symptom_onset", "TMJ_OA", "TMJ_OA_grade", "VAS",
        "VitaminD", "VitaminD_deficiency01", "ESR", "RF", "Zinc", "GSI",
        "TMJ_noise", "Muscle_stiffness", "Jaw_locking", "Bruxism",
    ]
    _heatmap_from_target(
        df, "VAS", candidates_vas,
        "Heat map of factors significantly associated with VAS",
        OUTDIR / "Heatmap_VAS_significant_factors.png",
    )

    candidates_oa = [
        "Age", "Sex_female_bin", "Symptom_onset", "VAS",
        "VitaminD", "ESR", "RF", "Zinc", "GSI",
        "TMJ_noise", "Muscle_stiffness", "Jaw_locking", "Bruxism",
    ]
    _heatmap_from_target(
        df, "TMJ_OA_grade", candidates_oa,
        "Heat map of factors significantly associated with TMJ OA severity",
        OUTDIR / "Heatmap_TMJOA_severity_significant_factors.png",
    )


if __name__ == "__main__":
    df = load_and_prepare_data()
    make_heatmaps(df)
