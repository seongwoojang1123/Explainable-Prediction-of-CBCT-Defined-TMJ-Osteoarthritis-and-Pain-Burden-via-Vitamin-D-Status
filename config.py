"""
Shared configuration and utility functions
==========================================
Centralizes paths, variable mappings, display labels,
and reusable helper functions for the TMJ OA + Vitamin D pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_PATH = Path("data/TMJOA_VitaminD_read_longnames.csv")
OUTDIR = Path("outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Variable mappings ────────────────────────────────────────────────────────

PSYCH_MAP = {
    "DEP": "DEP",
    "V22": "ANX",
    "V23": "HOS",
    "V24": "PHOB",
    "V25": "PAR",
    "V26": "PSY",
    "V27": "GSI_proxy_1",
    "V28": "GSI",
    "V29": "PDSI",
    "V30": "PST",
    "V31": "SOM",
    "V32": "OC",
    "V33": "IS",
}

PRETTY = {
    "TMJ_OA": "TMJ OA",
    "TMJ_OA_grade": "TMJ OA grade",
    "Sex_female": "Female sex",
    "Age": "Age",
    "VAS": "VAS",
    "Symptom_onset": "Symptom duration",
    "RF": "Rheumatoid factor",
    "VitaminD": "Vitamin D",
    "ESR": "ESR",
    "Zinc": "Zinc",
    "DEP": "DEP",
    "RF_elevated01": "Elevated RF",
    "VitaminD_deficiency01": "Vitamin D deficiency",
    "ESR_elevated01": "Elevated ESR",
    "Zinc_deficiency01": "Zinc deficiency",
    "DEP_increased": "DEP increased",
    "TMJ_noise": "TMJ noise",
    "TMJ_pain": "TMJ pain",
    "Muscle_stiffness": "Muscle stiffness",
    "Jaw_locking": "Jaw locking",
    "Bruxism": "Bruxism",
    "GSI": "Global Severity Index",
    "SOM": "Somatization",
    "OC": "Obsessive–compulsive",
    "IS": "Interpersonal sensitivity",
    "ANX": "Anxiety",
    "HOS": "Hostility",
    "PHOB": "Phobic anxiety",
    "PAR": "Paranoid ideation",
    "PSY": "Psychoticism",
    "PDSI": "Positive Distress Symptom Index",
    "PST": "Positive Symptom Total",
}

# ── Utility functions ────────────────────────────────────────────────────────

def load_and_prepare_data(path=DATA_PATH):
    """Load raw CSV and apply all preprocessing steps."""
    import warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv(path)

    # Rename psychological variables
    for old, new in PSYCH_MAP.items():
        if old in df.columns and new not in df.columns:
            df[new] = pd.to_numeric(df[old], errors="coerce")

    # Force numeric where relevant
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Recode sex (1=male, 2=female → 0/1)
    if "Sex_female" in df.columns:
        vals = sorted(pd.Series(df["Sex_female"]).dropna().unique().tolist())
        if vals == [1.0, 2.0] or vals == [1, 2]:
            df["Sex_female_bin"] = (df["Sex_female"] == 2).astype(float)
        else:
            df["Sex_female_bin"] = df["Sex_female"].astype(float)

    # Ensure binary variables are numeric 0/1
    binary_cols = [
        "TMJ_OA", "VitaminD_deficiency01", "RF_elevated01", "ESR_elevated01",
        "Zinc_deficiency01", "DEP_increased", "TMJ_noise", "TMJ_pain",
        "Muscle_stiffness", "Jaw_locking", "Bruxism",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # GSI fallback
    if "GSI" not in df.columns and "V28" in df.columns:
        df["GSI"] = pd.to_numeric(df["V28"], errors="coerce")

    # Log-transform symptom duration
    if "Symptom_onset" in df.columns:
        df["Symptom_onset_log1p"] = np.log1p(
            pd.to_numeric(df["Symptom_onset"], errors="coerce")
        )

    return df


# ── Descriptive helpers ──────────────────────────────────────────────────────

def mean_sd(series):
    """Return 'mean ± SD' string for a numeric series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    return f"{s.mean():.2f} ± {s.std(ddof=1):.2f}"


def count_pct(series, value=1):
    """Return 'n (pct%)' string for a binary series."""
    s = pd.to_numeric(series, errors="coerce")
    n = int((s == value).sum())
    d = int(s.notna().sum())
    pct = 100 * n / d if d > 0 else np.nan
    return f"{n} ({pct:.1f}%)"


def mann_whitney_p(x, y):
    """Two-sided Mann–Whitney U test p-value."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) == 0 or len(y) == 0:
        return np.nan
    return stats.mannwhitneyu(x, y, alternative="two-sided").pvalue


def chi2_p(x, y):
    """Chi-squared test p-value from a contingency table."""
    tbl = pd.crosstab(x, y)
    if tbl.size == 0:
        return np.nan
    return stats.chi2_contingency(tbl, correction=True)[1]


def fisher_p(x, y):
    """Fisher's exact test p-value (2×2 tables only)."""
    tbl = pd.crosstab(x, y)
    if tbl.shape != (2, 2):
        return np.nan
    return stats.fisher_exact(tbl)[1]


def add_sig_stars(p):
    """Return significance stars: *** / ** / * or empty string."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def ensure_numeric(df, cols):
    """Coerce selected columns to numeric."""
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def simple_box_scatter(ax, groups, labels, ylabel, title=None, p_text=None):
    """Draw box-and-scatter plot on a given Axes."""
    bp = ax.boxplot(groups, patch_artist=True, widths=0.55, zorder=1)
    for box in bp["boxes"]:
        box.set(facecolor="white", edgecolor="black", linewidth=1.2)
    for item in bp["whiskers"] + bp["caps"]:
        item.set(color="black", linewidth=1.0)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.5)

    rng = np.random.default_rng(42)
    for i, g in enumerate(groups, start=1):
        x = rng.normal(i, 0.05, size=len(g))
        ax.scatter(x, g, alpha=0.65, s=22, zorder=3)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.15)

    if p_text is not None:
        ymax = max(np.nanmax(g) for g in groups if len(g) > 0)
        ymin = min(np.nanmin(g) for g in groups if len(g) > 0)
        yline = ymax + (ymax - ymin) * 0.08
        h = (ymax - ymin) * 0.02
        ax.plot(
            [1, 1, len(groups), len(groups)],
            [yline, yline + h, yline + h, yline],
            color="black", linewidth=1,
        )
        ax.text(
            (1 + len(groups)) / 2, yline + h, p_text,
            ha="center", va="bottom", fontsize=10,
        )
