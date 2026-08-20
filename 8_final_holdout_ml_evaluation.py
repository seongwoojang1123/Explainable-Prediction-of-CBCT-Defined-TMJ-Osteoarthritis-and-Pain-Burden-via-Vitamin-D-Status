"""
8. Final Holdout ML Evaluation (Manuscript v2)
================================================
Reproduces the machine learning analysis reported in the manuscript:

  - Stratified 80:20 train/test split (internal holdout evaluation)
  - 5-fold stratified cross-validation within the training set
    (internal performance assessment during model development;
     hyperparameters were fixed, not tuned)
  - Median imputation inside each pipeline (no leakage)
  - Elastic Net: StandardScaler + L1/L2 mixing ratio 0.5
  - XGBoost: 80 estimators, max depth 3, learning rate 0.30,
    subsample 0.80, colsample_bytree 0.80, L2 regularization 1.0
  - Held-out test metrics: AUROC (bootstrap 95% CI, 2000 iterations),
    AUPRC, Brier score, calibration intercept/slope,
    sensitivity/specificity at threshold 0.50
  - DeLong paired ROC comparisons between nested models
  - Holm adjustment across the two principal comparisons
    (Block 2 vs Block 3, Elastic Net and XGBoost)
  - SHAP interpretation of the final Block-4 XGBoost model

Outputs (written to outputs/):
  - Table4_model_performance.csv
  - TableS3_delong_holm.csv
  - TableS4_shap_importance.csv
  - shap_long_table.csv          (raw plotting table for Figure 4 / S3)
  - heldout_predictions.csv      (per-patient test-set predictions)

NOTE BEFORE COMMITTING:
  Run this script on the original dataset and verify that the printed
  AUROC values match Table 4 of the manuscript (e.g., XGBoost Block 3
  AUROC = 0.834). Do not commit if the values do not reproduce.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, roc_curve,
)

from xgboost import XGBClassifier

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from config import OUTDIR, load_and_prepare_data

# ─────────────────────────────────────────────────────────────────────────────
# 0. Fixed settings (as reported in the manuscript Methods)
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42                # !! VERIFY: must match the seed of the final analysis
TEST_SIZE = 0.20
N_SPLITS = 5             # 5-fold stratified CV within the training set
THRESH = 0.50
N_BOOT = 2000            # bootstrap iterations for AUROC 95% CI

np.random.seed(SEED)

# Feature blocks (nested).
# !! VERIFY against the final analysis: in particular whether rheumatoid
#    factor entered as continuous "RF" or binary "RF_elevated01".
CLINICAL = ["Age", "Sex_female_bin", "Symptom_onset_log1p",
            "TMJ_noise", "Muscle_stiffness", "Jaw_locking", "Bruxism"]
LABS_WO_VITD = ["ESR", "RF_elevated01", "Zinc"]

BLOCKS = {
    "Clinical only":                      CLINICAL,
    "Clinical + labs without Vitamin D":  CLINICAL + LABS_WO_VITD,
    "Clinical + labs + Vitamin D":        CLINICAL + LABS_WO_VITD + ["VitaminD"],
    "Clinical + labs + Vitamin D + GSI":  CLINICAL + LABS_WO_VITD + ["VitaminD", "GSI"],
}
BLOCK_NAMES = list(BLOCKS.keys())

# Principal inferential comparisons (Holm adjustment applied across these two)
PRINCIPAL = [
    ("Elastic Net", BLOCK_NAMES[2], BLOCK_NAMES[1]),
    ("XGBoost",     BLOCK_NAMES[2], BLOCK_NAMES[1]),
]


def make_enet():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=0.5, C=1.0, max_iter=5000, random_state=SEED)),
    ])


def make_xgb():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.30,
            subsample=0.80, colsample_bytree=0.80, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            random_state=SEED, verbosity=0)),
    ])


MODELS = {"Elastic Net": make_enet, "XGBoost": make_xgb}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def calibration_intercept_slope(y_true, p):
    """Calibration slope: logistic recalibration on logit(p).
    Calibration intercept: intercept-only logistic model with logit(p)
    as offset (calibration-in-the-large)."""
    import statsmodels.api as sm
    eps = 1e-8
    logit = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    slope_fit = sm.GLM(y_true, sm.add_constant(logit),
                       family=sm.families.Binomial()).fit()
    slope = float(slope_fit.params[1])
    int_fit = sm.GLM(y_true, np.ones_like(logit),
                     family=sm.families.Binomial(), offset=logit).fit()
    intercept = float(int_fit.params[0])
    return intercept, slope


def sens_spec(y_true, p, thresh=THRESH):
    pred = (p >= thresh).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return sens, spec


def bootstrap_auroc_ci(y_true, p, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ─────────────────────────────────────────────────────────────────────────────
# 2. DeLong test (consistent sample estimators, ddof = 1)
# ─────────────────────────────────────────────────────────────────────────────

def _structural_components(y, p):
    pos, neg = p[y == 1], p[y == 0]
    n1, n0 = len(pos), len(neg)
    vp = np.array([((pi > neg).sum() + 0.5 * (pi == neg).sum()) / n0 for pi in pos])
    vn = np.array([((ni < pos).sum() + 0.5 * (ni == pos).sum()) / n1 for ni in neg])
    return vp, vn


def delong_p(y, p1, p2):
    vp1, vn1 = _structural_components(y, p1)
    vp2, vn2 = _structural_components(y, p2)
    n1, n0 = len(vp1), len(vn1)
    # sample variances/covariances (ddof=1) throughout
    v = (np.var(vp1, ddof=1) + np.var(vp2, ddof=1)
         - 2 * np.cov(vp1, vp2, ddof=1)[0, 1]) / n1 \
      + (np.var(vn1, ddof=1) + np.var(vn2, ddof=1)
         - 2 * np.cov(vn1, vn2, ddof=1)[0, 1]) / n0
    se = np.sqrt(max(v, 1e-12))
    z = (roc_auc_score(y, p1) - roc_auc_score(y, p2)) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def holm_adjust(pvals):
    """Holm step-down adjustment. Returns adjusted p in original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    df = load_and_prepare_data()
    needed = sorted({c for cols in BLOCKS.values() for c in cols} | {"TMJ_OA"})
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in dataset: {missing}")

    y_all = pd.to_numeric(df["TMJ_OA"], errors="coerce").values.astype(int)
    idx_all = np.arange(len(y_all))
    tr_idx, te_idx = train_test_split(
        idx_all, test_size=TEST_SIZE, stratify=y_all, random_state=SEED)
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]

    print(f"Total n={len(y_all)} | Train n={len(tr_idx)} | Test n={len(te_idx)}"
          f" | Test OA prevalence {y_te.mean():.3f}")

    rows, preds_store, pred_records = [], {}, []

    for algo, make_fn in MODELS.items():
        for block, feats in BLOCKS.items():
            X = df[feats].apply(pd.to_numeric, errors="coerce").values
            X_tr, X_te = X[tr_idx], X[te_idx]

            # 5-fold stratified CV within the training set
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=SEED)
            cv_aucs = []
            for f_tr, f_val in skf.split(X_tr, y_tr):
                m = make_fn()
                m.fit(X_tr[f_tr], y_tr[f_tr])
                cv_aucs.append(roc_auc_score(
                    y_tr[f_val], m.predict_proba(X_tr[f_val])[:, 1]))

            # final fit on full training set, evaluate once on held-out test
            final = make_fn()
            final.fit(X_tr, y_tr)
            p_te = final.predict_proba(X_te)[:, 1]
            preds_store[(algo, block)] = p_te

            ci_lo, ci_hi = bootstrap_auroc_ci(y_te, p_te)
            cint, cslope = calibration_intercept_slope(y_te, p_te)
            sens, spec = sens_spec(y_te, p_te)

            rows.append({
                "Algorithm": algo, "Model": block,
                "CV_AUROC_mean": np.mean(cv_aucs),
                "CV_AUROC_sd": np.std(cv_aucs, ddof=1),
                "AUROC": roc_auc_score(y_te, p_te),
                "AUROC_CI_low": ci_lo, "AUROC_CI_high": ci_hi,
                "AUPRC": average_precision_score(y_te, p_te),
                "Brier_score": brier_score_loss(y_te, p_te),
                "Calibration_intercept": cint, "Calibration_slope": cslope,
                "Sensitivity": sens, "Specificity": spec,
            })
            print(f"  {algo:<12s} {block:<36s} "
                  f"AUROC={rows[-1]['AUROC']:.3f} "
                  f"({ci_lo:.3f}-{ci_hi:.3f})")

            for pid, yt, pp in zip(te_idx, y_te, p_te):
                pred_records.append({"row_index": int(pid), "y_true": int(yt),
                                     "algorithm": algo, "block": block,
                                     "p_pred": float(pp)})

    table4 = pd.DataFrame(rows)
    table4.to_csv(OUTDIR / "Table4_model_performance.csv", index=False)
    pd.DataFrame(pred_records).to_csv(
        OUTDIR / "heldout_predictions.csv", index=False)

    # DeLong: adjacent nested comparisons + Block3-vs-Block1 style extras
    comparisons = []
    for algo in MODELS:
        pairs = [(BLOCK_NAMES[1], BLOCK_NAMES[0]),
                 (BLOCK_NAMES[2], BLOCK_NAMES[1]),
                 (BLOCK_NAMES[3], BLOCK_NAMES[2]),
                 (BLOCK_NAMES[2], BLOCK_NAMES[0]),
                 (BLOCK_NAMES[3], BLOCK_NAMES[0])]
        for curr, prev in pairs:
            p_curr, p_prev = preds_store[(algo, curr)], preds_store[(algo, prev)]
            comparisons.append({
                "Algorithm": algo,
                "Comparison": f"{curr} vs {prev}",
                "dAUROC": roc_auc_score(y_te, p_curr) - roc_auc_score(y_te, p_prev),
                "raw_p": delong_p(y_te, p_curr, p_prev),
                "Principal": (algo, curr, prev) in PRINCIPAL,
            })
    s3 = pd.DataFrame(comparisons)
    mask = s3["Principal"].values
    adj = np.full(len(s3), np.nan)
    adj[mask] = holm_adjust(s3.loc[mask, "raw_p"].values)
    s3["Holm_adjusted_p"] = adj
    s3.to_csv(OUTDIR / "TableS3_delong_holm.csv", index=False)

    print("\n[Principal comparisons — Holm adjusted]")
    for _, r in s3[s3["Principal"]].iterrows():
        print(f"  {r['Algorithm']}: dAUROC={r['dAUROC']:+.3f}, "
              f"raw p={r['raw_p']:.6f}, Holm p={r['Holm_adjusted_p']:.6f}")

    # SHAP on the final Block-4 XGBoost model
    if HAS_SHAP:
        feats = BLOCKS[BLOCK_NAMES[3]]
        X = df[feats].apply(pd.to_numeric, errors="coerce").values
        X_tr, X_te = X[tr_idx], X[te_idx]
        pipe = make_xgb()
        pipe.fit(X_tr, y_tr)
        X_te_imp = pipe.named_steps["imp"].transform(X_te)
        expl = shap.TreeExplainer(pipe.named_steps["clf"])
        sv = expl.shap_values(X_te_imp)
        mean_abs = np.abs(sv).mean(axis=0)
        imp = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs}) \
                .sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(OUTDIR / "TableS4_shap_importance.csv", index=False)
        long = pd.DataFrame({
            "feature": np.repeat(feats, len(te_idx)),
            "shap_value": sv.T.ravel(),
            "feature_value": X_te_imp.T.ravel(),
        })
        long.to_csv(OUTDIR / "shap_long_table.csv", index=False)
        print("\n[SHAP mean |value| — final Block-4 XGBoost]")
        print(imp.to_string(index=False))
    else:
        print("\nshap not installed — skipping SHAP outputs "
              "(pip install shap).")

    print("\nDone. Verify AUROC values against Table 4 before committing.")


if __name__ == "__main__":
    main()
