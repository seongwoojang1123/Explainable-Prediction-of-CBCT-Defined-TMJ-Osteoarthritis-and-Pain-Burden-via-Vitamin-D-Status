# Serum Vitamin D Status in CBCT-Defined Temporomandibular Joint Osteoarthritis: Associations with Structural Severity and Pain Burden

This repository contains the official analysis code for the study
**"Serum Vitamin D Status in CBCT-Defined Temporomandibular Joint Osteoarthritis: Associations with Structural Severity and Pain Burden — A Retrospective Cross-Sectional Study"**
by Yeon-Hee Lee, Seongwoo Jang, and colleagues.

The repository includes data preprocessing, statistical analysis, visualization, and machine learning evaluation modules for investigating the role of serum vitamin D in temporomandibular joint osteoarthritis (TMJ OA) presence, severity, and pain intensity.

This work primarily leverages multivariable regression approaches (binary logistic, ordinal logistic, and robust OLS) alongside machine learning models (Elastic Net, XGBoost) with SHAP-based interpretability, applied to clinical, laboratory, and psychological assessment data.

## Project Overview

Low vitamin D status has been implicated in various musculoskeletal and inflammatory conditions, yet its specific role in TMJ osteoarthritis remains underexplored.
This project investigates:

- The association between vitamin D levels and CBCT-defined TMJ OA **presence** (binary logistic regression)
- The association between vitamin D levels and TMJ OA **severity** (ordinal logistic regression)
- The relationship between vitamin D and **pain intensity** (VAS) with robust standard errors
- Predictive model performance comparing nested feature sets with and without vitamin D, evaluated in a held-out test set
- SHAP-based feature importance for model interpretability

Continuous serum 25(OH)D is the primary exposure; low vitamin D status (< 30 ng/mL, encompassing deficiency and insufficiency) is analyzed as a secondary categorical exposure.

Key aspects include:
- Comprehensive baseline comparison (Mann–Whitney U, Chi-squared, Fisher's exact) with Bonferroni correction
- Adjusted regression analyses controlling for age, sex, symptom duration, ESR, RF, zinc, GSI, and clinical symptoms
- Spearman correlation-based screening with significance-filtered heat maps
- Internal holdout evaluation of Elastic Net and XGBoost models: a prespecified stratified 80:20 train-test split, with 5-fold stratified cross-validation inside the training set for internal performance assessment (model hyperparameters were fixed, not tuned), and all reported performance metrics computed in the held-out test set (AUROC with bootstrap 95% CI, AUPRC, Brier score, calibration intercept/slope, sensitivity, specificity)
- DeLong paired ROC comparisons between nested models, with Holm adjustment across the two principal vitamin D incremental comparisons

## Repository Structure

- **config.py**
  - Centralized configuration: file paths, variable mappings, display labels
  - Shared utility functions (descriptive statistics, statistical tests, plotting helpers)
  - Data loading and preprocessing pipeline (`load_and_prepare_data`)

- **1_data_preprocessing.py**
  - Entry point for data loading and preprocessing
  - Psychological variable renaming (SCL-90-R subscales)
  - Sex recoding (1/2 → 0/1), binary variable enforcement
  - GSI fallback derivation, log-transformed symptom duration
  - Exports cleaned dataset to `outputs/preprocessed_data.csv`

- **2_table1_baseline_characteristics.py**
  - Table 1: baseline characteristics stratified by TMJ OA status
  - Continuous variables compared via Mann–Whitney U test
  - Categorical variables compared via Chi-squared or Fisher's exact test
  - Bonferroni-adjusted p-values for multiple comparison correction

- **3_table2_table3_regression_analysis.py**
  - Table 2: adjusted binary logistic regression (TMJ OA presence) and ordinal logistic regression (TMJ OA severity) for vitamin D (per 10 ng/mL) and low vitamin D status
  - Table 3: adjusted OLS regression with HC3-robust standard errors for VAS ~ vitamin D association
  - Both models include full covariate adjustment (age, sex, ESR, RF, zinc, GSI, clinical symptoms)

- **4_figure2_boxplots.py**
  - Figure 2: 3 × 2 box-and-scatter panel (Vitamin D, ESR, GSI)
  - Left column: Non-TMJ OA vs TMJ OA (Mann–Whitney U p-values)
  - Right column: Grade 0 vs 1 vs 2 (Kruskal–Wallis p-values)

- **5_figure5_vitaminD_VAS.py**
  - Figure 5: three-panel analysis of vitamin D and pain intensity
  - Panel A: VAS by low vitamin D status (box-scatter)
  - Panel B: Vitamin D level by VAS group (≤5 vs ≥6)
  - Panel C: Scatter plot with linear fit and Spearman correlation

- **6_heatmap_correlation.py**
  - Spearman correlation screening (p < 0.05 threshold)
  - Heat maps of factors associated with VAS and with TMJ OA severity
 
- **7_model_performance_and_SHAP.py**
  - Formats Table 4 and the SHAP importance table from the outputs of script 8
  - Contains no manually entered performance values

- **8_final_holdout_ml_evaluation.py**
  - Full machine learning pipeline reported in the manuscript
  - Stratified 80:20 split; 5-fold stratified CV within the training set; median imputation inside each pipeline; Elastic Net (StandardScaler, l1_ratio = 0.5) and XGBoost (80 estimators, depth 3, learning rate 0.30, subsample 0.80, colsample 0.80, L2 = 1.0)
  - Held-out test metrics, bootstrap AUROC CIs (2000 iterations), calibration intercept/slope
  - DeLong paired comparisons with Holm adjustment for the two principal comparisons
  - SHAP analysis of the final Block-4 XGBoost model (importance table + raw plotting table)
  - Exports held-out per-patient predictions for full reproducibility


## Models Used

- **Statistical Models**
  - Binary logistic regression (TMJ OA presence)
  - Ordinal logistic regression (TMJ OA severity grading)
  - OLS regression with HC3-robust standard errors (VAS pain intensity)

- **Machine Learning Models**
  - Elastic Net: L1/L2 regularized logistic regression
  - XGBoost: gradient-boosted decision trees
  - Both models evaluated by internal holdout evaluation: a stratified 80:20 train-test split with 5-fold stratified cross-validation inside the training set; all reported metrics computed in the held-out test set

- **Interpretability**
  - SHAP (SHapley Additive exPlanations) for feature importance of the final Block-4 XGBoost model
  - Calibration assessment (intercept, slope)

## Evaluation

- Performance Metrics
  - AUROC (bootstrap 95% CI, 2000 iterations)
  - AUPRC (area under precision–recall curve)
  - Brier score
  - Calibration intercept and slope
  - Sensitivity and specificity (threshold 0.50)

- Statistical Comparisons
  - Mann–Whitney U / Chi-squared / Fisher's exact / Kruskal–Wallis tests
  - Bonferroni correction (baseline comparisons)
  - DeLong paired ROC comparisons; Holm adjustment across the two principal vitamin D incremental comparisons
  - Spearman rank correlation with significance screening

## Requirements

```
numpy
pandas
scipy
statsmodels
matplotlib
scikit-learn
xgboost
shap
openpyxl
```

## Usage

```bash
# Step 1: Preprocess data
python 1_data_preprocessing.py

# Step 2: Generate Table 1
python 2_table1_baseline_characteristics.py

# Step 3: Regression analyses (Tables 2 & 3)
python 3_table2_table3_regression_analysis.py

# Step 4: Figure 2 (box-scatter plots)
python 4_figure2_boxplots.py

# Step 5: Figure 5 (Vitamin D vs VAS)
python 5_figure5_vitaminD_VAS.py

# Step 6: Correlation heat maps
python 6_heatmap_correlation.py

# Step 7: Full ML pipeline (Table 4, Table S3, Table S4, SHAP)
python 8_final_holdout_ml_evaluation.py

# Step 8: Format the performance and SHAP tables
python 7_model_performance_and_SHAP.py
```

## Contact

For any inquiries or collaboration opportunities:

- Yeon-Hee Lee: omod0209@gmail.com
- Seongwoo Jang: mook8105@koreacu.ac.kr
- Seonggwang Jeon: qq22512@hanyang.ac.kr
- Tae-Seok Kim: taiseok11@naver.com
- Sang-woo Lee: goodman23@snu.ac.kr

## Acknowledgments

- This work was supported by the Department of Orofacial Pain and Oral Medicine, Kyung Hee University Dental Hospital, and the Department of Convergence Information Studies, Korea Cyber University.
