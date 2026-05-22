# Association of Vitamin D Deficiency with TMJ Osteoarthritis and Pain Intensity: A Clinical and Machine Learning Approach

This repository contains the official analysis code for the study  
**"Association of Vitamin D Deficiency with TMJ Osteoarthritis and Pain Intensity"**  
by Yeon-Hee Lee, Seongwoo Jang, and colleagues.

The repository includes data preprocessing, statistical analysis, visualization, and machine learning evaluation modules for investigating the role of vitamin D in temporomandibular joint osteoarthritis (TMJ OA) presence, severity, and pain intensity.

This work primarily leverages multivariable regression approaches (binary logistic, ordinal logistic, and robust OLS) alongside machine learning models (Elastic Net, XGBoost) with SHAP-based interpretability, applied to clinical, laboratory, and psychological assessment data.

## Project Overview

Vitamin D deficiency has been implicated in various musculoskeletal and inflammatory conditions, yet its specific role in TMJ osteoarthritis remains underexplored.  
This project investigates:

- The association between vitamin D levels and TMJ OA **presence** (binary logistic regression)
- The association between vitamin D levels and TMJ OA **severity** (ordinal logistic regression)
- The relationship between vitamin D and **pain intensity** (VAS) with robust standard errors
- Predictive model performance comparing feature sets with and without vitamin D
- SHAP-based feature importance for model interpretability

Key aspects include:
- Comprehensive baseline comparison (Mann–Whitney U, Chi-squared, Fisher's exact) with Bonferroni correction
- Adjusted regression analyses controlling for age, sex, symptom duration, ESR, RF, zinc, GSI, and clinical symptoms
- Spearman correlation-based screening with significance-filtered heat maps
- Cross-validated Elastic Net and XGBoost model performance (AUROC, AUPRC, Brier score, calibration metrics)

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
  - Table 2: adjusted binary logistic regression (TMJ OA presence) and ordinal logistic regression (TMJ OA severity) for vitamin D (per 10 ng/mL) and vitamin D deficiency
  - Table 3: adjusted OLS regression with HC3-robust standard errors for VAS ~ vitamin D association
  - Both models include full covariate adjustment (age, sex, ESR, RF, zinc, GSI, clinical symptoms)

- **4_figure2_boxplots.py**
  - Figure 2: 3 × 2 box-and-scatter panel (Vitamin D, ESR, GSI)
  - Left column: Non-TMJ OA vs TMJ OA (Mann–Whitney U p-values)
  - Right column: Grade 0 vs 1 vs 2 (Kruskal–Wallis p-values)
  - Panel labels (a–f) with publication-quality formatting

- **5_figure5_vitaminD_VAS.py**
  - Figure 5: three-panel analysis of vitamin D and pain intensity
  - Panel A: VAS by vitamin D deficiency status (box-scatter)
  - Panel B: Vitamin D level by VAS group (≤5 vs ≥6)
  - Panel C: Scatter plot with linear fit and Spearman correlation

- **6_heatmap_correlation.py**
  - Spearman correlation screening (p < 0.05 threshold)
  - Heat map 1: factors significantly associated with VAS
  - Heat map 2: factors significantly associated with TMJ OA severity
  - Companion CSVs: screening results, full correlation matrices, p-value matrices

- **7_model_performance_and_SHAP.py**
  - Table 4: cross-validated model performance summary (Elastic Net, XGBoost)
  - Four incremental feature sets: Clinical only → +Labs → +Vitamin D → +GSI
  - Metrics: AUROC, AUPRC, Brier score, calibration intercept/slope, sensitivity, specificity
  - SHAP placeholder with instructions for reproducing Figure 4 when raw data becomes available

- **README.md**
  - Project overview, setup guide, and usage instructions

## Models Used

- **Statistical Models**
  - Binary logistic regression (TMJ OA presence)
  - Ordinal logistic regression (TMJ OA severity grading)
  - OLS regression with HC3-robust standard errors (VAS pain intensity)

- **Machine Learning Models**
  - Elastic Net: L1/L2 regularized logistic regression for sparse feature selection
  - XGBoost: gradient-boosted decision trees for nonlinear classification
  - Both models evaluated via stratified repeated k-fold cross-validation

- **Interpretability**
  - SHAP (SHapley Additive exPlanations) for feature importance and interaction analysis
  - Calibration assessment (intercept, slope) for clinical reliability

## Evaluation

- Performance Metrics
  - AUROC (with 95% CI)
  - AUPRC (area under precision–recall curve)
  - Brier score (calibration)
  - Calibration intercept and slope
  - Sensitivity and specificity

- Statistical Comparisons
  - Mann–Whitney U test for continuous variables
  - Chi-squared / Fisher's exact test for categorical variables
  - Kruskal–Wallis test for multi-group comparisons
  - Bonferroni correction for multiple testing
  - Spearman rank correlation with significance screening

## Requirements

```
numpy
pandas
scipy
statsmodels
matplotlib
```

Optional (for extended ML pipeline):
```
scikit-learn
xgboost
shap
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

# Step 7: Model performance summary & SHAP placeholder
python 7_model_performance_and_SHAP.py
```

## Contact

For any inquiries or collaboration opportunities:

- Yeon-Hee Lee: omod0209@gmail.com
- Seongwoo Jang: mook8105@koreacu.ac.kr

## Acknowledgments

- This work was supported by the Department of Orofacial Pain and Oral Medicine, Kyung Hee University Dental Hospital, and the Department of Convergence Information Studies, Korea Cyber University.
