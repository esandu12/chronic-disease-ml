# EC5203 - Part 2: Data Preprocessing and Feature Engineering

## 1. Dataset Information
- Original records: 280,985
- Original columns: 39
- Target column: sublabel
- Target classes: 8
- Exact duplicate rows removed: 2,284
- Remaining records: 278,701
- Original CSV preserved unchanged.

## 2. Dataset Splitting
Approved split: 70% Training / 15% Validation / 15% Testing

| Dataset | Records |
|---|---:|
| Training | 195,091 |
| Validation | 41,806 |
| Testing | 41,804 |

Random seed: 42

Method: Stratified grouped splitting.

Identical predictor records are kept in the same subset.
Patient-level independence cannot be verified.

## 3. Data Preprocessing
- Original candidate predictors: 24
- Numerical predictors: 16
- Categorical predictors: 8
- Missing values detected: 0
- Numerical imputation: Median
- Numerical scaling: RobustScaler
- Categorical imputation: Most frequent
- Categorical encoding: OneHotEncoder

All fitted preprocessing statistics come from training data.

## 4. Feature Engineering

Four experimental features were implemented:

| Feature | Formula |
|---|---|
| cholesterol_hdl_ratio | cholesterol / hdl |
| triglycerides_hdl_ratio | triglycerides / hdl |
| non_hdl_cholesterol | cholesterol - hdl |
| age_bmi_interaction | age * bmi |

These features require further validation of clinical definitions
and measurement units.

Pulse pressure was excluded because some records contain
inconsistent systolic and diastolic blood pressure measurements.

## 5. Final Feature Dimensions
- Original-feature pipeline: 38 encoded features
- Engineered-feature pipeline: 42 encoded features
- All transformed values passed finite-value checks.

## 6. Validation Experiment

Algorithm: Logistic Regression
Training records: 195,091
Validation records: 41,806

| Metric | Original | Engineered |
|---|---:|---:|
| Accuracy | 0.4634 | 0.4894 |
| Macro F1 | 0.2739 | 0.2961 |
| Weighted F1 | 0.5314 | 0.5552 |
| Balanced Accuracy | 0.4271 | 0.4739 |

Engineered features improved the reported validation metrics
in this experiment.

These are NOT final test-set results.

## 7. Known Limitations
- Strong class imbalance.
- No verified patient identifiers.
- 636 predictor-identical groups with conflicting targets
  were identified before the final split.
- 10,487 training records had systolic BP <= diastolic BP.
- Dataset clinical units remain unverified.
- IQR outliers were investigated but not automatically removed.
- Some excluded variables may contain target-related information.
- Clinical validity of engineered features is not established.

## 8. Reusable Python Modules

- src/preprocessing.py
- src/three_way_split.py
- src/feature_engineering.py
- src/pipeline.py

Correct integration:

    from src.preprocessing import prepare_dataset
    from src.three_way_split import split_three_ways
    from src.pipeline import build_pipeline

Use build_pipeline() inside each machine-learning model pipeline.

For cross-validation, preprocessing must be refitted within
each training fold.

The final test set must remain reserved for final evaluation.

## 9. Deliverables
- notebooks/02_preprocessing_feature_engineering.ipynb
- results/tables/part2_engineered_feature_catalogue.csv
- results/tables/part2_transformed_feature_catalogue.csv
- results/tables/part2_full_feature_comparison.csv
- results/figures/part2_feature_comparison.png
