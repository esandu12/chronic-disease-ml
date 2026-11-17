# EC5203 — Member 3 (part B) Handoff: Feature Selection & Decision Tree / SVM Baselines

Continues part A (`notebooks/03_feature_selection.ipynb`, `src/models_baseline_a.py`:
correlation screen, mutual information, Logistic Regression, KNN).

## Added in part B
- `src/models_baseline_b.py`: `build_decision_tree_pipeline()`, `build_svm_pipeline()`
  (same signature/options as part A: `use_engineering`, `selected`, `k_features`)
- `notebooks/03b_feature_selection_baseline_b.ipynb`
- Feature selection method 3: Decision Tree importance → `tree_importance_scores.csv`
- Feature selection method 4: RFE (Decision Tree, k=20) → `rfe_ranking.csv`
- Agreement of MI / tree importance / RFE → `feature_selection_agreement.csv`
- Part B metrics → `baseline_b_validation_metrics.csv`, `class_wise_validation_metrics_b.csv`
- All four baselines → `all_baseline_validation_metrics.csv`
- Settings → `baseline_b_settings.json`

## Protocol (same as part A)
- `prepare_dataset()` + `split_three_ways()` (70/15/15 grouped, seed 42)
- Preprocessing: `src.pipeline.build_pipeline` (fitted on training data only)
- Configurations: Original / Engineered / Engineered_Selected (MI, k=20, selector inside the pipeline)
- Fit on training data, evaluate on validation. **Test set not used.**
- Decision Tree: full training set (195,091 rows), `max_depth=12, min_samples_leaf=20, class_weight="balanced"`
- SVM: RBF, `C=1, gamma="scale", class_weight="balanced"`, stratified 10,000-row training subset
  (RBF SVM does not scale to 195k rows; same approach as part A's KNN subset)
- RFE fitted on a stratified 20,000-row training subset

## Validation results (macro F1)

| Model | Original | Engineered | Engineered_Selected |
|---|---:|---:|---:|
| Decision Tree | 0.3687 | **0.3702** | 0.3675 |
| SVM (RBF) | 0.3605 | 0.3603 | 0.3408 |
| Logistic Regression (part A) | 0.2739 | 0.2962 | 0.2840 |
| KNN (part A) | 0.2872 | 0.2855 | 0.2810 |

Best baseline: **Decision Tree, Engineered** (accuracy 0.647, weighted F1 0.664, macro F1 0.370).

## Feature selection findings
- 16 numeric features were chosen by all three methods (MI, tree importance, RFE):
  HbA1c_level, age, alcohol_intake, bmi, cholesterol, crp_level, diastolic_bp, glucose, hdl,
  heart_rate, homocysteine_level, ldl, salt_intake, sleep_hours, systolic_bp, triglycerides.
- Top tree importance: HbA1c_level, alcohol_intake, systolic_bp, homocysteine_level, age.
- Engineered `age_bmi_interaction` and `triglycerides_hdl_ratio` were picked by tree importance and RFE
  (MI in part A was computed on the original features only, so it could not pick them).
- Categorical one-hot features were rarely selected.
- Selecting 20 features did not improve any baseline; it kept performance close with fewer features.

## Limitations
- Strong class imbalance: rare classes (HT, HT_HY, DI_HT, DI_HT_HY) have F1 around 0.11–0.18.
- SVM trained on a 10k subset; RFE on a 20k subset.
- No hyperparameter tuning yet (Member 4).
- Clinical units unverified; some disease indicators excluded as possible leakage.

## For Member 4
- Use the same split, preprocessing, configurations and metrics.
- Final five-algorithm comparison on the **test** set: LR, KNN, Decision Tree, SVM + ensembles.
