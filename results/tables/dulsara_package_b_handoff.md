# EC5203 — Package B Handoff (Dulsara): Additional Baselines, Class Imbalance & Tuning

## Deliverables
- `src/models_baseline_b.py` — Decision Tree and SVM pipelines (same interface as `models_baseline_a.py`)
- `src/imbalance_experiments.py` — fold-safe imbalance pipelines (imbalanced-learn), CV tuning, validation evaluation
- `notebooks/03b_feature_selection_baseline_b.ipynb` — first Decision Tree / SVM results on Original / Engineered / Engineered_Selected
  (also contains tree-importance and RFE checks that support Praboda's feature-selection work)
- `notebooks/04_baselines_imbalance_dulsara.ipynb` — tuning + imbalance comparison
- `results/tables/dulsara_cv_tuning.csv`, `dulsara_imbalance_comparison.csv`,
  `dulsara_class_wise_validation_metrics.csv`, `dulsara_settings.json`
- `results/figures/04_dulsara_imbalance_comparison.png`, `04_dulsara_confusion_matrices.png`

## Protocol
- `prepare_dataset()` → `split_three_ways()` (70/15/15 grouped, seed 42) → `build_pipeline(use_engineering=True)`
- Decision Tree: full training set (195,091 rows). SVM (RBF): stratified 10,000-row training subset
  (documented simplification: RBF SVM training cost grows ~quadratically with rows).
- Resampling (RandomOverSampler, SMOTE) sits inside an imbalanced-learn Pipeline → applied only to training data / CV training folds.
- Tuning: 3-fold stratified CV on training data, scored by macro F1. Validation used only for the imbalance comparison.
- **Test set not used.** Random seed 42 everywhere.

## Bounded CV tuning (class-weighted)
| Model | Grid | Best | CV macro F1 | Runtime |
|---|---|---|---:|---:|
| Decision Tree | max_depth {8,12,16,20} × min_samples_leaf {1,20,50} | max_depth=16, min_samples_leaf=1 | 0.3755 | 43 s |
| SVM (RBF) | C {0.3,1,3,10} × gamma {scale,0.01,0.1} | C=3, gamma=scale | 0.3510 | 47 s |

## Imbalance comparison (validation, tuned settings)
| Model | Strategy | Macro F1 | Balanced acc. | Accuracy | Mean recall of 4 rare classes |
|---|---|---:|---:|---:|---:|
| Decision Tree | smote | **0.377** | 0.496 | 0.655 | 0.380 |
| Decision Tree | random_oversample | **0.377** | 0.498 | 0.656 | 0.381 |
| Decision Tree | class_weight | 0.371 | 0.482 | 0.638 | 0.358 |
| Decision Tree | unweighted | 0.359 | 0.365 | 0.751 | 0.041 |
| SVM (RBF) | smote | 0.369 | 0.440 | 0.656 | 0.257 |
| SVM (RBF) | class_weight | 0.364 | 0.441 | 0.643 | 0.266 |
| SVM (RBF) | random_oversample | 0.359 | 0.438 | 0.631 | 0.270 |
| SVM (RBF) | unweighted | 0.329 | 0.339 | 0.738 | 0.001 |

Rare classes = HT, HT_HY, DI_HT, DI_HT_HY.

## Findings
- Without imbalance handling both models almost never predict the rare classes (rare-class recall 0.04 and 0.00),
  even though accuracy looks highest (0.75 / 0.74). Accuracy alone is misleading here.
- Class weighting or training-only oversampling raises rare-class recall from ~0 to 0.27–0.38 and increases macro F1.
- Best Package B configuration: **Decision Tree (max_depth=16, min_samples_leaf=1) with training-only random oversampling
  or SMOTE — validation macro F1 0.377**. Random oversampling is simpler and faster (12 s vs 28 s fit).
- Best SVM: RBF, C=3, gamma=scale, with SMOTE — macro F1 0.369.
- Rare classes still have low precision: the model trades many false positives for recall.

## Limitations
- SVM trained on a 10k subset; bounded 3-fold grids only.
- Tuning was done with class weighting; the tuned settings were reused for the other strategies.
- Strong class imbalance; no verified patient IDs; clinical units unverified.

## For Esandu (final comparison)
Candidate settings to freeze from Package B:
- Decision Tree: max_depth=16, min_samples_leaf=1, random oversampling (training only)
- SVM (RBF): C=3, gamma=scale, SMOTE (training only), 10k training subset
