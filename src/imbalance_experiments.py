"""Class-imbalance and tuning experiments for Decision Tree and SVM (Package B).

All resampling happens inside an imbalanced-learn Pipeline, so it is applied
only to the data passed to ``fit`` (training data or CV training folds).
Validation and test data are never resampled.
"""

from __future__ import annotations

import time

import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.pipeline import build_pipeline

RARE_CLASSES = ["HT", "HT_HY", "DI_HT", "DI_HT_HY"]

STRATEGIES = ["unweighted", "class_weight", "random_oversample", "smote"]


def _classifier(model, class_weight, params):
    if model == "decision_tree":
        return DecisionTreeClassifier(
            class_weight=class_weight, random_state=42, **params
        )
    if model == "svm":
        return SVC(kernel="rbf", class_weight=class_weight, random_state=42, **params)
    raise ValueError(f"Unknown model: {model}")


def build_imbalance_pipeline(
    model,
    X_reference,
    strategy="class_weight",
    use_engineering=True,
    params=None,
    random_state=42,
):
    """Preprocessing -> optional training-only resampler -> classifier."""

    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")

    params = dict(params or {})
    # imbalanced-learn does not accept a nested Pipeline as a step,
    # so the shared preprocessing steps are copied in directly (same objects/settings).
    steps = list(build_pipeline(X_reference, use_engineering=use_engineering).steps)

    if strategy == "random_oversample":
        steps.append(("sampler", RandomOverSampler(random_state=random_state)))
    elif strategy == "smote":
        steps.append(("sampler", SMOTE(random_state=random_state, k_neighbors=5)))

    class_weight = "balanced" if strategy == "class_weight" else None
    steps.append(("classifier", _classifier(model, class_weight, params)))

    return ImbPipeline(steps)


def tune_with_cv(pipeline, param_grid, X, y, n_splits=3, random_state=42, n_jobs=-1):
    """Bounded grid search on training data only, scored by macro F1."""

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring={
            "macro_f1": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
            "weighted_f1": "f1_weighted",
        },
        refit="macro_f1",
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=False,
    )
    start = time.perf_counter()
    search.fit(X, y)
    runtime = time.perf_counter() - start

    results = pd.DataFrame(search.cv_results_)
    keep = [c for c in results.columns if c.startswith("param_")] + [
        "mean_test_macro_f1", "std_test_macro_f1",
        "mean_test_balanced_accuracy", "mean_test_weighted_f1",
        "mean_fit_time", "rank_test_macro_f1",
    ]
    results = results[keep].sort_values("rank_test_macro_f1").reset_index(drop=True)
    return search, results, runtime


def evaluate_on_validation(pipeline, X_train, y_train, X_val, y_val, model_name, strategy):
    """Fit on training data, score on validation; includes rare-class recall."""

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    pred = pipeline.predict(X_val)

    labels = sorted(pd.unique(y_val))
    recalls = recall_score(y_val, pred, labels=labels, average=None, zero_division=0)
    recall_by_class = dict(zip(labels, recalls))

    row = {
        "Model": model_name,
        "Strategy": strategy,
        "Training_Rows": len(y_train),
        "Accuracy": accuracy_score(y_val, pred),
        "Macro_F1": f1_score(y_val, pred, average="macro", zero_division=0),
        "Weighted_F1": f1_score(y_val, pred, average="weighted", zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_val, pred),
        "Rare_Class_Mean_Recall": sum(recall_by_class[c] for c in RARE_CLASSES if c in recall_by_class)
        / len(RARE_CLASSES),
        "Fit_Seconds": round(fit_seconds, 1),
    }
    for c in RARE_CLASSES:
        row[f"Recall_{c}"] = recall_by_class.get(c, 0.0)

    report = pd.DataFrame(
        classification_report(y_val, pred, output_dict=True, zero_division=0)
    ).transpose()
    report["Model"] = model_name
    report["Strategy"] = strategy
    return row, report, pred
