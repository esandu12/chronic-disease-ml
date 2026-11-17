"""Decision Tree and SVM baseline model pipelines (Member 3, part B).

Mirrors src/models_baseline_a.py so that all four baselines share the
same preprocessing, feature-selection step and configuration options.
"""

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.pipeline import build_pipeline
from src.feature_selection import MutualInformationSelector


def _base_steps(X_reference, use_engineering, selected, k_features):
    steps = [
        (
            "preprocessor",
            build_pipeline(
                X_reference,
                use_engineering=use_engineering
            )
        )
    ]

    if selected:
        steps.append((
            "feature_selector",
            MutualInformationSelector(
                k=k_features,
                random_state=42
            )
        ))

    return steps


def build_decision_tree_pipeline(
    X_reference,
    use_engineering=False,
    selected=False,
    k_features=20,
    max_depth=12,
    min_samples_leaf=20
):
    steps = _base_steps(X_reference, use_engineering, selected, k_features)

    steps.append((
        "classifier",
        DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=42
        )
    ))

    return Pipeline(steps)


def build_svm_pipeline(
    X_reference,
    use_engineering=False,
    selected=False,
    k_features=20,
    C=1.0,
    kernel="rbf",
    gamma="scale"
):
    steps = _base_steps(X_reference, use_engineering, selected, k_features)

    steps.append((
        "classifier",
        SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight="balanced",
            random_state=42
        )
    ))

    return Pipeline(steps)
