
"""Logistic Regression and KNN baseline model pipelines."""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from src.pipeline import build_pipeline
from src.feature_selection import MutualInformationSelector


def build_logistic_regression_pipeline(
    X_reference,
    use_engineering=False,
    selected=False,
    k_features=20
):
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

    steps.append((
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42
        )
    ))

    return Pipeline(steps)


def build_knn_pipeline(
    X_reference,
    use_engineering=False,
    selected=False,
    k_features=20,
    n_neighbors=7,
    weights="distance"
):
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

    steps.append((
        "classifier",
        KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric="minkowski",
            p=2,
            n_jobs=-1
        )
    ))

    return Pipeline(steps)
