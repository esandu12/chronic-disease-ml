
"""Feature selection utilities for the chronic disease prediction project."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils.validation import check_is_fitted


def correlation_screen(
    X: pd.DataFrame,
    threshold: float = 0.90
) -> pd.DataFrame:
    """
    Identify highly correlated numerical feature pairs.
    Use training data only.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    numeric = X.select_dtypes(include=np.number)

    if numeric.shape[1] < 2:
        return pd.DataFrame(
            columns=[
                "Feature_1",
                "Feature_2",
                "Absolute_Correlation"
            ]
        )

    correlation_matrix = numeric.corr().abs()

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1
        ).astype(bool)
    )

    rows = []

    for column in upper_triangle.columns:
        for row in upper_triangle.index:

            value = upper_triangle.loc[row, column]

            if pd.notna(value) and value >= threshold:
                rows.append({
                    "Feature_1": row,
                    "Feature_2": column,
                    "Absolute_Correlation": float(value)
                })

    return pd.DataFrame(
        rows,
        columns=[
            "Feature_1",
            "Feature_2",
            "Absolute_Correlation"
        ]
    )


def mutual_information_scores(
    X,
    y,
    feature_names=None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate mutual information scores for transformed features.
    """

    values = np.asarray(X)

    if values.ndim != 2:
        raise ValueError(
            "X must be a two-dimensional feature matrix"
        )

    if feature_names is None:
        feature_names = [
            f"feature_{i}"
            for i in range(values.shape[1])
        ]

    if len(feature_names) != values.shape[1]:
        raise ValueError(
            "Feature names must match the number of columns"
        )

    discrete_features = np.array([
        (
            np.unique(values[:, i]).size <= 20
            and np.allclose(
                values[:, i],
                np.round(values[:, i])
            )
        )
        for i in range(values.shape[1])
    ])

    scores = mutual_info_classif(
        values,
        np.asarray(y),
        discrete_features=discrete_features,
        random_state=random_state,
        n_neighbors=3
    )

    result = pd.DataFrame({
        "Feature": list(feature_names),
        "Mutual_Information": scores
    })

    return result.sort_values(
        by="Mutual_Information",
        ascending=False
    ).reset_index(drop=True)


def _mutual_information_score(X, y, random_state=42):
    """Scoring function used by SelectKBest."""

    values = np.asarray(X)

    discrete_features = np.array([
        (
            np.unique(values[:, i]).size <= 20
            and np.allclose(
                values[:, i],
                np.round(values[:, i])
            )
        )
        for i in range(values.shape[1])
    ])

    scores = mutual_info_classif(
        values,
        y,
        discrete_features=discrete_features,
        random_state=random_state,
        n_neighbors=3
    )

    return scores, None


class MutualInformationSelector(
    BaseEstimator,
    TransformerMixin
):
    """
    Select the top-k features using mutual information.

    The selector is fitted only on the training data
    when used inside a machine learning pipeline.
    """

    def __init__(
        self,
        k: int = 20,
        random_state: int = 42
    ):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):

        values = np.asarray(X)

        if values.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional matrix"
            )

        selected_k = min(
            int(self.k),
            values.shape[1]
        )

        score_function = partial(
            _mutual_information_score,
            random_state=self.random_state
        )

        self.selector_ = SelectKBest(
            score_func=score_function,
            k=selected_k
        )

        self.selector_.fit(values, y)

        self.n_features_in_ = values.shape[1]

        self.support_ = self.selector_.get_support()

        return self

    def transform(self, X):

        check_is_fitted(
            self,
            "selector_"
        )

        return self.selector_.transform(
            np.asarray(X)
        )

    def get_support(self):

        check_is_fitted(
            self,
            "selector_"
        )

        return self.selector_.get_support()
