"""EC5203 Part 2: Experimental feature engineering."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ClinicalFeatureEngineer(TransformerMixin, BaseEstimator):
    """Generate candidate features from original measurements."""

    REQUIRED_COLUMNS = [
        "age",
        "bmi",
        "cholesterol",
        "hdl",
        "triglycerides",
    ]

    def fit(self, X, y=None):
        self._validate_input(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        self._validate_input(X)

        result = X.copy()

        hdl = result["hdl"].replace(0, np.nan)

        result["cholesterol_hdl_ratio"] = (
            result["cholesterol"] / hdl
        )

        result["triglycerides_hdl_ratio"] = (
            result["triglycerides"] / hdl
        )

        result["non_hdl_cholesterol"] = (
            result["cholesterol"] - result["hdl"]
        )

        result["age_bmi_interaction"] = (
            result["age"] * result["bmi"]
        )

        engineered = [
            "cholesterol_hdl_ratio",
            "triglycerides_hdl_ratio",
            "non_hdl_cholesterol",
            "age_bmi_interaction",
        ]

        values = result[engineered].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "Engineered features contain NaN or infinite values."
            )

        return result

    def _validate_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        missing = [
            col for col in self.REQUIRED_COLUMNS
            if col not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )
