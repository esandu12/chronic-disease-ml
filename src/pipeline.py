"""Reusable preprocessing pipeline for EC5203."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.feature_engineering import ClinicalFeatureEngineer


def build_pipeline(X_train, use_engineering=True):
    """Build an unfitted preprocessing pipeline."""

    if use_engineering:
        preview = ClinicalFeatureEngineer().fit_transform(
            X_train.head(5)
        )
    else:
        preview = X_train

    numeric_columns = preview.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = preview.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    transformer = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    if use_engineering:
        return Pipeline([
            ("feature_engineering", ClinicalFeatureEngineer()),
            ("preprocessing", transformer)
        ])

    return Pipeline([
        ("preprocessing", transformer)
    ])
