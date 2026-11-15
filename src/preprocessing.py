"""Dataset preparation for EC5203 Part 2."""

from pathlib import Path
import pandas as pd

TARGET = "sublabel"

EXCLUDED_COLUMNS = [
    "label",
    "disease_flags",
    "diabetes",
    "composite_key",
    "source_dataset",
    "hypertension",
    "heart_disease",
    "age_normalized",
    "age_level",
    "bmi_level",
    "blood_pressure",
    "high_blood_pressure",
    "low_hdl_cholesterol",
    "high_ldl_cholesterol",
]


def prepare_dataset(csv_path):
    """Load raw data and separate candidate predictors and target."""

    path = Path(csv_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required = [TARGET, *EXCLUDED_COLUMNS]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    X = df.drop(columns=[TARGET, *EXCLUDED_COLUMNS])
    y = df[TARGET].copy()

    if y.isna().any():
        raise ValueError("Target contains missing values.")

    if X.empty:
        raise ValueError("No predictors remain.")

    print("=== DATASET PREPARATION ===")
    print("Original shape:", df.shape)
    print("Predictor shape:", X.shape)
    print("Target shape:", y.shape)
    print("Exact duplicates:", df.duplicated().sum())
    print("Target classes:", y.nunique())
    print("\nClass distribution:")
    print(y.value_counts())

    return X, y


if __name__ == "__main__":
    X, y = prepare_dataset(
        "data/raw/Data Warehouse Multiclass.csv"
    )

def split_dataset(*args, **kwargs):
    """Deprecated: use split_three_ways instead."""
    raise RuntimeError(
        "The old 80/20 split is disabled. "
        "Use src.three_way_split.split_three_ways "
        "for the approved 70/15/15 split."
    )
