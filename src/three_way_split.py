"""EC5203: approximately 70/15/15 grouped dataset split."""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.preprocessing import prepare_dataset


def split_three_ways(X, y, random_state=42):
    df = X.copy()
    df["__target__"] = y.to_numpy()

    before = len(df)
    df = df.drop_duplicates().copy()
    removed = before - len(df)

    X = df.drop(columns="__target__")
    y = df["__target__"]

    groups = pd.util.hash_pandas_object(X, index=False)

    splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=random_state
    )

    # Allocate seven folds to training and three to holdout.
    fold_ids = pd.Series(index=X.index, dtype="int64")

    for fold, (_, test_idx) in enumerate(
        splitter.split(X, y, groups)
    ):
        fold_ids.iloc[test_idx] = fold

    train_mask = fold_ids < 7
    holdout_mask = ~train_mask

    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()

    X_holdout = X.loc[holdout_mask].copy()
    y_holdout = y.loc[holdout_mask].copy()

    holdout_groups = groups.loc[holdout_mask]

    inner = StratifiedGroupKFold(
        n_splits=2,
        shuffle=True,
        random_state=random_state + 1
    )

    val_idx, test_idx = next(
        inner.split(X_holdout, y_holdout, holdout_groups)
    )

    X_val = X_holdout.iloc[val_idx].copy()
    X_test = X_holdout.iloc[test_idx].copy()
    y_val = y_holdout.iloc[val_idx].copy()
    y_test = y_holdout.iloc[test_idx].copy()

    print("\n=== DUPLICATE HANDLING ===")
    print("Original records:", before)
    print("Exact duplicate records removed:", removed)
    print("Remaining records:", len(X))

    print("\n=== 70/15/15 DATA SPLIT ===")
    for name, part in [
        ("Training", X_train),
        ("Validation", X_val),
        ("Testing", X_test)
    ]:
        print(
            f"{name}: {len(part)} "
            f"({len(part) / len(X) * 100:.2f}%)"
        )

    summary = pd.DataFrame({
        "Original": y.value_counts(normalize=True),
        "Training": y_train.value_counts(normalize=True),
        "Validation": y_val.value_counts(normalize=True),
        "Testing": y_test.value_counts(normalize=True)
    }).fillna(0) * 100

    print("\n=== CLASS DISTRIBUTION (%) ===")
    print(summary.round(2))

    def fingerprints(frame):
        return set(
            pd.util.hash_pandas_object(frame, index=False)
        )

    a = fingerprints(X_train)
    b = fingerprints(X_val)
    c = fingerprints(X_test)

    assert not (a & b or a & c or b & c)
    assert set(y_train) == set(y_val) == set(y_test) == set(y)

    print("\nSUCCESS: No overlapping predictor groups.")
    print("SUCCESS: All eight classes represented.")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X, y = prepare_dataset(
        "data/raw/Data Warehouse Multiclass.csv"
    )
    split_three_ways(X, y)
