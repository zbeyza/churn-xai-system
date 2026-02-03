from __future__ import annotations

"""Preprocessing for the churn dataset.

Keeps the feature engineering modest and predictable, so model behavior
is easy to reason about.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_FILE, TARGET_COL


def preprocess(
    data_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], pd.DataFrame]:
    """Load, clean, encode, and split the churn dataset."""
    path = DATA_FILE if data_path is None else Path(data_path)
    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found in CSV.")

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Map churn label to 1/0 for modeling.
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    if df[TARGET_COL].isna().any():
        raise ValueError(
            "Target column contains unexpected values; expected only 'Yes' or 'No'."
        )

    if "TotalCharges" in df.columns:
        # Coerce blank strings to NaN, then impute with median.
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    y = df[TARGET_COL]
    X_raw = df.drop(columns=[TARGET_COL])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # One-hot encode categoricals and align train/test columns.
    X_train = pd.get_dummies(X_train_raw, drop_first=False)
    X_test = pd.get_dummies(X_test_raw, drop_first=False)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    feature_names = list(X_train.columns)
    raw_test_df = X_test_raw.copy()

    return X_train, X_test, y_train, y_test, feature_names, raw_test_df
