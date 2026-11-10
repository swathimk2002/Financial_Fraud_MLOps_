
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplit:
    """Stores the three dataset splits."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


TARGET_COLUMN = "isFraud"

EXCLUDED_COLUMNS = [
    "nameOrig",
    "nameDest",
    "isFlaggedFraud",
    "newbalanceOrig",
    "newbalanceDest",
]


def split_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    random_state: int = 42,
) -> DatasetSplit:
    """Split the data into training, validation and testing sets."""

    # Check whether the target column exists.
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' is missing."
        )

    # Check whether the target has two classes.
    if df[target_column].nunique() != 2:
        raise ValueError(
            "Target column must contain exactly two classes."
        )

    # Remove columns that should not be used as features.
    excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column in df.columns
    ]

    # Separate input features and target.
    X = df.drop(columns=[target_column, *excluded])
    y = df[target_column]

    # Split 70% for training and 30% for temporary data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    # Split the temporary data equally into validation and testing.
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    # Return all three splits.
    return DatasetSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )
