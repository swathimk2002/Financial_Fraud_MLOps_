
"""Create features available before a transaction is processed."""

import pandas as pd


REQUIRED_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create pre-transaction features from transaction data."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns for feature engineering: {missing_columns}"
        )

    # Keep the original data unchanged.
    features = df.copy()

    # Compare the transaction amount with the current origin balance.
    features["amount_to_orig_balance_ratio"] = (
        features["amount"]
        / (features["oldbalanceOrg"].abs() + 1)
    )

    # Compare the transaction amount with the current destination balance.
    features["amount_to_dest_balance_ratio"] = (
        features["amount"]
        / (features["oldbalanceDest"].abs() + 1)
    )

    return features
