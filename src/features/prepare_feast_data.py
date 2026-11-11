
"""Prepare transaction features for the Feast feature store."""

from pathlib import Path

import pandas as pd

from src.features.engineer import engineer_features


# Project root: D:\Financial_Fraud_MLOps
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input dataset (original sample remains unchanged).
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "samples"
    / "paysim_sample_200k.csv"
)

# Output directory for Feast feature data.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
)

OUTPUT_FILE = OUTPUT_DIR / "transaction_features.parquet"

# Fixed anchor used to create synthetic timestamps.
TIMESTAMP_ANCHOR = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")

# Exact features expected by the existing model.
MODEL_FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_orig_balance_ratio",
    "amount_to_dest_balance_ratio",
]


def prepare_feast_data() -> pd.DataFrame:
    """Prepare and save the transaction feature source."""

    print("Loading original sample dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("Input dataset is empty.")

    print(f"Loaded {len(df):,} transactions.")

    # Create IDs before sorting so each original row keeps its identity.
    df.insert(
        0,
        "transaction_id",
        [f"txn_{i:09d}" for i in range(len(df))],
    )

    # Create synthetic event timestamps from PaySim simulation steps.
    df["event_timestamp"] = (
        TIMESTAMP_ANCHOR
        + pd.to_timedelta(df["step"], unit="h")
    )

    print("Generating existing model features...")

    # Reuse the exact feature-engineering function used by the project.
    features = engineer_features(df)

    # Keep only the entity, event timestamp and model features.
    feast_columns = [
        "transaction_id",
        "event_timestamp",
        *MODEL_FEATURES,
    ]

    features = features[feast_columns].copy()

    # Sort chronologically while retaining transaction IDs.
    features = features.sort_values(
        by=["event_timestamp", "transaction_id"],
        kind="stable",
    ).reset_index(drop=True)

    # Basic integrity checks.
    if features["transaction_id"].isna().any():
        raise ValueError("Transaction IDs contain missing values.")

    if features["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs are not unique.")

    if features["event_timestamp"].isna().any():
        raise ValueError("Event timestamps contain missing values.")

    if features[MODEL_FEATURES].isna().any().any():
        raise ValueError("Model features contain missing values.")

    numeric_features = features.select_dtypes(include="number")

    

    if numeric_features.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Infinite numerical feature values detected.")

    # Create the output directory if it does not exist.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save a separate Parquet file; never overwrite the original CSV.
    features.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
    )

    print("\n========== FEAST DATA PREPARATION COMPLETE ==========")
    print(f"Rows: {len(features):,}")
    print(f"Columns: {features.columns.tolist()}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Minimum timestamp: {features['event_timestamp'].min()}")
    print(f"Maximum timestamp: {features['event_timestamp'].max()}")
    print("Transaction IDs unique: True")
    print("Sorted by event timestamp: True")

    return features


if __name__ == "__main__":
    prepare_feast_data()
