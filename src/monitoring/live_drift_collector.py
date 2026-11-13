"""Collect live model features for data drift monitoring."""

from collections import deque
from threading import Lock

import pandas as pd


FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_orig_balance_ratio",
    "amount_to_dest_balance_ratio",
]


class LiveDriftCollector:
    """Validate and collect live model features in batches."""

    def __init__(self, batch_size: int = 500):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.batch_size = batch_size
        self._records = deque(maxlen=batch_size)
        self._lock = Lock()

    def collect(self, features: pd.DataFrame):
        """Collect feature records and return a completed batch when ready."""

        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame.")

        if features.empty:
            raise ValueError("Feature DataFrame cannot be empty.")

        missing_columns = [
            column
            for column in FEATURE_COLUMNS
            if column not in features.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required features: {missing_columns}"
            )

        feature_batch = features[FEATURE_COLUMNS].copy()

        if feature_batch.isnull().any().any():
            raise ValueError("Feature DataFrame contains missing values.")

        records = feature_batch.to_dict(orient="records")

        completed_batch = None

        with self._lock:
            self._records.extend(records)

            if len(self._records) == self.batch_size:
                completed_batch = pd.DataFrame(
                    list(self._records),
                    columns=FEATURE_COLUMNS,
                )
                self._records.clear()

        return completed_batch

    def pending_count(self) -> int:
        """Return the number of records waiting for a complete batch."""

        with self._lock:
            return len(self._records)
