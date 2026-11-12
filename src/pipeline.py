
"""Run transaction ingestion, validation, cleaning, and quarantine."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ingestion.loader import load_file
from src.validation.validator import ValidationReport, validate_transactions


NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

BALANCE_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

VALID_TRANSACTION_TYPES = {
    "PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"
}

LABEL_COLUMNS = ["isFraud", "isFlaggedFraud"]


@dataclass
class DataPipelineResult:
    """Store the results of the data pipeline."""

    source_path: str
    dataframe: pd.DataFrame
    validation_report: ValidationReport
    cleaned_dataframe: pd.DataFrame
    quarantined_dataframe: pd.DataFrame

    @property
    def is_valid(self) -> bool:
        """Return whether the dataset passed validation."""
        return self.validation_report.is_valid


def run_data_pipeline(
    file_path: str | Path,
    read_options: dict[str, Any] | None = None,
) -> DataPipelineResult:
    """Load, validate, and separate valid and invalid transactions."""

    path = Path(file_path)
    dataframe = load_file(path, read_options=read_options)
    validation_report = validate_transactions(dataframe)

    # If required columns are missing, quarantine the entire dataset.
    if any(
        "Missing required columns" in error
        for error in validation_report.errors
    ):
        return DataPipelineResult(
            source_path=str(path),
            dataframe=dataframe,
            validation_report=validation_report,
            cleaned_dataframe=dataframe.iloc[0:0].copy(),
            quarantined_dataframe=dataframe.copy(),
        )

    invalid_mask = pd.Series(False, index=dataframe.index)

    # Quarantine rows containing missing values.
    invalid_mask |= dataframe.isna().any(axis=1)

    # Check numeric columns and infinite values.
    for column in NUMERIC_COLUMNS:
        numeric_values = pd.to_numeric(
            dataframe[column], errors="coerce"
        )
        invalid_mask |= (
            numeric_values.isna()
            | ~np.isfinite(numeric_values)
        )

    # Reject negative amounts and balances.
    invalid_mask |= dataframe["amount"] < 0

    for column in BALANCE_COLUMNS:
        invalid_mask |= dataframe[column] < 0

    # Check fraud labels.
    for column in LABEL_COLUMNS:
        invalid_mask |= ~dataframe[column].isin([0, 1])

    # Check transaction types.
    invalid_mask |= ~dataframe["type"].isin(VALID_TRANSACTION_TYPES)

    cleaned_dataframe = dataframe.loc[~invalid_mask].copy()
    quarantined_dataframe = dataframe.loc[invalid_mask].copy()

    return DataPipelineResult(
        source_path=str(path),
        dataframe=dataframe,
        validation_report=validation_report,
        cleaned_dataframe=cleaned_dataframe,
        quarantined_dataframe=quarantined_dataframe,
    )


def save_pipeline_results(result: DataPipelineResult) -> None:
    """Save cleaned transactions and quarantined records."""

    cleaned_path = Path("data/processed/transactions_cleaned.csv")
    quarantine_path = Path("data/validated/transactions_quarantined.csv")

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)

    result.cleaned_dataframe.to_csv(cleaned_path, index=False)

    if not result.quarantined_dataframe.empty:
        result.quarantined_dataframe.to_csv(
            quarantine_path, index=False
        )
