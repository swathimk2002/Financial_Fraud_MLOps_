
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.loader import load_file
from src.validation.validator import (
    ValidationReport,
    validate_transactions,
)


@dataclass
class DataPipelineResult:
    """Contains the output of ingestion and validation."""

    source_path: str
    dataframe: pd.DataFrame
    validation_report: ValidationReport

    @property
    def is_valid(self) -> bool:
        """Return whether the dataset passed validation."""
        return self.validation_report.is_valid


def run_data_pipeline(
    file_path: str | Path,
    read_options: dict[str, Any] | None = None,
) -> DataPipelineResult:
    """
    Run ingestion followed by transaction data validation.

    The original loaded DataFrame is not modified.
    """

    path = Path(file_path)

    # Stage 1: Load the input file.
    dataframe = load_file(path, read_options=read_options)

    # Stage 2: Validate the loaded transaction data.
    validation_report = validate_transactions(dataframe)

    # Stage 3: Return the combined result.
    return DataPipelineResult(
        source_path=str(path),
        dataframe=dataframe,
        validation_report=validation_report,
    )

"""Run transaction ingestion, validation, and basic cleaning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.loader import load_file
from src.validation.validator import ValidationReport, validate_transactions


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

    # Stop if required columns are missing.
    if any("Missing required columns" in error
       for error in validation_report.errors):
     return DataPipelineResult(
        source_path=str(path),
        dataframe=dataframe,
        validation_report=validation_report,
        cleaned_dataframe=dataframe.iloc[0:0].copy(),
        quarantined_dataframe=dataframe.copy(),
    )

    # Identify invalid rows using the existing validation rules.
    invalid_mask = pd.Series(False, index=dataframe.index)

    for column, count in validation_report.missing_values.items():
        if count:
            invalid_mask |= dataframe[column].isna()

    for column in dataframe.columns:
        if column in ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
                      "oldbalanceDest", "newbalanceDest"]:
            numeric = pd.to_numeric(dataframe[column], errors="coerce")
            invalid_mask |= numeric.isna() | ~numeric.map(
                lambda value: pd.notna(value) and abs(value) != float("inf")
            )

    if "amount" in dataframe.columns:
        invalid_mask |= dataframe["amount"] < 0

    for column in ["oldbalanceOrg", "newbalanceOrig",
                   "oldbalanceDest", "newbalanceDest"]:
        invalid_mask |= dataframe[column] < 0

    for column in ["isFraud", "isFlaggedFraud"]:
        invalid_mask |= ~dataframe[column].isin([0, 1])

    if "type" in dataframe.columns:
        invalid_mask |= ~dataframe["type"].isin(
            ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
        )

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
    """Save cleaned transactions and quarantine records."""

    cleaned_path = Path("data/processed/transactions_cleaned.csv")
    quarantine_path = Path("data/validated/transactions_quarantined.csv")

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)

    result.cleaned_dataframe.to_csv(cleaned_path, index=False)

    if not result.quarantined_dataframe.empty:
        result.quarantined_dataframe.to_csv(quarantine_path, index=False)
