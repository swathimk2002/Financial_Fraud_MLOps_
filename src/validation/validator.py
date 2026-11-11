
"""Validate financial transaction datasets."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """Store dataset validation results."""

    is_valid: bool
    total_rows: int
    total_columns: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0

    def to_dict(self) -> dict:
        """Convert the report into a dictionary."""
        return {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "errors": self.errors,
            "warnings": self.warnings,
            "missing_values": self.missing_values,
            "duplicate_rows": self.duplicate_rows,
        }


REQUIRED_COLUMNS = {
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest",
    "newbalanceDest", "isFraud", "isFlaggedFraud",
}

NUMERIC_COLUMNS = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]

BALANCE_COLUMNS = [
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]

TRANSACTION_TYPES = {
    "PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT",
}


def validate_transactions(df: pd.DataFrame) -> ValidationReport:
    """Check transaction data without modifying the input."""

    errors = []
    warnings = []

    if df.empty:
        errors.append("Dataset is empty.")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        errors.append(f"Missing required columns: {sorted(missing_columns)}")

    missing_values = {
        column: int(count)
        for column, count in df.isnull().sum().items()
        if count > 0
    }

    if missing_values:
        errors.append("Dataset contains missing values.")

    duplicate_rows = int(df.duplicated().sum())

    if duplicate_rows:
        warnings.append(f"Dataset contains {duplicate_rows} duplicate rows.")

    # Check numerical columns and infinite values.
    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            errors.append(f"Column '{column}' must be numeric.")
            continue

        values = df[column].dropna()

        if not np.isfinite(values).all():
            errors.append(f"Column '{column}' contains infinite values.")

    # Check negative amounts and balances.
    if "amount" in df.columns and pd.api.types.is_numeric_dtype(df["amount"]):
        if (df["amount"].dropna() < 0).any():
            errors.append("Transaction amounts cannot be negative.")

    for column in BALANCE_COLUMNS:
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            if (df[column].dropna() < 0).any():
                errors.append(f"Column '{column}' contains negative balances.")

    # Check transaction categories.
    if "type" in df.columns:
        unexpected = set(df["type"].dropna().unique()) - TRANSACTION_TYPES

        if unexpected:
            warnings.append(f"Unexpected transaction types: {sorted(unexpected)}")

    # Check fraud labels.
    for column in ["isFraud", "isFlaggedFraud"]:
        if column in df.columns and not df[column].dropna().isin([0, 1]).all():
            errors.append(f"Column '{column}' must contain only 0 or 1.")

    return ValidationReport(
        is_valid=len(errors) == 0,
        total_rows=len(df),
        total_columns=len(df.columns),
        errors=errors,
        warnings=warnings,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
    )
