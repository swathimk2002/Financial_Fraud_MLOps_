from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationReport:
    """Stores the results of dataset validation."""

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
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}


def validate_transactions(df: pd.DataFrame) -> ValidationReport:
    """
    Validate a transaction dataset without modifying the input DataFrame.
    """

    errors = []
    warnings = []

    if df.empty:
        errors.append("Dataset is empty.")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        errors.append(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    missing_values = {
        column: int(count)
        for column, count in df.isnull().sum().items()
        if count > 0
    }

    if missing_values:
        errors.append("Dataset contains missing values.")

    duplicate_rows = int(df.duplicated().sum())

    if duplicate_rows > 0:
        warnings.append(
            f"Dataset contains {duplicate_rows} duplicate rows."
        )

    if "amount" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["amount"]):
            errors.append("Column 'amount' must be numeric.")
        elif (df["amount"] < 0).any():
            errors.append("Transaction amounts cannot be negative.")

    if "isFraud" in df.columns:
        if not df["isFraud"].isin([0, 1]).all():
            errors.append("Column 'isFraud' must contain only 0 or 1.")

    if "isFlaggedFraud" in df.columns:
        if not df["isFlaggedFraud"].isin([0, 1]).all():
            errors.append(
                "Column 'isFlaggedFraud' must contain only 0 or 1."
            )

    return ValidationReport(
        is_valid=len(errors) == 0,
        total_rows=len(df),
        total_columns=len(df.columns),
        errors=errors,
        warnings=warnings,
        missing_values=missing_values,
        duplicate_rows=duplicate_rows,
    )
