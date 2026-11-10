
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
