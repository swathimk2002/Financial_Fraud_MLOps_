from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".xlsx",
    ".xls",
}


def load_file(
    file_path: str | Path,
    read_options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Load a supported local data file into a Pandas DataFrame.

    Supported formats:
        CSV, JSON, JSONL, NDJSON, Parquet, XLSX and XLS.
    """
    path = Path(file_path)
    options = read_options or {}

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if extension == ".csv":
        dataframe = pd.read_csv(path, **options)

    elif extension in {".jsonl", ".ndjson"}:
        dataframe = pd.read_json(path, lines=True, **options)

    elif extension == ".json":
        dataframe = pd.read_json(path, **options)

    elif extension == ".parquet":
        dataframe = pd.read_parquet(path, **options)

    elif extension in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path, **options)

    else:
        raise ValueError(f"No loader implemented for: {extension}")

    if dataframe.empty:
        raise ValueError(f"The input file contains no records: {path}")

    return dataframe
