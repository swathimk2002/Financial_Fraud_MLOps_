import pandas as pd
import pytest

from src.ingestion.loader import load_file


def test_load_valid_csv(tmp_path):
    """Verify that a valid CSV loads correctly."""
    file_path = tmp_path / "transactions.csv"
    file_path.write_text("amount,isFraud\n100,0\n200,1\n", encoding="utf-8")

    df = load_file(file_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["amount", "isFraud"]


def test_missing_file():
    """Verify that a missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_file("nonexistent_file.csv")


def test_unsupported_extension(tmp_path):
    """Verify that unsupported file formats are rejected."""
    file_path = tmp_path / "transactions.txt"
    file_path.write_text("sample data", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_file(file_path)


def test_empty_dataset(tmp_path):
    """Verify that a CSV containing headers but no records is rejected."""
    file_path = tmp_path / "empty.csv"
    file_path.write_text("amount,isFraud\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no records"):
        load_file(file_path)


def test_custom_csv_read_options(tmp_path):
    """Verify that custom Pandas read options are supported."""
    file_path = tmp_path / "transactions.csv"
    file_path.write_text("amount;isFraud\n100;0\n200;1\n", encoding="utf-8")

    df = load_file(file_path, read_options={"sep": ";"})

    assert df.shape == (2, 2)
    assert list(df.columns) == ["amount", "isFraud"]

def test_load_valid_json(tmp_path):
    """Verify that a JSON array of records loads correctly."""
    file_path = tmp_path / "transactions.json"
    file_path.write_text(
        '[{"amount": 100, "isFraud": 0}, '
        '{"amount": 200, "isFraud": 1}]',
        encoding="utf-8",
    )

    df = load_file(file_path)

    assert df.shape == (2, 2)
    assert list(df.columns) == ["amount", "isFraud"]


def test_load_valid_jsonl(tmp_path):
    """Verify that JSON Lines records load correctly."""
    file_path = tmp_path / "transactions.jsonl"
    file_path.write_text(
        '{"amount": 100, "isFraud": 0}\n'
        '{"amount": 200, "isFraud": 1}\n',
        encoding="utf-8",
    )

    df = load_file(file_path)

    assert df.shape == (2, 2)
    assert df["isFraud"].tolist() == [0, 1]


def test_load_valid_parquet(tmp_path):
    """Verify that Parquet files load correctly."""
    file_path = tmp_path / "transactions.parquet"
    original_df = pd.DataFrame({
        "amount": [100, 200],
        "isFraud": [0, 1],
    })
    original_df.to_parquet(file_path, index=False)

    df = load_file(file_path)

    pd.testing.assert_frame_equal(df, original_df)


def test_load_valid_excel(tmp_path):
    """Verify that Excel files load correctly."""
    file_path = tmp_path / "transactions.xlsx"
    original_df = pd.DataFrame({
        "amount": [100, 200],
        "isFraud": [0, 1],
    })
    original_df.to_excel(file_path, index=False)

    df = load_file(file_path)

    pd.testing.assert_frame_equal(df, original_df)
