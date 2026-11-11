
import pandas as pd

from src.pipeline import run_data_pipeline


def create_valid_transactions():
    """Create a small valid PaySim transaction dataset."""
    return pd.DataFrame({
        "step": [1, 2],
        "type": ["PAYMENT", "TRANSFER"],
        "amount": [100.0, 200.0],
        "nameOrig": ["C1", "C2"],
        "oldbalanceOrg": [500.0, 600.0],
        "newbalanceOrig": [400.0, 400.0],
        "nameDest": ["M1", "C3"],
        "oldbalanceDest": [0.0, 100.0],
        "newbalanceDest": [100.0, 300.0],
        "isFraud": [0, 1],
        "isFlaggedFraud": [0, 0],
    })


def test_pipeline_with_valid_csv(tmp_path):
    dataframe = create_valid_transactions()
    file_path = tmp_path / "valid_transactions.csv"
    dataframe.to_csv(file_path, index=False)

    result = run_data_pipeline(file_path)

    assert result.is_valid is True
    assert len(result.dataframe) == 2
    assert result.validation_report.errors == []
    assert result.source_path == str(file_path)


def test_pipeline_with_invalid_csv(tmp_path):
    dataframe = create_valid_transactions()
    dataframe = dataframe.drop(columns=["amount"])

    file_path = tmp_path / "invalid_transactions.csv"
    dataframe.to_csv(file_path, index=False)

    result = run_data_pipeline(file_path)

    assert result.is_valid is False
    assert len(result.dataframe) == 2
    assert any(
        "Missing required columns" in error
        for error in result.validation_report.errors
    )


def test_pipeline_with_missing_file():
    import pytest

    with pytest.raises(FileNotFoundError):
        run_data_pipeline("data/samples/nonexistent.csv")

def test_pipeline_quarantines_invalid_rows(tmp_path):
    """Separate invalid transactions from valid transactions."""
    dataframe = create_valid_transactions()
    dataframe.loc[0, "amount"] = -100
    dataframe.loc[1, "amount"] = None

    file_path = tmp_path / "transactions_with_invalid_rows.csv"
    dataframe.to_csv(file_path, index=False)

    result = run_data_pipeline(file_path)

    assert len(result.cleaned_dataframe) == 0
    assert len(result.quarantined_dataframe) == 2
    assert len(result.dataframe) == 2
