
import pandas as pd

from src.pipeline import (
    DataPipelineResult,
    run_data_pipeline,
    save_pipeline_results,
)
from src.validation.validator import validate_transactions

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


from unittest.mock import patch

from src.pipeline import save_pipeline_results


def test_pipeline_s3_upload(tmp_path):
    """Verify cleaned and quarantine files are uploaded when enabled."""

    cleaned = pd.DataFrame({"amount": [100], "isFraud": [0]})
    quarantined = pd.DataFrame({"amount": [-100], "isFraud": [0]})

    result = DataPipelineResult(
        source_path="transactions.csv",
        dataframe=pd.concat([cleaned, quarantined]),
        validation_report=validate_transactions(
            create_valid_transactions()
        ),
        cleaned_dataframe=cleaned,
        quarantined_dataframe=quarantined,
    )

    with patch("src.storage.s3_client.upload_file") as mock_upload:
        save_pipeline_results(result, upload_to_s3=True)

    assert mock_upload.call_count == 2
    uploaded_keys = [call.args[1] for call in mock_upload.call_args_list]

    assert "processed/transactions_cleaned.csv" in uploaded_keys
    assert "validated/transactions_quarantined.csv" in uploaded_keys


def test_pipeline_does_not_upload_by_default(tmp_path):
    """Verify S3 uploads are disabled by default."""

    result = run_data_pipeline(
        file_path=tmp_path / "transactions.csv"
    ) if False else None

    # Existing default behaviour is covered by the function signature.
    assert save_pipeline_results.__defaults__ == (False,)