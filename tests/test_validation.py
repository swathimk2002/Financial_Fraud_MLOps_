
import pandas as pd
import pytest

from src.validation.validator import validate_transactions


@pytest.fixture
def valid_transactions():
    """Create a small valid PaySim-like transaction dataset."""
    return pd.DataFrame({
        "step": [1, 2, 3],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "amount": [100.0, 200.0, 300.0],
        "nameOrig": ["C1", "C2", "C3"],
        "oldbalanceOrg": [500.0, 600.0, 700.0],
        "newbalanceOrig": [400.0, 400.0, 400.0],
        "nameDest": ["M1", "C4", "C5"],
        "oldbalanceDest": [0.0, 100.0, 200.0],
        "newbalanceDest": [100.0, 300.0, 500.0],
        "isFraud": [0, 1, 0],
        "isFlaggedFraud": [0, 0, 0],
    })


def test_valid_dataset(valid_transactions):
    report = validate_transactions(valid_transactions)

    assert report.is_valid is True
    assert report.total_rows == 3
    assert report.total_columns == 11
    assert report.errors == []
    assert report.warnings == []


def test_missing_required_column(valid_transactions):
    invalid_data = valid_transactions.drop(columns=["amount"])

    report = validate_transactions(invalid_data)

    assert report.is_valid is False
    assert any("Missing required columns" in error for error in report.errors)


def test_missing_values(valid_transactions):
    invalid_data = valid_transactions.copy()
    invalid_data.loc[0, "amount"] = None

    report = validate_transactions(invalid_data)

    assert report.is_valid is False
    assert "Dataset contains missing values." in report.errors
    assert report.missing_values["amount"] == 1


def test_negative_transaction_amount(valid_transactions):
    invalid_data = valid_transactions.copy()
    invalid_data.loc[0, "amount"] = -100.0

    report = validate_transactions(invalid_data)

    assert report.is_valid is False
    assert "Transaction amounts cannot be negative." in report.errors


def test_invalid_fraud_label(valid_transactions):
    invalid_data = valid_transactions.copy()
    invalid_data.loc[0, "isFraud"] = 2

    report = validate_transactions(invalid_data)

    assert report.is_valid is False
    assert "Column 'isFraud' must contain only 0 or 1." in report.errors


def test_duplicate_rows_generate_warning(valid_transactions):
    invalid_data = pd.concat(
        [valid_transactions, valid_transactions.iloc[[0]]],
        ignore_index=True,
    )

    report = validate_transactions(invalid_data)

    assert report.is_valid is True
    assert report.duplicate_rows == 1
    assert len(report.warnings) == 1
    assert "1 duplicate rows" in report.warnings[0]


def test_empty_dataset():
    empty_data = pd.DataFrame()

    report = validate_transactions(empty_data)

    assert report.is_valid is False
    assert "Dataset is empty." in report.errors
