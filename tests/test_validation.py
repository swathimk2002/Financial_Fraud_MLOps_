
"""Tests for transaction data validation."""

import pandas as pd
import pytest

from src.validation.validator import validate_transactions


@pytest.fixture
def valid_transactions():
    """Create a valid sample transaction dataset."""
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
    """Accept a valid dataset."""
    report = validate_transactions(valid_transactions)
    assert report.is_valid
    assert report.total_rows == 3
    assert report.total_columns == 11
    assert report.errors == []
    assert report.warnings == []


def test_missing_required_column(valid_transactions):
    """Reject datasets with missing columns."""
    df = valid_transactions.drop(columns=["amount"])
    report = validate_transactions(df)
    assert not report.is_valid
    assert any("Missing required columns" in e for e in report.errors)


def test_missing_values(valid_transactions):
    """Reject datasets containing missing values."""
    df = valid_transactions.copy()
    df.loc[0, "amount"] = None
    report = validate_transactions(df)
    assert not report.is_valid
    assert report.missing_values["amount"] == 1


def test_negative_transaction_amount(valid_transactions):
    """Reject negative transaction amounts."""
    df = valid_transactions.copy()
    df.loc[0, "amount"] = -100
    report = validate_transactions(df)
    assert not report.is_valid


def test_invalid_fraud_label(valid_transactions):
    """Reject invalid fraud labels."""
    df = valid_transactions.copy()
    df.loc[0, "isFraud"] = 2
    report = validate_transactions(df)
    assert not report.is_valid


def test_duplicate_rows_generate_warning(valid_transactions):
    """Warn about duplicate transactions."""
    df = pd.concat([valid_transactions, valid_transactions.iloc[[0]]])
    report = validate_transactions(df)
    assert report.is_valid
    assert report.duplicate_rows == 1
    assert report.warnings


def test_empty_dataset():
    """Reject an empty dataset."""
    report = validate_transactions(pd.DataFrame())
    assert not report.is_valid
    assert "Dataset is empty." in report.errors


def test_infinite_amount(valid_transactions):
    """Reject infinite transaction amounts."""
    df = valid_transactions.copy()
    df.loc[0, "amount"] = float("inf")
    assert not validate_transactions(df).is_valid


def test_non_numeric_amount(valid_transactions):
    """Reject non-numeric transaction amounts."""
    df = valid_transactions.copy()
    df["amount"] = df["amount"].astype(str)
    assert not validate_transactions(df).is_valid


def test_unexpected_transaction_type(valid_transactions):
    """Warn about unexpected transaction types."""
    df = valid_transactions.copy()
    df.loc[0, "type"] = "UNKNOWN"
    report = validate_transactions(df)
    assert report.is_valid
    assert report.warnings


def test_negative_balance(valid_transactions):
    """Reject negative account balances."""
    df = valid_transactions.copy()
    df.loc[0, "oldbalanceOrg"] = -100
    assert not validate_transactions(df).is_valid


def test_invalid_flagged_fraud_label(valid_transactions):
    """Reject invalid flagged fraud labels."""
    df = valid_transactions.copy()
    df.loc[0, "isFlaggedFraud"] = 2
    assert not validate_transactions(df).is_valid
