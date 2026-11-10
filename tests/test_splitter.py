
"""Tests for dataset splitting."""

import pandas as pd

from src.preprocessing.splitter import split_dataset


def test_split_dataset_creates_expected_sizes():
    df = pd.DataFrame({
        "step": range(200),
        "type": ["TRANSFER", "CASH_OUT"] * 100,
        "amount": [100.0] * 200,
        "oldbalanceOrg": [500.0] * 200,
        "oldbalanceDest": [1000.0] * 200,
        "newbalanceOrig": [400.0] * 200,
        "newbalanceDest": [1100.0] * 200,
        "nameOrig": [f"C{i}" for i in range(200)],
        "nameDest": [f"M{i}" for i in range(200)],
        "isFlaggedFraud": [0] * 200,
        "isFraud": [0, 1] * 100,
        "amount_to_orig_balance_ratio": [0.2] * 200,
        "amount_to_dest_balance_ratio": [0.1] * 200,
    })

    split = split_dataset(df)

    assert len(split.X_train) == 140
    assert len(split.X_validation) == 30
    assert len(split.X_test) == 30

    assert len(split.y_train) == 140
    assert len(split.y_validation) == 30
    assert len(split.y_test) == 30


def test_split_dataset_excludes_target_and_leakage_columns():
    df = pd.DataFrame({
        "step": range(200),
        "type": ["TRANSFER", "CASH_OUT"] * 100,
        "amount": [100.0] * 200,
        "oldbalanceOrg": [500.0] * 200,
        "oldbalanceDest": [1000.0] * 200,
        "newbalanceOrig": [400.0] * 200,
        "newbalanceDest": [1100.0] * 200,
        "nameOrig": [f"C{i}" for i in range(200)],
        "nameDest": [f"M{i}" for i in range(200)],
        "isFlaggedFraud": [0] * 200,
        "isFraud": [0, 1] * 100,
        "amount_to_orig_balance_ratio": [0.2] * 200,
        "amount_to_dest_balance_ratio": [0.1] * 200,
    })

    split = split_dataset(df)

    excluded_columns = {
        "isFraud",
        "isFlaggedFraud",
        "nameOrig",
        "nameDest",
        "newbalanceOrig",
        "newbalanceDest",
    }

    assert excluded_columns.isdisjoint(split.X_train.columns)
    assert excluded_columns.isdisjoint(split.X_validation.columns)
    assert excluded_columns.isdisjoint(split.X_test.columns)
