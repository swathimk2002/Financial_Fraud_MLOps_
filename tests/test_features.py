
"""Tests for pre-transaction feature engineering."""

import pandas as pd
import pytest

from src.features.engineer import engineer_features


def test_engineer_features_creates_ratios():
    df = pd.DataFrame({
        "amount": [100.0, 200.0],
        "oldbalanceOrg": [500.0, 0.0],
        "oldbalanceDest": [1000.0, 400.0],
    })

    result = engineer_features(df)

    assert "amount_to_orig_balance_ratio" in result.columns
    assert "amount_to_dest_balance_ratio" in result.columns

    assert result.loc[0, "amount_to_orig_balance_ratio"] == pytest.approx(
        100 / 501
    )
    assert result.loc[1, "amount_to_orig_balance_ratio"] == pytest.approx(
        200
    )


def test_engineer_features_does_not_modify_original():
    df = pd.DataFrame({
        "amount": [100.0],
        "oldbalanceOrg": [500.0],
        "oldbalanceDest": [1000.0],
    })

    original = df.copy()
    engineer_features(df)

    pd.testing.assert_frame_equal(df, original)


def test_engineer_features_missing_column():
    df = pd.DataFrame({
        "amount": [100.0],
        "oldbalanceOrg": [500.0],
    })

    with pytest.raises(ValueError, match="Missing columns"):
        engineer_features(df)
