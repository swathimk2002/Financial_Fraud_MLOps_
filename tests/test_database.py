
"""Tests for PostgreSQL database operations."""

from unittest.mock import MagicMock, patch

from src.database.operations import insert_prediction_event


def create_prediction_event(predicted_fraud):
    """Create a sample API prediction event."""

    transaction_id = "API_TEST_001"

    return {
        "transaction": {
            "transaction_id": transaction_id,
            "step": 100,
            "transaction_type": "CASH_OUT",
            "amount": 150000.0,
            "old_balance_orig": 150000.0,
            "old_balance_dest": 0.0,
        },
        "prediction": {
            "transaction_id": transaction_id,
            "predicted_fraud": predicted_fraud,
            "fraud_probability": 0.85,
            "model_name": "Financial_Fraud_Detection_Model",
            "model_version": 1,
            "threshold": 0.6,
        },
    }


@patch("src.database.operations.get_engine")
def test_legitimate_prediction_does_not_create_alert(mock_get_engine):
    """A legitimate prediction should not create a fraud alert."""

    connection = MagicMock()
    engine = MagicMock()

    engine.begin.return_value.__enter__.return_value = connection
    mock_get_engine.return_value = engine

    result = MagicMock()
    result.scalar_one_or_none.return_value = 101

    connection.execute.side_effect = [MagicMock(), result]

    event = create_prediction_event(0)

    prediction_id = insert_prediction_event(event)

    assert prediction_id == 101
    assert connection.execute.call_count == 2
    engine.dispose.assert_called_once()


@patch("src.database.operations.get_engine")
def test_fraudulent_prediction_creates_alert(mock_get_engine):
    """A fraudulent prediction should create a fraud alert."""

    connection = MagicMock()
    engine = MagicMock()

    engine.begin.return_value.__enter__.return_value = connection
    mock_get_engine.return_value = engine

    result = MagicMock()
    result.scalar_one_or_none.return_value = 102

    connection.execute.side_effect = [
        MagicMock(),
        result,
        MagicMock(),
    ]

    event = create_prediction_event(1)

    prediction_id = insert_prediction_event(event)

    assert prediction_id == 102
    assert connection.execute.call_count == 3

    alert_parameters = connection.execute.call_args.args[1]

    assert alert_parameters["transaction_id"] == "API_TEST_001"
    assert alert_parameters["prediction_id"] == 102


@patch("src.database.operations.get_engine")
def test_duplicate_prediction_does_not_create_alert(mock_get_engine):
    """A duplicate prediction should not create another alert."""

    connection = MagicMock()
    engine = MagicMock()

    engine.begin.return_value.__enter__.return_value = connection
    mock_get_engine.return_value = engine

    result = MagicMock()
    result.scalar_one_or_none.return_value = None

    connection.execute.side_effect = [MagicMock(), result]

    event = create_prediction_event(1)

    prediction_id = insert_prediction_event(event)

    assert prediction_id is None
    assert connection.execute.call_count == 2


@patch("src.database.operations.get_engine")
def test_database_error_is_propagated(mock_get_engine):
    """A database failure should propagate to the caller."""

    connection = MagicMock()
    engine = MagicMock()

    engine.begin.return_value.__enter__.return_value = connection
    mock_get_engine.return_value = engine

    connection.execute.side_effect = RuntimeError("Database failure")

    event = create_prediction_event(1)

    import pytest

    with pytest.raises(RuntimeError, match="Database failure"):
        insert_prediction_event(event)

    engine.dispose.assert_called_once()
