
"""Tests for Kafka transaction streaming."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.streaming.producer import JSONSerializer
from src.streaming.consumer import (
    JSONDeserializer,
    REQUIRED_FIELDS,
    process_transaction,
    send_to_dead_letter,
)


@pytest.fixture
def valid_transaction():
    """Create a valid sample transaction."""
    return {
        "transaction_id": "TEST_STREAM_001",
        "step": 100,
        "type": "TRANSFER",
        "amount": 5000,
        "nameOrig": "C12345",
        "oldbalanceOrg": 10000,
        "newbalanceOrig": 5000,
        "nameDest": "C67890",
        "oldbalanceDest": 2000,
        "newbalanceDest": 7000,
        "isFraud": 0,
        "isFlaggedFraud": 0,
    }


def test_json_serialization(valid_transaction):
    """Convert a transaction dictionary into JSON bytes."""
    serializer = JSONSerializer()
    result = serializer.serialize("test-topic", None, valid_transaction)

    assert json.loads(result.decode("utf-8")) == valid_transaction


def test_json_deserialization(valid_transaction):
    """Convert JSON bytes into a transaction dictionary."""
    deserializer = JSONDeserializer()
    data = json.dumps(valid_transaction).encode("utf-8")

    result = deserializer.deserialize("test-topic", None, data)

    assert result == valid_transaction


def test_valid_transaction_is_stored(valid_transaction):
    """Store a valid transaction in PostgreSQL."""
    with patch("src.streaming.consumer.insert_transaction") as mock_insert:
        process_transaction(valid_transaction)

    mock_insert.assert_called_once()

    stored_data = mock_insert.call_args.args[0]
    assert stored_data["transaction_id"] == "TEST_STREAM_001"
    assert stored_data["transaction_type"] == "TRANSFER"
    assert stored_data["is_fraud"] == 0


def test_missing_required_fields(valid_transaction):
    """Reject transactions with missing fields."""
    transaction = valid_transaction.copy()
    del transaction["amount"]

    with pytest.raises(ValueError, match="Missing transaction fields"):
        process_transaction(transaction)


def test_non_dictionary_transaction():
    """Reject transactions that are not dictionaries."""
    with pytest.raises(ValueError, match="must be a dictionary"):
        process_transaction(["invalid"])


def test_empty_transaction_id(valid_transaction):
    """Reject an empty transaction ID."""
    valid_transaction["transaction_id"] = " "

    with pytest.raises(ValueError, match="Transaction ID"):
        process_transaction(valid_transaction)


@pytest.mark.parametrize("fraud_label", [2, -1, "1"])
def test_invalid_fraud_label(valid_transaction, fraud_label):
    """Reject invalid fraud labels."""
    valid_transaction["isFraud"] = fraud_label

    with pytest.raises(ValueError, match="isFraud"):
        process_transaction(valid_transaction)


@pytest.mark.parametrize("flag", [2, -1, "1"])
def test_invalid_flagged_fraud_label(valid_transaction, flag):
    """Reject invalid flagged-fraud labels."""
    valid_transaction["isFlaggedFraud"] = flag

    with pytest.raises(ValueError, match="isFlaggedFraud"):
        process_transaction(valid_transaction)


def test_dead_letter_message():
    """Publish an invalid transaction with its error details."""
    producer = MagicMock()
    producer.send.return_value.get.return_value = None

    transaction = {"transaction_id": "INVALID_TEST_001"}
    error = ValueError("Missing required fields")

    send_to_dead_letter(producer, transaction, error)

    producer.send.assert_called_once_with(
        "financial-transactions-dead-letter",
        value={
            "original_transaction": transaction,
            "error": "Missing required fields",
        },
    )


def test_required_fields_count():
    """Ensure all expected transaction fields are required."""
    assert len(REQUIRED_FIELDS) == 12
