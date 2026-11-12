import json
import os

from kafka import KafkaConsumer, KafkaProducer
from kafka.serializer import Deserializer, Serializer

from src.database.operations import (
    insert_transaction,
    insert_prediction_event,
)


class JSONDeserializer(Deserializer):
    """Convert JSON bytes into Python dictionaries."""

    def deserialize(self, topic, headers, data):
        return json.loads(data.decode("utf-8"))


class JSONSerializer(Serializer):
    """Convert Python dictionaries into JSON bytes."""

    def serialize(self, topic, headers, data):
        return json.dumps(data).encode("utf-8")


KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9093",
)


REQUIRED_FIELDS = {
    "transaction_id",
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}


def process_transaction(transaction):
    """Validate and store one transaction in PostgreSQL."""

    if not isinstance(transaction, dict):
        raise ValueError("Transaction must be a dictionary.")

    missing_fields = REQUIRED_FIELDS - transaction.keys()

    if missing_fields:
        raise ValueError(f"Missing transaction fields: {missing_fields}")

    if not isinstance(transaction["transaction_id"], str) or not transaction["transaction_id"].strip():
        raise ValueError("Transaction ID must be a non-empty string.")

    if transaction["isFraud"] not in (0, 1):
        raise ValueError("isFraud must be 0 or 1.")

    if transaction["isFlaggedFraud"] not in (0, 1):
        raise ValueError("isFlaggedFraud must be 0 or 1.")

    database_transaction = {
        "transaction_id": transaction["transaction_id"],
        "step": transaction["step"],
        "transaction_type": transaction["type"],
        "amount": transaction["amount"],
        "name_orig": transaction["nameOrig"],
        "old_balance_orig": transaction["oldbalanceOrg"],
        "new_balance_orig": transaction["newbalanceOrig"],
        "name_dest": transaction["nameDest"],
        "old_balance_dest": transaction["oldbalanceDest"],
        "new_balance_dest": transaction["newbalanceDest"],
        "is_fraud": transaction["isFraud"],
        "is_flagged_fraud": transaction["isFlaggedFraud"],
    }

    insert_transaction(database_transaction)

    print(f"Transaction stored: {transaction['transaction_id']}")


def send_to_dead_letter(producer, transaction, error):
    """Send an invalid transaction to the dead-letter topic."""

    dead_letter_message = {
        "original_transaction": transaction,
        "error": str(error),
    }

    future = producer.send(
        "financial-transactions-dead-letter",
        value=dead_letter_message,
    )

    future.get(timeout=10)

    print("Invalid transaction sent to the dead-letter topic.")


def start_consumer():
    """Consume and process transactions from Kafka."""

    consumer = KafkaConsumer(
        "financial-transactions-test",
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="financial-fraud-consumer-test",
        value_deserializer=JSONDeserializer(),
    )

    dead_letter_producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=JSONSerializer(),
    )

    print("Kafka consumer started. Waiting for transactions...")

    try:
        for message in consumer:
            try:
                process_transaction(message.value)

            except ValueError as error:
                send_to_dead_letter(
                    dead_letter_producer,
                    message.value,
                    error,
                )

            except Exception as error:
                print(f"Processing failed. Message not committed: {error}")
                raise

            consumer.commit()

    finally:
        consumer.close()
        dead_letter_producer.flush()
        dead_letter_producer.close()


def process_prediction_event(event):
    """Validate and store an API prediction event in PostgreSQL."""

    if not isinstance(event, dict):
        raise ValueError("Prediction event must be a dictionary.")

    if "transaction" not in event or "prediction" not in event:
        raise ValueError("Event must contain transaction and prediction.")

    transaction = event["transaction"]
    prediction = event["prediction"]

    if not isinstance(transaction, dict) or not isinstance(prediction, dict):
        raise ValueError("Transaction and prediction must be dictionaries.")

    if transaction.get("transaction_id") != prediction.get("transaction_id"):
        raise ValueError("Transaction IDs do not match.")

    required_transaction_fields = {
        "transaction_id",
        "step",
        "transaction_type",
        "amount",
        "old_balance_orig",
        "old_balance_dest",
    }

    required_prediction_fields = {
        "transaction_id",
        "predicted_fraud",
        "fraud_probability",
        "model_name",
        "model_version",
        "threshold",
    }

    if not required_transaction_fields.issubset(transaction):
        raise ValueError("Missing required transaction fields.")

    if not required_prediction_fields.issubset(prediction):
        raise ValueError("Missing required prediction fields.")

    if prediction["predicted_fraud"] not in (0, 1):
        raise ValueError("predicted_fraud must be 0 or 1.")

    insert_prediction_event(event)

    print(
        f"Prediction stored: {transaction['transaction_id']} "
        f"| Fraud: {prediction['predicted_fraud']}"
    )


def start_prediction_consumer():
    """Consume API prediction events and store them in PostgreSQL."""

    consumer = KafkaConsumer(
        "financial-predictions-test",
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="financial-fraud-prediction-consumer",
        value_deserializer=JSONDeserializer(),
    )

    print("Kafka consumer started. Waiting for transactions...")

    try:
        while True:
            messages = consumer.poll(timeout_ms=1000)

            if not messages:
                continue

            print("Assigned partitions:", consumer.assignment())

            for partition_messages in messages.values():
                for message in partition_messages:
                    print("Message received:", message.value)

                    process_prediction_event(message.value)

                    consumer.commit()

                    print("Kafka offset committed.")

    finally:
        consumer.close()


if __name__ == "__main__":
    start_prediction_consumer()