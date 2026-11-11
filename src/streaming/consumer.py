
import json

from kafka import KafkaConsumer, KafkaProducer
from kafka.serializer import Deserializer, Serializer

from src.database.operations import insert_transaction


class JSONDeserializer(Deserializer):
    """Convert JSON bytes into Python dictionaries."""

    def deserialize(self, topic, headers, data):
        return json.loads(data.decode("utf-8"))


class JSONSerializer(Serializer):
    """Convert Python dictionaries into JSON bytes."""

    def serialize(self, topic, headers, data):
        return json.dumps(data).encode("utf-8")


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

    print("Invalid transaction sent to dead-letter topic.")


def start_consumer():
    """Consume and process transactions from Kafka."""

    consumer = KafkaConsumer(
        "financial-transactions-test",
        bootstrap_servers="localhost:9093",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id="financial-fraud-consumer-test",
        value_deserializer=JSONDeserializer(),
    )

    dead_letter_producer = KafkaProducer(
        bootstrap_servers="localhost:9093",
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


if __name__ == "__main__":
    start_consumer()
