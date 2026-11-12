import json
import os

from kafka import KafkaProducer
from kafka.serializer import Serializer


class JSONSerializer(Serializer):
    """Serialize Python dictionaries into JSON bytes."""

    def serialize(self, topic, headers, data):
        return json.dumps(data).encode("utf-8")


class TransactionProducer:
    """Publish financial transactions to a Kafka topic."""

    def __init__(
        self,
        bootstrap_servers=None,
        topic="financial-transactions-test"
    ):
        self.topic = topic

        if bootstrap_servers is None:
            bootstrap_servers = os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS",
                "localhost:9093",
            )

        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=JSONSerializer()
        )

    def send_transaction(self, transaction):
        """Send one transaction to Kafka."""

        future = self.producer.send(self.topic, value=transaction)
        future.get(timeout=10)

    def close(self):
        """Close the Kafka producer safely."""

        self.producer.flush()
        self.producer.close()
