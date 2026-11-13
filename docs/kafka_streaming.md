# Kafka Streaming

## Purpose
Apache Kafka streams financial transaction messages.

## Implementation
- Producer: src/streaming/producer.py
- Consumer: src/streaming/consumer.py
- Docker container: financial-fraud-kafka
- Kafka version: 4.1.2

## Topics
- financial-transactions
- financial-transactions-test
- financial-transactions-dead-letter
- financial-predictions-test

## Connection
- Internal Docker address: financial-fraud-kafka:19092
- Host port: 9093

## Message Format
Transaction messages use JSON format.

## Verification
Broker connectivity, topic availability and message consumption were tested successfully.
