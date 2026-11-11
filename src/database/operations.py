
"""Database operations for transactions, predictions, and fraud alerts."""

from sqlalchemy import text

from src.database.connection import get_engine


def insert_transaction(transaction):
    """Insert a transaction into the transactions table."""

    query = text("""
        INSERT INTO transactions (
            transaction_id,
            step,
            transaction_type,
            amount,
            name_orig,
            old_balance_orig,
            new_balance_orig,
            name_dest,
            old_balance_dest,
            new_balance_dest,
            is_fraud,
            is_flagged_fraud
        )
        VALUES (
            :transaction_id,
            :step,
            :transaction_type,
            :amount,
            :name_orig,
            :old_balance_orig,
            :new_balance_orig,
            :name_dest,
            :old_balance_dest,
            :new_balance_dest,
            :is_fraud,
            :is_flagged_fraud
        )
        ON CONFLICT (transaction_id) DO NOTHING
    """)

    engine = get_engine()

    try:
        with engine.begin() as connection:
            connection.execute(query, transaction)
    finally:
        engine.dispose()


def insert_prediction(prediction):
    """Save a model prediction and return its prediction ID."""

    query = text("""
        INSERT INTO predictions (
            transaction_id,
            predicted_fraud,
            fraud_probability,
            model_name,
            model_version,
            threshold
        )
        VALUES (
            :transaction_id,
            :predicted_fraud,
            :fraud_probability,
            :model_name,
            :model_version,
            :threshold
        )
        RETURNING prediction_id
    """)

    engine = get_engine()

    try:
        with engine.begin() as connection:
            result = connection.execute(query, prediction)
            return result.scalar_one()
    finally:
        engine.dispose()


def insert_fraud_alert(transaction_id, prediction_id):
    """Create a fraud alert for a predicted fraudulent transaction."""

    query = text("""
        INSERT INTO fraud_alerts (
            transaction_id,
            prediction_id
        )
        VALUES (
            :transaction_id,
            :prediction_id
        )
    """)

    engine = get_engine()

    try:
        with engine.begin() as connection:
            connection.execute(
                query,
                {
                    "transaction_id": transaction_id,
                    "prediction_id": prediction_id,
                },
            )
    finally:
        engine.dispose()
