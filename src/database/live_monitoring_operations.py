"""Database operations for persistent live drift monitoring."""

import json
from uuid import uuid4

from sqlalchemy import text

from src.database.connection import get_engine


def save_feature_observations(features):
    """Persist individual live feature observations awaiting batching."""

    if features.empty:
        raise ValueError("Cannot save an empty feature DataFrame.")

    required_columns = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "amount_to_orig_balance_ratio",
        "amount_to_dest_balance_ratio",
    ]

    missing = [
        column for column in required_columns
        if column not in features.columns
    ]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    feature_batch = features[required_columns].copy()

    if feature_batch.isnull().any().any():
        raise ValueError("Feature DataFrame contains missing values.")

    query = text("""
        INSERT INTO live_feature_observations (
            step,
            transaction_type,
            amount,
            old_balance_orig,
            old_balance_dest,
            amount_to_orig_balance_ratio,
            amount_to_dest_balance_ratio
        )
        VALUES (
            :step,
            :transaction_type,
            :amount,
            :old_balance_orig,
            :old_balance_dest,
            :amount_to_orig_balance_ratio,
            :amount_to_dest_balance_ratio
        )
    """)

    records = feature_batch.to_dict(orient="records")

    parameters = [
        {
            "step": record["step"],
            "transaction_type": record["type"],
            "amount": record["amount"],
            "old_balance_orig": record["oldbalanceOrg"],
            "old_balance_dest": record["oldbalanceDest"],
            "amount_to_orig_balance_ratio": record[
                "amount_to_orig_balance_ratio"
            ],
            "amount_to_dest_balance_ratio": record[
                "amount_to_dest_balance_ratio"
            ],
        }
        for record in records
    ]

    engine = get_engine()

    try:
        with engine.begin() as connection:
            connection.execute(query, parameters)

        return len(parameters)

    finally:
        engine.dispose()


def claim_feature_batch(batch_size=500):
    """Atomically claim a complete batch of unassigned observations."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    batch_id = uuid4()

    engine = get_engine()

    try:
        with engine.begin() as connection:
            selected = connection.execute(
                text("""
                    SELECT observation_id
                    FROM live_feature_observations
                    WHERE processed = FALSE
                      AND batch_id IS NULL
                    ORDER BY observation_id
                    FOR UPDATE SKIP LOCKED
                    LIMIT :batch_size
                """),
                {"batch_size": batch_size},
            )

            observation_ids = [
                row.observation_id for row in selected
            ]

            if len(observation_ids) < batch_size:
                return None

            connection.execute(
                text("""
                    UPDATE live_feature_observations
                    SET batch_id = :batch_id,
                        claimed_at = CURRENT_TIMESTAMP
                    WHERE observation_id = ANY(
                        CAST(:observation_ids AS BIGINT[])
                    )
                """),
                {
                    "batch_id": str(batch_id),
                    "observation_ids": observation_ids,
                },
            )

            result = connection.execute(
                text("""
                    SELECT
                        observation_id,
                        batch_id,
                        step,
                        transaction_type AS type,
                        amount,
                        old_balance_orig AS "oldbalanceOrg",
                        old_balance_dest AS "oldbalanceDest",
                        amount_to_orig_balance_ratio,
                        amount_to_dest_balance_ratio
                    FROM live_feature_observations
                    WHERE batch_id = :batch_id
                      AND processed = FALSE
                    ORDER BY observation_id
                """),
                {"batch_id": str(batch_id)},
            )

            rows = [dict(row._mapping) for row in result]

            return {
                "batch_id": batch_id,
                "observations": rows,
            }

    finally:
        engine.dispose()


def get_pending_batches():
    """Retrieve previously claimed batches for recovery after restart."""

    query = text("""
        SELECT DISTINCT batch_id
        FROM live_feature_observations
        WHERE batch_id IS NOT NULL
          AND processed = FALSE
        ORDER BY batch_id
    """)

    engine = get_engine()

    try:
        with engine.connect() as connection:
            result = connection.execute(query)
            return [row.batch_id for row in result]

    finally:
        engine.dispose()


def get_batch_observations(batch_id):
    """Retrieve observations belonging to a pending batch."""

    query = text("""
        SELECT
            observation_id,
            batch_id,
            step,
            transaction_type AS type,
            amount,
            old_balance_orig AS "oldbalanceOrg",
            old_balance_dest AS "oldbalanceDest",
            amount_to_orig_balance_ratio,
            amount_to_dest_balance_ratio
        FROM live_feature_observations
        WHERE batch_id = :batch_id
          AND processed = FALSE
        ORDER BY observation_id
    """)

    engine = get_engine()

    try:
        with engine.connect() as connection:
            result = connection.execute(
                query,
                {"batch_id": str(batch_id)},
            )

            return [dict(row._mapping) for row in result]

    finally:
        engine.dispose()


def complete_drift_evaluation(
    batch_id,
    reference_rows,
    current_rows,
    drift_detected,
    drift_share,
    feature_results,
):
    """Save an evaluation and complete its observations atomically."""

    if current_rows <= 0:
        raise ValueError("current_rows must be greater than zero.")

    evaluation_query = text("""
        INSERT INTO drift_evaluations (
            batch_id,
            reference_rows,
            current_rows,
            drift_detected,
            drift_share,
            feature_results
        )
        VALUES (
            :batch_id,
            :reference_rows,
            :current_rows,
            :drift_detected,
            :drift_share,
            CAST(:feature_results AS JSONB)
        )
        ON CONFLICT (batch_id)
        DO UPDATE SET
            reference_rows = EXCLUDED.reference_rows,
            current_rows = EXCLUDED.current_rows,
            drift_detected = EXCLUDED.drift_detected,
            drift_share = EXCLUDED.drift_share,
            feature_results = EXCLUDED.feature_results,
            evaluated_at = CURRENT_TIMESTAMP
    """)

    processed_query = text("""
        UPDATE live_feature_observations
        SET processed = TRUE
        WHERE batch_id = :batch_id
          AND processed = FALSE
    """)

    parameters = {
        "batch_id": str(batch_id),
        "reference_rows": reference_rows,
        "current_rows": current_rows,
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "feature_results": json.dumps(feature_results),
    }

    engine = get_engine()

    try:
        with engine.begin() as connection:
            connection.execute(evaluation_query, parameters)

            result = connection.execute(
                processed_query,
                {"batch_id": str(batch_id)},
            )

            return result.rowcount

    finally:
        engine.dispose()
