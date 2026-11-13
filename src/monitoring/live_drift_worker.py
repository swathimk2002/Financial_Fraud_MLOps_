"""Persistent batch worker for live feature drift monitoring."""

import json
import logging
import time
from pathlib import Path

import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

from src.database.live_monitoring_operations import (
    claim_feature_batch,
    complete_drift_evaluation,
    get_batch_observations,
    get_pending_batches,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "transaction_features.parquet"
)

FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_orig_balance_ratio",
    "amount_to_dest_balance_ratio",
]

BATCH_SIZE = 500
POLL_INTERVAL_SECONDS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


def load_reference_data():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {REFERENCE_PATH}"
        )

    df = pd.read_parquet(REFERENCE_PATH)

    missing = [
        column for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing reference features: {missing}")

    df = df[FEATURE_COLUMNS].copy()

    if df.empty or df.isnull().any().any():
        raise ValueError("Reference data is empty or contains nulls.")

    return df


def evaluate_batch(batch_id, observations, reference_df):
    if len(observations) != BATCH_SIZE:
        raise ValueError(
            f"Expected {BATCH_SIZE} observations; "
            f"received {len(observations)}."
        )

    current_df = pd.DataFrame(observations)[FEATURE_COLUMNS].copy()

    if current_df.isnull().any().any():
        raise ValueError("Current batch contains missing values.")

    report = Report(
        metrics=[DataDriftPreset(columns=FEATURE_COLUMNS)],
        metadata={
            "project": "Financial Fraud Detection MLOps",
            "monitoring_type": "Live feature drift",
            "batch_id": str(batch_id),
        },
    )

    snapshot = report.run(
        reference_data=Dataset.from_pandas(reference_df),
        current_data=Dataset.from_pandas(current_df),
    )

    report_data = json.loads(snapshot.json())

    feature_results = {}
    drift_detected = False
    drift_share = 0.0

    for metric in report_data["metrics"]:
        name = metric["metric_name"]
        value = metric["value"]

        if name.startswith("DriftedColumnsCount"):
            drift_detected = value["count"] > 0
            drift_share = value["share"]

        elif name.startswith("ValueDrift"):
            column = metric["config"]["column"]

            feature_results[column] = {
                "p_value": value,
                "method": metric["config"].get("method"),
                "threshold": metric["config"].get("threshold"),
            }

    if not feature_results:
        raise ValueError("No feature drift results returned.")

    processed = complete_drift_evaluation(
        batch_id=batch_id,
        reference_rows=len(reference_df),
        current_rows=len(current_df),
        drift_detected=drift_detected,
        drift_share=drift_share,
        feature_results=feature_results,
    )

    logger.info(
        "Batch %s completed: %s observations, drift=%s, share=%.4f",
        batch_id,
        processed,
        drift_detected,
        drift_share,
    )

    return processed


def process_pending_batches(reference_df):
    count = 0

    for batch_id in get_pending_batches():
        observations = get_batch_observations(batch_id)

        if observations:
            evaluate_batch(batch_id, observations, reference_df)
            count += 1

    return count


def run_worker():
    reference_df = load_reference_data()

    logger.info(
        "Live drift worker started. Reference rows: %s",
        len(reference_df),
    )

    while True:
        try:
            process_pending_batches(reference_df)

            batch = claim_feature_batch(BATCH_SIZE)

            if batch is not None:
                evaluate_batch(
                    batch["batch_id"],
                    batch["observations"],
                    reference_df,
                )
            else:
                time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Worker stopped.")
            break

        except Exception:
            logger.exception("Worker error.")
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_worker()

