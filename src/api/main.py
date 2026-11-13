
"""FastAPI application for financial fraud detection."""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
import os

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from pydantic import BaseModel, Field

from src.features.engineer import engineer_features
from src.mlops.mlflow_manager import initialize_mlflow
from src.streaming.producer import TransactionProducer
from src.database.live_monitoring_operations import save_feature_observations
import time

from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"

MODEL_NAME = "Financial_Fraud_Detection_Model"
MODEL_ALIAS = "champion"

FEAST_REPO_PATH = PROJECT_ROOT / "feature_store"

FEAST_FEATURES = [
    "transaction_features:step",
    "transaction_features:type",
    "transaction_features:amount",
    "transaction_features:oldbalanceOrg",
    "transaction_features:oldbalanceDest",
    "transaction_features:amount_to_orig_balance_ratio",
    "transaction_features:amount_to_dest_balance_ratio",
]


# --------------------------------------------------
# 2. LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# PROMETHEUS METRICS
# --------------------------------------------------

PREDICTION_REQUESTS = Counter(
    "fraud_prediction_requests_total",
    "Total number of prediction requests.",
)

PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total number of successful fraud predictions.",
    ["prediction"],
)

PREDICTION_ERRORS = Counter(
    "fraud_prediction_errors_total",
    "Total number of prediction processing errors.",
)

PREDICTION_DURATION = Histogram(
    "fraud_prediction_duration_seconds",
    "Time taken to process prediction requests.",
)

# --------------------------------------------------
# 3. LOAD MODEL METADATA
# --------------------------------------------------

if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    THRESHOLD = metadata["classification_threshold"]
    FEATURE_COLUMNS = metadata["feature_columns"]

elif __import__("os").getenv("CI") == "true":
    logger.warning("Using test-only metadata defaults in CI.")

    THRESHOLD = 0.6
    FEATURE_COLUMNS = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "amount_to_orig_balance_ratio",
        "amount_to_dest_balance_ratio",
    ]

else:
    raise FileNotFoundError(
        f"Required model metadata not found: {METADATA_PATH}"
    )

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


# --------------------------------------------------
# 4. MODEL, KAFKA AND FEAST OBJECTS
# --------------------------------------------------

model = None
producer = None
feature_store = None


# --------------------------------------------------
# 5. LOAD MODEL AND CONNECT SERVICES AT STARTUP
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize MLflow, Kafka and Feast at API startup."""

    global model, producer, feature_store

    try:
        initialize_mlflow()

        logger.info("Loading champion model from MLflow.")

        model = mlflow.sklearn.load_model(MODEL_URI)

        logger.info("Champion model loaded successfully.")

        logger.info("Connecting to Kafka.")

        producer = TransactionProducer(
            topic="financial-predictions-test"
        )

        logger.info("Kafka producer initialized successfully.")

        # Initialize Feast independently so that a Feast issue
        # does not prevent the existing prediction API from starting.
        try:
            logger.info("Initializing Feast feature store.")

            feature_store = FeatureStore(
                repo_path=str(FEAST_REPO_PATH)
            )

            logger.info("Feast feature store initialized successfully.")

        except Exception:
            feature_store = None
            logger.exception("Feast initialization failed.")

        yield

    except Exception:
        logger.exception("API startup failed.")
        raise

    finally:
        if producer is not None:
            producer.close()
            producer = None

        model = None
        feature_store = None


# --------------------------------------------------
# 6. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Financial Fraud Detection API",
    description=(
        "Predict fraudulent financial transactions using "
        "MLflow, Kafka and Feast."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# 7. TRANSACTION INPUT VALIDATION
# --------------------------------------------------

class TransactionInput(BaseModel):
    """Validate incoming pre-transaction details."""

    step: int = Field(ge=1)

    type: str = Field(
        pattern="^(CASH_IN|CASH_OUT|DEBIT|PAYMENT|TRANSFER)$"
    )

    amount: float = Field(ge=0, allow_inf_nan=False)

    oldbalanceOrg: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

    oldbalanceDest: float = Field(
        ge=0,
        allow_inf_nan=False,
    )

# --------------------------------------------------
# PROMETHEUS METRICS ENDPOINT
# --------------------------------------------------

@app.get("/metrics")
def prometheus_metrics():
    """Expose application metrics for Prometheus."""

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

# --------------------------------------------------
# 8. HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():
    """Check whether the API, model and Kafka are available."""

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    if producer is None:
        raise HTTPException(
            status_code=503,
            detail="Kafka producer is not available.",
        )

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "kafka": "connected",
        "feast": (
            "connected" if feature_store is not None else "unavailable"
        ),
    }


# --------------------------------------------------
# 9. FRAUD PREDICTION ENDPOINT
# --------------------------------------------------


@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    """Predict fraud, publish to Kafka and record Prometheus metrics."""

    # Count every request reaching this endpoint
    PREDICTION_REQUESTS.inc()

    start_time = time.perf_counter()

    try:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not available.",
            )

        if producer is None:
            raise HTTPException(
                status_code=503,
                detail="Kafka producer is not available.",
            )

        transaction_df = pd.DataFrame(
            [transaction.model_dump()]
        )

        features = engineer_features(transaction_df)

        features = features[FEATURE_COLUMNS]

        try:
            saved_count = save_feature_observations(features)
            logger.info(
                "Saved %s live feature observations.",
                saved_count,
            )
        except Exception:
            logger.exception(
                "Live feature monitoring storage failed."
            )

        probability = float(
            model.predict_proba(features)[0, 1]
        )

        prediction = int(probability >= THRESHOLD)

        transaction_id = str(uuid.uuid4())

        event = {
            "transaction": {
                "transaction_id": transaction_id,
                "step": transaction.step,
                "transaction_type": transaction.type,
                "amount": transaction.amount,
                "old_balance_orig": transaction.oldbalanceOrg,
                "old_balance_dest": transaction.oldbalanceDest,
            },
            "prediction": {
                "transaction_id": transaction_id,
                "predicted_fraud": prediction,
                "fraud_probability": probability,
                "model_name": MODEL_NAME,
                "model_version": 1,
                "threshold": THRESHOLD,
            },
        }

        producer.send_transaction(event)

        # Count successful predictions after Kafka publishing
        prediction_label = (
            "fraud" if prediction == 1 else "legitimate"
        )

        PREDICTIONS_TOTAL.labels(
            prediction=prediction_label
        ).inc()

        logger.info(
            "Prediction event published to Kafka: %s",
            transaction_id,
        )

        return {
            "transaction_id": transaction_id,
            "prediction": prediction,
            "classification": (
                "Fraudulent" if prediction == 1 else "Legitimate"
            ),
            "fraud_probability": probability,
            "threshold": THRESHOLD,
            "model_name": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
            "kafka_status": "published",
        }

    except HTTPException:
        PREDICTION_ERRORS.inc()
        raise

    except Exception as error:
        PREDICTION_ERRORS.inc()

        logger.exception(
            "Fraud prediction or Kafka publishing failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process or publish the prediction.",
        ) from error

    finally:
        PREDICTION_DURATION.observe(
            time.perf_counter() - start_time
        )


# --------------------------------------------------
# 10. FEAST HISTORICAL FEATURE RETRIEVAL ENDPOINT
# --------------------------------------------------

@app.get("/features/{transaction_id}")
def get_transaction_features(transaction_id: str):
    """Retrieve historical transaction features from Feast."""

    if feature_store is None:
        raise HTTPException(
            status_code=503,
            detail="Feast feature store is not available.",
        )

    try:
        result = feature_store.get_online_features(
            features=FEAST_FEATURES,
            entity_rows=[
                {"transaction_id": transaction_id}
            ],
        ).to_dict()

        retrieved_features = {
            name: values[0]
            for name, values in result.items()
            if name != "transaction_id"
        }

        if not retrieved_features or all(
            value is None for value in retrieved_features.values()
        ):
            raise HTTPException(
                status_code=404,
                detail="Transaction features were not found in Feast.",
            )

        return {
            "transaction_id": transaction_id,
            "features": retrieved_features,
        }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "Feast feature retrieval failed for transaction: %s",
            transaction_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve transaction features from Feast.",
        ) from error


