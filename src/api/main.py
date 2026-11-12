
"""FastAPI application for financial fraud detection."""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.engineer import engineer_features
from src.mlops.mlflow_manager import initialize_mlflow
from src.streaming.producer import TransactionProducer


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"

MODEL_NAME = "Financial_Fraud_Detection_Model"
MODEL_ALIAS = "champion"


# --------------------------------------------------
# 2. LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. LOAD MODEL METADATA
# --------------------------------------------------

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

THRESHOLD = metadata["classification_threshold"]
FEATURE_COLUMNS = metadata["feature_columns"]

MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


# --------------------------------------------------
# 4. MODEL AND KAFKA PRODUCER
# --------------------------------------------------

model = None
producer = None


# --------------------------------------------------
# 5. LOAD MODEL AND CONNECT TO KAFKA AT STARTUP
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and initialize Kafka at API startup."""

    global model, producer

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

        yield

    except Exception:
        logger.exception("API startup failed.")
        raise

    finally:
        if producer is not None:
            producer.close()
            producer = None

        model = None


# --------------------------------------------------
# 6. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Financial Fraud Detection API",
    description="Predict fraudulent financial transactions using MLflow and Kafka.",
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
# 8. HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():
    """Check whether the API, model, and Kafka producer are available."""

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
    }


# --------------------------------------------------
# 9. FRAUD PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    """Predict fraud and publish the result to Kafka."""

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

    try:
        # Convert input into a DataFrame.
        transaction_df = pd.DataFrame(
            [transaction.model_dump()]
        )

        # Generate pre-transaction features.
        features = engineer_features(transaction_df)

        # Maintain training feature order.
        features = features[FEATURE_COLUMNS]

        # Calculate fraud probability.
        probability = float(
            model.predict_proba(features)[0, 1]
        )

        # Apply the existing classification threshold.
        prediction = int(probability >= THRESHOLD)

        # Generate a unique transaction ID.
        transaction_id = str(uuid.uuid4())

        # Prepare the transaction and prediction event.
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

        # Publish the event to Kafka.
        producer.send_transaction(event)

        logger.info(
            "Prediction event published to Kafka: %s",
            transaction_id,
        )

        # Return the prediction response.
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

    except Exception as error:
        logger.exception("Fraud prediction or Kafka publishing failed.")

        raise HTTPException(
            status_code=500,
            detail="Unable to process or publish the prediction.",
        ) from error
