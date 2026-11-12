
"""Tests for the FastAPI fraud prediction endpoints."""

from contextlib import asynccontextmanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main
from unittest.mock import MagicMock

class MockModel:
    """Provide a fake model for API testing."""

    def predict_proba(self, features):
        """Return a fixed fraud probability as a NumPy array."""
        return np.array([[0.2, 0.8]])


@asynccontextmanager
async def skip_startup(app):
    """Skip real MLflow model loading during API tests."""
    yield


@pytest.fixture(autouse=True)
def setup_test_model(monkeypatch):
    """Use mock model and Kafka producer during API tests."""

    monkeypatch.setattr(
        main.app.router,
        "lifespan_context",
        skip_startup,
    )

    main.model = MockModel()
    main.producer = MagicMock()

    yield

    main.model = None
    main.producer = None

def test_health_check():
    """Check that the health endpoint works."""

    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == main.MODEL_NAME


def test_predict_fraud():
    """Check that the prediction endpoint returns a fraud prediction."""

    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            json={
                "step": 100,
                "type": "CASH_OUT",
                "amount": 150000,
                "oldbalanceOrg": 150000,
                "oldbalanceDest": 0,
            },
        )

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["classification"] == "Fraudulent"
    assert response.json()["fraud_probability"] == 0.8


def test_invalid_transaction():
    """Check that invalid transaction data is rejected."""

    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            json={
                "step": 100,
                "type": "INVALID_TYPE",
                "amount": -150000,
                "oldbalanceOrg": 150000,
                "oldbalanceDest": 0,
            },
        )

    assert response.status_code == 422


def test_model_unavailable():
    """Check that prediction is rejected when the model is unavailable."""

    with TestClient(main.app) as client:
        main.model = None

        response = client.post(
            "/predict",
            json={
                "step": 100,
                "type": "CASH_OUT",
                "amount": 150000,
                "oldbalanceOrg": 150000,
                "oldbalanceDest": 0,
            },
        )

    assert response.status_code == 503
