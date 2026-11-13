# API and Docker Deployment

## Purpose
FastAPI provides an interface for financial fraud predictions.

## Implementation
- API source: src/api/main.py
- Dockerfile: Dockerfile
- Compose configuration: docker-compose.yml
- Container: financial-fraud-api

## API Endpoints
- Health check: http://localhost:8000/health
- Prometheus metrics: http://localhost:8000/metrics

## Configuration
The API connects to MLflow, Feast, Kafka and PostgreSQL using environment variables and Docker networking.

## Storage
MLflow database, MLflow artifacts, model artifacts and Feast data are mounted from the project directory.

## Verification
The API health endpoint returned healthy. The champion model, Kafka and Feast connections were verified.

## Containerization
Docker is used to package and run the API service.
