# System Architecture

## Overview
The project uses a 10-phase architecture for financial fraud detection.

## Architecture Flow
1. Data Ingestion - Load CSV, JSON, JSONL, Parquet and Excel files.
2. Data Validation - Check data quality and required columns.
3. Data Preprocessing - Clean and transform transaction data.
4. Feature Engineering - Generate transaction-to-balance ratios.
5. Feature Store - Use Feast for historical and online features.
6. Model Training - Train Logistic Regression, Random Forest and XGBoost.
7. Model Selection - Compare models using fraud detection metrics.
8. Model Registry - Track and register models using MLflow.
9. Deployment - Serve predictions using FastAPI and Docker.
10. Monitoring - Monitor metrics and data drift.

## Supporting Technologies
- AWS S3: Cloud storage.
- Kafka: Transaction streaming.
- PostgreSQL: Transaction and monitoring records.
- Prometheus: Application metrics.
- Grafana: Monitoring dashboards.
- Evidently: Data drift evaluation.
- DVC: Dataset versioning.
- GitHub Actions: Automated testing.

## Main API Endpoints
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

## Notes
Pre-transaction features are used for the primary fraud detection model to avoid using information that becomes available only after a transaction.
