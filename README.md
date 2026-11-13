# Financial Fraud Detection MLOps

## Overview
An end-to-end financial fraud detection project using Python, machine learning, AWS and MLOps.

## Objectives
- Ingest and validate transaction data.
- Engineer features and train fraud detection models.
- Automatically select the best model.
- Use Feast and MLflow for MLOps.
- Use AWS S3 for cloud storage.
- Use Kafka for transaction streaming.
- Use PostgreSQL for data storage.
- Deploy predictions using FastAPI and Docker.
- Monitor using Evidently, Prometheus and Grafana.
- Automate tests using GitHub Actions.

## Dataset
- Dataset: PaySim synthetic transactions.
- Original records: 6,362,620.
- Working sample: 200,000.
- Fraud records: 268.
- Target: isFraud.
- Random seed: 42.

## Technologies
Python, Pandas, Scikit-learn, XGBoost, Feast, MLflow, AWS S3, Kafka, PostgreSQL, FastAPI, Docker, Evidently, Prometheus, Grafana, DVC and GitHub Actions.

## Main Folders
- data: datasets
- src: source code
- scripts: execution scripts
- configs: configuration
- feature_store: Feast files
- artifacts: model artifacts
- reports: evaluation reports
- docs: documentation
- tests: automated tests

## Setup
Activate the virtual environment:

    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements.txt

Run model comparison:

    python -m scripts.run_model_comparison

Run tests:

    python -m pytest -v

## Model
Logistic Regression, Random Forest and XGBoost are compared. XGBoost was selected as the champion model.

MLflow model: Financial_Fraud_Detection_Model

## Local Services
- FastAPI: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- MLflow: http://127.0.0.1:5000

## Note
PaySim is synthetic data. Performance on real banking transactions has not been established.

## Detailed Documentation

- docs/system_architecture.md
- docs/data_pipeline.md
- docs/dataset_card.md
- docs/aws_s3_storage.md
- docs/feature_store.md
- docs/model_development.md
- docs/kafka_streaming.md
- docs/postgresql_database.md
- docs/mlflow_model_registry.md
- docs/api_deployment.md
- docs/monitoring.md
- docs/ci_cd.md
- docs/testing.md
- docs/troubleshooting.md
- docs/professor_demo.md
