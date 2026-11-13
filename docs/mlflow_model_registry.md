# MLflow Model Registry

## Purpose
MLflow tracks machine learning experiments and manages registered models.

## Configuration
- Tracking database: mlflow.db
- Artifact directory: mlartifacts/
- Tracking URL: http://127.0.0.1:5000
- Experiment: Financial_Fraud_Detection

## Registered Model
- Model name: Financial_Fraud_Detection_Model
- Champion alias: champion
- Verified version: 1
- Status: READY

## Implementation
File: src/mlops/mlflow_manager.py

## Verification
The tracking server returned OK. The champion model was retrieved and loaded successfully as a PyFuncModel.

## Storage
The existing SQLite database and model artifacts are maintained in the project directory.
