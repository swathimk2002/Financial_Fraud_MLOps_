# PostgreSQL Database

## Purpose
PostgreSQL stores transaction, prediction, fraud alert and monitoring records.

## Configuration
- Docker container: financial-fraud-postgres
- Host port: 5433
- Container port: 5432
- Database: financial_fraud
- Database user: fraud_admin

## Tables
- transactions
- predictions
- fraud_alerts
- live_feature_observations
- drift_evaluations

## Implementation
- Connection: src/database/connection.py
- Operations: src/database/operations.py
- Monitoring: src/database/live_monitoring_operations.py

## Verification
Database connectivity and table operations were verified. Transaction, prediction, alert and monitoring records were present.

## Security
Database credentials are maintained outside source code.
