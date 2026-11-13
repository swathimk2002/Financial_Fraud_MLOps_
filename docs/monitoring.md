# Monitoring and Observability

## Purpose
Monitoring tracks API performance, prediction activity and changes in transaction data.

## Tools
- Prometheus: collects application metrics.
- Grafana: displays monitoring dashboards.
- Evidently: evaluates data drift.
- PostgreSQL: stores drift evaluation results.

## Implementation
- Drift evaluation: src/monitoring/data_drift.py
- Live collector: src/monitoring/live_drift_collector.py
- Background worker: src/monitoring/live_drift_worker.py

## Dashboard URLs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## API Metrics
Metrics endpoint: http://localhost:8000/metrics

## Verification
The API Prometheus target was UP. Prediction metrics were visible in Grafana. A live drift evaluation was stored in PostgreSQL.

## Drift Detection
The recorded evaluation compared 200,000 reference rows with 500 current rows. No drift was detected in that evaluation.
