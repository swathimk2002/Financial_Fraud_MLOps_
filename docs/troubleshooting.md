# Troubleshooting

## MLflow UI Not Opening
- Check whether the MLflow server is running.
- Verify port 5000.
- Check logs in logs/mlflow-server-error.log.

## FastAPI Not Responding
- Check the financial-fraud-api container.
- Verify port 8000.
- Check the API health endpoint.

## Kafka Connection Failure
- Check the Kafka container.
- Use financial-fraud-kafka:19092 for internal Docker communication.

## PostgreSQL Connection Failure
- Check the PostgreSQL container.
- Verify database configuration and credentials.

## AWS S3 Access Failure
- Verify AWS CLI credentials.
- Check AWS_REGION and AWS_S3_BUCKET.

## General Advice
Check logs and service health before restarting containers. Never delete existing Docker volumes or project databases during troubleshooting.
