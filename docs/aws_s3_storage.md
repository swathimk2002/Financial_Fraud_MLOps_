
# AWS S3 Storage

## Overview
AWS S3 is used for cloud storage in the Financial Fraud Detection MLOps project.

- Bucket: `financial-fraud-mlops-swathi-2026`
- Region: `ap-south-1`
- Versioning: Enabled
- Encryption: AES256
- Public access: Blocked

## Configuration
Bucket settings are maintained in `.env` using:
- `AWS_S3_BUCKET`
- `AWS_REGION`

AWS credentials are managed through AWS CLI. The `.env` file is excluded from Git.

## Implementation
File: `src/storage/s3_client.py`

Functions:
- `create_s3_client()` â€“ Connects to S3.
- `upload_file()` â€“ Uploads local files.
- `download_file()` â€“ Downloads S3 objects.
- `object_exists()` â€“ Checks object existence.

## Testing
Test file: `tests/test_s3_storage.py`

- S3 tests: 11 passed.
- Latest complete project test suite: 70 passed, 3 warnings.

