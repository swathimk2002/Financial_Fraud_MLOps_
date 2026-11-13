# Data Pipeline

## Overview
The data pipeline loads, validates, preprocesses and prepares transaction data for fraud detection.

## 1. Data Ingestion
Implemented in: src/ingestion/loader.py

Supported formats:
- CSV
- JSON
- JSONL / NDJSON
- Parquet
- Excel

## 2. Data Validation
Implemented in: src/validation/validator.py

Validation checks include:
- Required columns
- Missing values
- Duplicate records
- Negative transaction amounts
- Valid fraud labels

## 3. Preprocessing
Implemented in: src/preprocessing/

- Stratified train, validation and test split.
- 70% training data.
- 15% validation data.
- 15% test data.
- Target column: isFraud.

## 4. Feature Engineering
Implemented in: src/features/engineer.py

Engineered features include:
- amount_to_orig_balance_ratio
- amount_to_dest_balance_ratio

## 5. Data Storage
- Raw data: data/raw/
- Validated data: data/validated/
- Processed data: data/processed/
- Sample data: data/samples/

## Important
The primary model uses pre-transaction features. Post-transaction balance columns are excluded to reduce data leakage.
