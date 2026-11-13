# Phase 4: Feature Store (Feast)

## 1. Purpose
Feast manages transaction features for model training and real-time fraud prediction.

## 2. Configuration
- Feature store: Feast 0.67.0
- Provider: Local
- Registry: SQLite
- Online store: SQLite
- Historical data source: Parquet

## 3. Entity and FeatureView
- Entity: `transaction`
- Join key: `transaction_id`
- FeatureView: `transaction_features`

Seven features:
1. step
2. type
3. amount
4. oldbalanceOrg
5. oldbalanceDest
6. amount_to_orig_balance_ratio
7. amount_to_dest_balance_ratio

## 4. Data Preparation
- 200,000 transactions processed.
- Unique transaction IDs generated.
- Timestamps created from PaySim's step.
- Only seven model features, entity ID and timestamp retained.
- Data saved as Parquet.

## 5. Online Materialization
- SQLite online store populated.
- 200,000 unique transactions.
- 1,400,000 feature records (200,000 Ã— 7).
- Online retrieval tested successfully.
- Retrieved values matched the source data.


## 6. Configuration Note

- FeatureView configuration: `online=True`, `offline=False`.
- The setting was chosen to avoid the earlier Feast offline write-back schema mismatch.
- Historical feature retrieval uses the configured Parquet file source.
- Online features are stored in SQLite.
- Historical retrieval was successfully verified with the final configuration.
- No unnecessary offline write-back was enabled.

## 7. Testing and Verification

**Online store:**
- 200,000 unique transactions materialized.
- 1,400,000 feature records stored.
- Seven features retrieved successfully.
- Retrieved values matched the original Parquet source.

**Historical retrieval:**
- Tested using Feast `get_historical_features()`.
- Five transactions retrieved successfully.
- Seven features returned per transaction.
- Nine columns returned, including entity ID and timestamp.
- Missing values: 0.

**Existing project test suite:**
- Latest complete project test suite: 70 passed, 3 warnings.

**Phase 4 implementation status:** Historical feature retrieval and online feature retrieval successfully verified. Core Feast implementation completed.

