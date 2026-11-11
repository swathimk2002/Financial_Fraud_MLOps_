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
- 1,400,000 feature records (200,000 × 7).
- Online retrieval tested successfully.
- Retrieved values matched the source data.

## 6. Configuration Note
`offline=False` was configured to avoid the offline write-back schema error. The historical retrieval capability must be reverified with this final configuration.

## 7. Testing
Existing project test suite: 29 passed.

**Phase 4 implementation progress:** Online feature storage and retrieval verified. Final historical retrieval verification remains pending.