
"""Feast entity and feature definitions for financial transactions."""

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.file_source import FileSource
from feast.data_format import ParquetFormat
from feast.types import String, Int64, Float64
from feast.value_type import ValueType


# --------------------------------------------------
# 1. ENTITY
# --------------------------------------------------
# An entity identifies the transaction to which
# the feature values belong.

transaction = Entity(
    name="transaction",
    join_keys=["transaction_id"],
    value_type=ValueType.STRING,
    description="A uniquely identified PaySim transaction.",
)


# --------------------------------------------------
# 2. OFFLINE DATA SOURCE
# --------------------------------------------------
# This path is relative to the feature_store folder.

transaction_source = FileSource(
    path="../data/processed/features/transaction_features.parquet",
    name="transaction_parquet_source",
    timestamp_field="event_timestamp",
    file_format=ParquetFormat(),
    description="Prepared historical PaySim transaction features.",
)


# --------------------------------------------------
# 3. FEATURE VIEW
# --------------------------------------------------
# A FeatureView groups the features used by our model.

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    schema=[
        Field(name="step", dtype=Int64),
        Field(name="type", dtype=String),
        Field(name="amount", dtype=Float64),
        Field(name="oldbalanceOrg", dtype=Float64),
        Field(name="oldbalanceDest", dtype=Float64),
        Field(
            name="amount_to_orig_balance_ratio",
            dtype=Float64,
        ),
        Field(
            name="amount_to_dest_balance_ratio",
            dtype=Float64,
        ),
    ],
    source=transaction_source,
    online=True,
    offline=False,
    description=(
        "Pre-transaction features for financial fraud detection. "
        "Excludes target labels and post-transaction information."
    ),
    tags={
        "project": "financial_fraud_detection",
        "purpose": "pre_transaction_fraud_detection",
        "dataset": "paysim",
    },
)

