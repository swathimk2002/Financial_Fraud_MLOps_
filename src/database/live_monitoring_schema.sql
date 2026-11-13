-- Persistent live feature observations
CREATE TABLE IF NOT EXISTS live_feature_observations (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    batch_id UUID NOT NULL,

    step INTEGER NOT NULL,
    transaction_type VARCHAR(30) NOT NULL,

    amount DOUBLE PRECISION NOT NULL,
    old_balance_orig DOUBLE PRECISION NOT NULL,
    old_balance_dest DOUBLE PRECISION NOT NULL,

    amount_to_orig_balance_ratio DOUBLE PRECISION NOT NULL,
    amount_to_dest_balance_ratio DOUBLE PRECISION NOT NULL,

    collected_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    processed BOOLEAN NOT NULL DEFAULT FALSE
);


-- Persistent live drift evaluation results
CREATE TABLE IF NOT EXISTS drift_evaluations (
    evaluation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    batch_id UUID NOT NULL UNIQUE,

    reference_rows INTEGER NOT NULL,
    current_rows INTEGER NOT NULL,

    drift_detected BOOLEAN NOT NULL,
    drift_share DOUBLE PRECISION NOT NULL,

    feature_results JSONB NOT NULL,

    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- Query optimization indexes
CREATE INDEX IF NOT EXISTS idx_live_observations_batch
    ON live_feature_observations(batch_id);

CREATE INDEX IF NOT EXISTS idx_live_observations_processed
    ON live_feature_observations(processed);

CREATE INDEX IF NOT EXISTS idx_live_observations_collected_at
    ON live_feature_observations(collected_at);

CREATE INDEX IF NOT EXISTS idx_drift_evaluations_evaluated_at
    ON drift_evaluations(evaluated_at);
