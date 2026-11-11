
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(100) PRIMARY KEY,

    step INTEGER NOT NULL,
    transaction_type VARCHAR(30) NOT NULL,
    amount NUMERIC(18, 2) NOT NULL,

    name_orig VARCHAR(100),
    old_balance_orig NUMERIC(18, 2),
    new_balance_orig NUMERIC(18, 2),

    name_dest VARCHAR(100),
    old_balance_dest NUMERIC(18, 2),
    new_balance_dest NUMERIC(18, 2),

    is_fraud SMALLINT CHECK (is_fraud IN (0, 1)),
    is_flagged_fraud SMALLINT CHECK (is_flagged_fraud IN (0, 1)),

    processing_status VARCHAR(30) NOT NULL DEFAULT 'received',

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMPTZ
);


CREATE TABLE IF NOT EXISTS predictions (
    prediction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    transaction_id VARCHAR(100) NOT NULL
        REFERENCES transactions(transaction_id),

    predicted_fraud SMALLINT NOT NULL
        CHECK (predicted_fraud IN (0, 1)),

    fraud_probability NUMERIC(8, 7) NOT NULL
        CHECK (fraud_probability BETWEEN 0 AND 1),

    model_name VARCHAR(150) NOT NULL,
    model_version VARCHAR(50),

    threshold NUMERIC(8, 7),

    predicted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    transaction_id VARCHAR(100) NOT NULL
        REFERENCES transactions(transaction_id),

    prediction_id BIGINT NOT NULL
        REFERENCES predictions(prediction_id),

    severity VARCHAR(20) NOT NULL DEFAULT 'high',

    alert_status VARCHAR(30) NOT NULL DEFAULT 'open',

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMPTZ
);


CREATE INDEX IF NOT EXISTS idx_transactions_status
    ON transactions(processing_status);

CREATE INDEX IF NOT EXISTS idx_transactions_created_at
    ON transactions(created_at);

CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id
    ON predictions(transaction_id);

CREATE INDEX IF NOT EXISTS idx_fraud_alerts_status
    ON fraud_alerts(alert_status);
